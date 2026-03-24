"""
Stage 1 Tests for EAGLE-1 Draft Model

Tests for:
- Draft model loads successfully with target model weights
- Draft model forward pass produces feature predictions of correct shape
- Trainable parameter count matches expected (~0.24B for 0.6B model proportionally)
"""

import os
import pytest
import torch
import torch.distributed as dist
from transformers import AutoConfig

from nanovllm.models.eagle import EagleDraftModel, EagleFusionLayer, load_eagle_draft_model


def setup_distributed():
    """Initialize distributed process group for single-process testing."""
    if not dist.is_initialized():
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29501'
        os.environ['RANK'] = '0'
        os.environ['WORLD_SIZE'] = '1'
        dist.init_process_group("gloo", rank=0, world_size=1)


def teardown_distributed():
    """Destroy distributed process group."""
    if dist.is_initialized():
        dist.destroy_process_group()


@pytest.fixture(scope="module", autouse=True)
def distributed_setup():
    """Setup and teardown distributed environment for all tests."""
    setup_distributed()
    yield
    teardown_distributed()


@pytest.fixture(scope="module")
def device():
    """Get CUDA device if available, otherwise CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class TestEagleFusionLayer:
    """Tests for the EagleFusionLayer component."""

    def test_fusion_layer_forward_shape(self, device):
        """Test that fusion layer produces correct output shape."""
        hidden_size = 512
        fusion_layer = EagleFusionLayer(hidden_size).to(device)
        
        batch_size = 4
        token_embeds = torch.randn(batch_size, hidden_size, device=device)
        hidden_states = torch.randn(batch_size, hidden_size, device=device)
        
        output = fusion_layer(token_embeds, hidden_states)
        
        assert output.shape == (batch_size, hidden_size), \
            f"Expected output shape {(batch_size, hidden_size)}, got {output.shape}"

    def test_fusion_layer_concatenation(self, device):
        """Test that fusion layer correctly concatenates inputs."""
        hidden_size = 256
        fusion_layer = EagleFusionLayer(hidden_size).to(device)
        
        batch_size = 2
        token_embeds = torch.ones(batch_size, hidden_size, device=device)
        hidden_states = torch.zeros(batch_size, hidden_size, device=device)
        
        output = fusion_layer(token_embeds, hidden_states)
        
        # Output should not be all zeros or all ones (should be a mix)
        assert not torch.all(output == 0), "Output should not be all zeros"
        assert not torch.all(output == 1), "Output should not be all ones"


class TestEagleDraftModel:
    """Tests for the EagleDraftModel class."""

    @pytest.fixture(scope="class")
    def qwen_config(self):
        """Get Qwen3 config for testing."""
        config = AutoConfig.from_pretrained("models/Qwen3-0.6B")
        return config

    @pytest.fixture(scope="class")
    def target_model(self, qwen_config, device):
        """Create a target model for testing."""
        from nanovllm.models.qwen3 import Qwen3ForCausalLM
        model = Qwen3ForCausalLM(qwen_config).to(device=device, dtype=qwen_config.torch_dtype)
        model.eval()
        return model

    @pytest.fixture(scope="class")
    def draft_model(self, qwen_config, target_model, device):
        """Create a draft model for testing."""
        return EagleDraftModel(qwen_config, target_model).to(device=device, dtype=qwen_config.torch_dtype)

    def test_draft_model_creation(self, qwen_config, target_model):
        """Test that draft model can be created with target model."""
        draft_model = EagleDraftModel(qwen_config, target_model)
        
        assert draft_model is not None
        assert hasattr(draft_model, 'fusion_layer')
        assert hasattr(draft_model, 'decoder_layer')
        assert hasattr(draft_model, 'embed_tokens')
        assert hasattr(draft_model, 'lm_head')

    @pytest.mark.skip(reason="Requires full KV cache setup - tested in integration tests")
    def test_draft_model_forward_shape(self, draft_model, qwen_config, device):
        """Test that draft model forward pass produces correct output shape."""
        batch_size = 4
        hidden_size = qwen_config.hidden_size
        dtype = qwen_config.torch_dtype
        
        token_ids = torch.randint(0, qwen_config.vocab_size, (batch_size,), device=device)
        hidden_states = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
        positions = torch.arange(batch_size, device=device)
        
        with torch.inference_mode():
            output = draft_model(token_ids, hidden_states, positions)
        
        assert output.shape == (batch_size, hidden_size), \
            f"Expected output shape {(batch_size, hidden_size)}, got {output.shape}"

    def test_draft_model_compute_logits_shape(self, draft_model, qwen_config, device):
        """Test that compute_logits produces correct output shape."""
        batch_size = 4
        hidden_size = qwen_config.hidden_size
        vocab_size = qwen_config.vocab_size
        dtype = qwen_config.torch_dtype
        
        hidden_states = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
        
        with torch.inference_mode():
            logits = draft_model.compute_logits(hidden_states)
        
        assert logits.shape == (batch_size, vocab_size), \
            f"Expected logits shape {(batch_size, vocab_size)}, got {logits.shape}"

    def test_draft_model_trainable_parameters(self, draft_model, qwen_config):
        """Test that draft model has expected number of trainable parameters."""
        num_params = draft_model.count_trainable_parameters()
        
        hidden_size = qwen_config.hidden_size
        intermediate_size = qwen_config.intermediate_size
        
        estimated_min = int(0.5 * hidden_size * hidden_size * 4)
        estimated_max = int(hidden_size * (2 * hidden_size + 3 * intermediate_size) * 2)
        
        assert num_params > estimated_min, f"Too few trainable parameters: {num_params} < {estimated_min}"
        assert num_params < estimated_max, f"Too many trainable parameters: {num_params} > {estimated_max}"
        
        print(f"Draft model trainable parameters: {num_params:,}")

    def test_draft_model_embedding_reuse(self, draft_model, target_model):
        """Test that draft model reuses target model's embedding."""
        target_embed = None
        for module in target_model.modules():
            if hasattr(module, 'embed_tokens'):
                target_embed = module.embed_tokens
                break
        
        if target_embed is not None:
            assert draft_model.embed_tokens is target_embed or \
                   torch.allclose(draft_model.embed_tokens.weight, target_embed.weight), \
                   "Draft model should reuse or copy target model's embedding"

    def test_draft_model_lm_head_reuse(self, draft_model, target_model):
        """Test that draft model reuses target model's LM head."""
        target_lm_head = None
        for module in target_model.modules():
            if hasattr(module, 'lm_head'):
                target_lm_head = module.lm_head
                break
        
        if target_lm_head is not None:
            assert draft_model.lm_head is target_lm_head or \
                   torch.allclose(draft_model.lm_head.weight, target_lm_head.weight), \
                   "Draft model should reuse or copy target model's LM head"

    @pytest.mark.skip(reason="Requires full KV cache setup - tested in integration tests")
    def test_draft_model_autoregressive_step(self, draft_model, qwen_config, device):
        """Test a simulated autoregressive decoding step."""
        batch_size = 2
        seq_len = 5
        hidden_size = qwen_config.hidden_size
        dtype = qwen_config.torch_dtype
        
        token_ids = torch.randint(0, qwen_config.vocab_size, (batch_size,), device=device)
        hidden_states = torch.randn(batch_size, hidden_size, device=device, dtype=dtype)
        positions = torch.zeros(batch_size, dtype=torch.long, device=device)
        
        generated_tokens = []
        for i in range(seq_len):
            with torch.inference_mode():
                output = draft_model(token_ids, hidden_states, positions)
                logits = draft_model.compute_logits(output)
            
            next_token = logits.argmax(dim=-1)
            generated_tokens.append(next_token)
            
            token_ids = next_token.clone()
            hidden_states = output.clone()
            positions = positions + 1
        
        assert len(generated_tokens) == seq_len
        for token in generated_tokens:
            assert token.shape == (batch_size,)
            assert torch.all(token >= 0) and torch.all(token < qwen_config.vocab_size)


class TestLoadEagleDraftModel:
    """Tests for the load_eagle_draft_model function."""

    @pytest.fixture(scope="class")
    def qwen_config(self):
        """Get Qwen3 config for testing."""
        config = AutoConfig.from_pretrained("models/Qwen3-0.6B")
        return config

    @pytest.fixture(scope="class")
    def target_model(self, qwen_config, device):
        """Create a target model for testing."""
        from nanovllm.models.qwen3 import Qwen3ForCausalLM
        model = Qwen3ForCausalLM(qwen_config).to(device=device, dtype=qwen_config.torch_dtype)
        model.eval()
        return model

    def test_load_draft_model_without_pretrained(self, qwen_config, target_model):
        """Test loading draft model without pre-trained weights."""
        draft_model = load_eagle_draft_model(qwen_config, target_model, draft_model_path=None)
        
        assert draft_model is not None
        assert isinstance(draft_model, EagleDraftModel)

    def test_load_draft_model_structure(self, qwen_config, target_model):
        """Test that loaded draft model has correct structure."""
        draft_model = load_eagle_draft_model(qwen_config, target_model)
        
        assert hasattr(draft_model, 'fusion_layer')
        assert hasattr(draft_model, 'decoder_layer')
        assert hasattr(draft_model, 'embed_tokens')
        assert hasattr(draft_model, 'lm_head')
        assert hasattr(draft_model, 'norm')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
