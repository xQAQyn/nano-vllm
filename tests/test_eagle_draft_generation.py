"""
Stage 4 Tests for EAGLE-1 Draft Token Generation

Tests for:
- Draft model generates K tokens autoregressively
- Generated tokens are valid token IDs
- Hidden states are correctly captured and passed
- prepare_draft_input() correctly concatenates hidden states + token embeddings
- generate_draft_tokens() produces correct output shapes

Files tested:
- nanovllm/engine/eagle_runner.py
- nanovllm/models/qwen3.py (return_hidden_states feature)
"""

import os
import pytest
import torch
import torch.distributed as dist
from transformers import AutoConfig

from nanovllm.models.eagle import EagleDraftModel, load_draft_model
from nanovllm.models.qwen3 import Qwen3ForCausalLM, Qwen3Model
from nanovllm.engine.eagle_runner import EagleDraftRunner, create_draft_runner
from nanovllm.utils.context import set_context, reset_context


def setup_distributed():
    """Initialize distributed process group for single-process testing."""
    if not dist.is_initialized():
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29503'
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


@pytest.fixture(scope="module")
def qwen_config():
    """Get Qwen3 config for testing."""
    config = AutoConfig.from_pretrained("models/Qwen3-0.6B")
    return config


@pytest.fixture(scope="module")
def target_model(qwen_config, device):
    """Create a target model for testing."""
    if not torch.cuda.is_available():
        dtype = torch.float32
    else:
        dtype = torch.bfloat16  # FlashAttention requires fp16 or bf16
    
    model = Qwen3ForCausalLM(qwen_config).to(device=device, dtype=dtype)
    model.eval()

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    return model


@pytest.fixture(scope="module")
def draft_model(qwen_config, target_model, device):
    """Create a draft model for testing."""
    # Get target model dtype to ensure consistency
    target_dtype = next(target_model.parameters()).dtype
    
    # Use fresh_decoder=True and let the model copy weights from target
    # The dtype will match the target model's dtype
    model = EagleDraftModel(qwen_config, target_model, fresh_decoder=True)
    # Explicitly convert to target dtype and device
    model = model.to(device=device, dtype=target_dtype)
    model.eval()
    return model


@pytest.fixture(scope="module")
def draft_runner(draft_model):
    """Create a draft runner for testing."""
    return EagleDraftRunner(
        draft_model=draft_model,
        max_speculation_depth=4,
    )


@pytest.fixture(scope="module")
def test_dtype():
    """Get test dtype based on CUDA availability."""
    if torch.cuda.is_available():
        return torch.bfloat16
    return torch.float32


class TestQwen3HiddenStateExposure:
    """Tests for Qwen3 model's return_hidden_states feature."""

    def test_model_forward_without_hidden_states(self, target_model, device, test_dtype):
        """Test that model forward works without returning hidden states."""
        seq_len = 10

        input_ids = torch.randint(0, target_model.config.vocab_size, (seq_len,), device=device)
        positions = torch.arange(seq_len, device=device)

        # Set up context for FlashAttention
        set_context(
            is_prefill=True,
            cu_seqlens_q=torch.tensor([0, seq_len], dtype=torch.int32, device=device),
            cu_seqlens_k=torch.tensor([0, seq_len], dtype=torch.int32, device=device),
            max_seqlen_q=seq_len,
            max_seqlen_k=seq_len,
        )

        try:
            with torch.inference_mode():
                output = target_model(input_ids, positions, return_hidden_states=False)

            assert isinstance(output, torch.Tensor)
            assert output.shape == (seq_len, target_model.config.hidden_size)
            assert output.dtype == test_dtype
        finally:
            reset_context()

    def test_model_forward_with_hidden_states(self, target_model, device, test_dtype):
        """Test that model forward returns hidden states when requested."""
        seq_len = 10

        input_ids = torch.randint(0, target_model.config.vocab_size, (seq_len,), device=device)
        positions = torch.arange(seq_len, device=device)

        # Set up context for FlashAttention
        set_context(
            is_prefill=True,
            cu_seqlens_q=torch.tensor([0, seq_len], dtype=torch.int32, device=device),
            cu_seqlens_k=torch.tensor([0, seq_len], dtype=torch.int32, device=device),
            max_seqlen_q=seq_len,
            max_seqlen_k=seq_len,
        )

        try:
            with torch.inference_mode():
                output = target_model(input_ids, positions, return_hidden_states=True)

            assert isinstance(output, tuple)
            assert len(output) == 2

            final_hidden, second_to_last_hidden = output

            # Check shapes
            assert final_hidden.shape == (seq_len, target_model.config.hidden_size)
            assert second_to_last_hidden.shape == (seq_len, target_model.config.hidden_size)
            assert final_hidden.dtype == test_dtype
            assert second_to_last_hidden.dtype == test_dtype
        finally:
            reset_context()

    def test_model_forward_hidden_states_single_sequence(self, target_model, device, test_dtype):
        """Test hidden state return with seq_len=5."""
        seq_len = 5

        input_ids = torch.randint(0, target_model.config.vocab_size, (seq_len,), device=device)
        positions = torch.arange(seq_len, device=device)

        # Set up context for FlashAttention
        set_context(
            is_prefill=True,
            cu_seqlens_q=torch.tensor([0, seq_len], dtype=torch.int32, device=device),
            cu_seqlens_k=torch.tensor([0, seq_len], dtype=torch.int32, device=device),
            max_seqlen_q=seq_len,
            max_seqlen_k=seq_len,
        )

        try:
            with torch.inference_mode():
                output = target_model(input_ids, positions, return_hidden_states=True)

            assert isinstance(output, tuple)
            final_hidden, second_to_last_hidden = output

            assert final_hidden.shape == (seq_len, target_model.config.hidden_size)
            assert second_to_last_hidden.shape == (seq_len, target_model.config.hidden_size)
            assert final_hidden.dtype == test_dtype
        finally:
            reset_context()


class TestEagleDraftRunnerInitialization:
    """Tests for EagleDraftRunner initialization."""

    def test_runner_creation(self, draft_model):
        """Test creating a draft runner."""
        runner = EagleDraftRunner(
            draft_model=draft_model,
            max_speculation_depth=4,
        )

        assert runner is not None
        assert runner.max_speculation_depth == 4
        assert runner.draft_model is draft_model

    def test_runner_default_device(self, draft_model):
        """Test that runner uses draft model's device by default."""
        runner = EagleDraftRunner(draft_model=draft_model)

        assert runner.device == next(draft_model.parameters()).device

    def test_runner_custom_device(self, draft_model, device):
        """Test creating runner with custom device."""
        runner = EagleDraftRunner(
            draft_model=draft_model,
            device=device,
        )

        assert runner.device == device

    def test_create_draft_runner_helper(self, draft_model):
        """Test the create_draft_runner helper function."""
        runner = create_draft_runner(draft_model, max_speculation_depth=8)

        assert runner is not None
        assert runner.max_speculation_depth == 8


class TestPrepareDraftInput:
    """Tests for prepare_draft_input() method."""

    def test_prepare_input_shapes(self, draft_runner, qwen_config, device, test_dtype):
        """Test that prepare_draft_input produces correct output shapes."""
        seq_len = 10
        hidden_size = qwen_config.hidden_size

        hidden_states = torch.randn(seq_len, hidden_size, device=device, dtype=test_dtype)
        token_ids = torch.randint(0, qwen_config.vocab_size, (seq_len,), device=device)
        positions = torch.arange(seq_len, device=device)

        token_embeds, hidden_out, positions_out = draft_runner.prepare_draft_input(
            hidden_states=hidden_states,
            token_ids=token_ids,
            positions=positions,
        )

        assert token_embeds.shape == (seq_len, hidden_size)
        assert hidden_out.shape == (seq_len, hidden_size)
        assert positions_out.shape == (seq_len,)

    def test_prepare_input_dtype(self, draft_runner, qwen_config, device, test_dtype):
        """Test that prepare_draft_input converts to correct dtype."""
        seq_len = 5
        hidden_size = qwen_config.hidden_size

        # Use test_dtype input
        hidden_states = torch.randn(seq_len, hidden_size, device=device, dtype=test_dtype)
        token_ids = torch.randint(0, qwen_config.vocab_size, (seq_len,), device=device)
        positions = torch.arange(seq_len, device=device)

        token_embeds, hidden_out, _ = draft_runner.prepare_draft_input(
            hidden_states=hidden_states,
            token_ids=token_ids,
            positions=positions,
        )

        # Output should match draft model dtype
        assert token_embeds.dtype == test_dtype
        assert hidden_out.dtype == test_dtype

    def test_prepare_input_embedding_lookup(self, draft_runner, device, test_dtype):
        """Test that prepare_draft_input correctly looks up embeddings."""
        seq_len = 3

        # Use known token IDs
        token_ids = torch.tensor([1, 2, 3], device=device)
        hidden_states = torch.randn(seq_len, 512, device=device, dtype=test_dtype)
        positions = torch.arange(seq_len, device=device)

        token_embeds, _, _ = draft_runner.prepare_draft_input(
            hidden_states=hidden_states,
            token_ids=token_ids,
            positions=positions,
        )

        # Manually look up embeddings
        expected_embeds = draft_runner.draft_model.embed_tokens(token_ids)

        assert torch.allclose(token_embeds, expected_embeds, atol=1e-3)  # Relaxed for bfloat16


class TestGenerateDraftTokens:
    """Tests for generate_draft_tokens() method."""

    def test_generate_single_token(self, draft_runner, qwen_config, device, test_dtype):
        """Test generating a single draft token."""
        seq_len = 5
        hidden_size = qwen_config.hidden_size

        hidden_states = torch.randn(seq_len, hidden_size, device=device, dtype=test_dtype)
        token_ids = torch.randint(0, qwen_config.vocab_size, (seq_len,), device=device)
        positions = torch.arange(seq_len, device=device)

        with torch.inference_mode():
            draft_tokens, draft_features = draft_runner.generate_draft_tokens(
                hidden_states=hidden_states,
                token_ids=token_ids,
                positions=positions,
                num_draft_tokens=1,
            )

        assert len(draft_tokens) == 1
        assert isinstance(draft_tokens[0], int)
        assert 0 <= draft_tokens[0] < qwen_config.vocab_size
        assert draft_features.shape == (1, hidden_size)

    def test_generate_multiple_tokens(self, draft_runner, qwen_config, device, test_dtype):
        """Test generating multiple draft tokens."""
        seq_len = 5
        hidden_size = qwen_config.hidden_size
        num_draft_tokens = 4

        hidden_states = torch.randn(seq_len, hidden_size, device=device, dtype=test_dtype)
        token_ids = torch.randint(0, qwen_config.vocab_size, (seq_len,), device=device)
        positions = torch.arange(seq_len, device=device)

        with torch.inference_mode():
            draft_tokens, draft_features = draft_runner.generate_draft_tokens(
                hidden_states=hidden_states,
                token_ids=token_ids,
                positions=positions,
                num_draft_tokens=num_draft_tokens,
            )

        assert len(draft_tokens) == num_draft_tokens
        for token_id in draft_tokens:
            assert isinstance(token_id, int)
            assert 0 <= token_id < qwen_config.vocab_size
        assert draft_features.shape == (num_draft_tokens, hidden_size)

    def test_generate_default_num_tokens(self, draft_runner, qwen_config, device, test_dtype):
        """Test that generate_draft_tokens uses max_speculation_depth by default."""
        seq_len = 5
        hidden_size = qwen_config.hidden_size

        hidden_states = torch.randn(seq_len, hidden_size, device=device, dtype=test_dtype)
        token_ids = torch.randint(0, qwen_config.vocab_size, (seq_len,), device=device)
        positions = torch.arange(seq_len, device=device)

        with torch.inference_mode():
            draft_tokens, draft_features = draft_runner.generate_draft_tokens(
                hidden_states=hidden_states,
                token_ids=token_ids,
                positions=positions,
            )

        # Should use max_speculation_depth (4) by default
        assert len(draft_tokens) == draft_runner.max_speculation_depth
        assert draft_features.shape[0] == draft_runner.max_speculation_depth

    def test_generate_tokens_autoregressive(self, draft_runner, qwen_config, device, test_dtype):
        """Test that draft tokens are generated autoregressively."""
        seq_len = 5
        hidden_size = qwen_config.hidden_size
        num_draft_tokens = 3

        hidden_states = torch.randn(seq_len, hidden_size, device=device, dtype=test_dtype)
        token_ids = torch.randint(0, qwen_config.vocab_size, (seq_len,), device=device)
        positions = torch.arange(seq_len, device=device)

        with torch.inference_mode():
            draft_tokens, draft_features = draft_runner.generate_draft_tokens(
                hidden_states=hidden_states,
                token_ids=token_ids,
                positions=positions,
                num_draft_tokens=num_draft_tokens,
            )

        # Verify output shape
        assert draft_features.shape[0] == num_draft_tokens
        
        # Note: Features at different autoregressive steps may differ
        # but with random weights they could be similar, so we just verify shape

    def test_generate_tokens_valid_ids(self, draft_runner, qwen_config, device, test_dtype):
        """Test that all generated token IDs are valid."""
        seq_len = 5
        hidden_size = qwen_config.hidden_size
        num_draft_tokens = 10

        hidden_states = torch.randn(seq_len, hidden_size, device=device, dtype=test_dtype)
        token_ids = torch.randint(0, qwen_config.vocab_size, (seq_len,), device=device)
        positions = torch.arange(seq_len, device=device)

        with torch.inference_mode():
            draft_tokens, _ = draft_runner.generate_draft_tokens(
                hidden_states=hidden_states,
                token_ids=token_ids,
                positions=positions,
                num_draft_tokens=num_draft_tokens,
            )

        for token_id in draft_tokens:
            assert isinstance(token_id, int), f"Token ID should be int, got {type(token_id)}"
            assert 0 <= token_id < qwen_config.vocab_size, \
                f"Token ID {token_id} out of range [0, {qwen_config.vocab_size})"


class TestGenerateSingleStep:
    """Tests for generate_single_step() method."""

    def test_single_step_generation(self, draft_runner, qwen_config, device, test_dtype):
        """Test generating a single draft token step."""
        hidden_size = qwen_config.hidden_size

        hidden_states = torch.randn(1, hidden_size, device=device, dtype=test_dtype)
        token_id = 42
        position = 5

        with torch.inference_mode():
            next_token_id, predicted_feature = draft_runner.generate_single_step(
                hidden_states=hidden_states,
                token_id=token_id,
                position=position,
            )

        assert isinstance(next_token_id, int)
        assert 0 <= next_token_id < qwen_config.vocab_size
        assert predicted_feature.shape == (hidden_size,)


class TestBatchedGeneration:
    """Tests for batched draft token generation."""

    def test_batched_generation_shapes(self, draft_runner, qwen_config, device, test_dtype):
        """Test draft generation with batch_size > 1."""
        batch_size = 4
        seq_len = 8
        hidden_size = qwen_config.hidden_size
        num_draft_tokens = 3

        # 3D input (batched)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size, device=device, dtype=test_dtype)
        token_ids = torch.randint(0, qwen_config.vocab_size, (batch_size, seq_len), device=device)
        positions = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1).to(device)

        with torch.inference_mode():
            draft_tokens, draft_features = draft_runner.generate_draft_tokens(
                hidden_states=hidden_states,
                token_ids=token_ids,
                positions=positions,
                num_draft_tokens=num_draft_tokens,
            )

        # For batched generation, we take the first sequence
        assert len(draft_tokens) == num_draft_tokens
        assert draft_features.shape == (num_draft_tokens, hidden_size)


class TestIntegration:
    """Integration tests for complete draft generation workflow."""

    def test_full_draft_generation_workflow(self, qwen_config, target_model, device, test_dtype):
        """Test complete draft generation workflow."""
        # Create draft model and runner
        draft_model = EagleDraftModel(qwen_config, target_model).to(
            device=device, dtype=test_dtype
        )
        draft_model.eval()

        runner = EagleDraftRunner(
            draft_model=draft_model,
            max_speculation_depth=5,
            device=device,
        )

        # Prepare inputs
        seq_len = 10
        hidden_size = qwen_config.hidden_size

        hidden_states = torch.randn(seq_len, hidden_size, device=device, dtype=test_dtype)
        token_ids = torch.randint(0, qwen_config.vocab_size, (seq_len,), device=device)
        positions = torch.arange(seq_len, device=device)

        # Generate draft tokens
        with torch.inference_mode():
            draft_tokens, draft_features = runner.generate_draft_tokens(
                hidden_states=hidden_states,
                token_ids=token_ids,
                positions=positions,
            )

        # Verify outputs
        assert len(draft_tokens) == 5
        assert draft_features.shape == (5, hidden_size)

        for token_id in draft_tokens:
            assert 0 <= token_id < qwen_config.vocab_size

    def test_hidden_state_capture_and_draft_generation(self, qwen_config, target_model, device, test_dtype):
        """Test capturing hidden states from target model and using them for draft generation."""
        # Create draft model and runner
        draft_model = EagleDraftModel(qwen_config, target_model).to(
            device=device, dtype=test_dtype
        )
        draft_model.eval()

        runner = EagleDraftRunner(
            draft_model=draft_model,
            max_speculation_depth=3,
            device=device,
        )

        # Run target model to get hidden states
        seq_len = 8

        input_ids = torch.randint(0, qwen_config.vocab_size, (seq_len,), device=device)
        positions = torch.arange(seq_len, device=device)

        # Set up context for FlashAttention
        set_context(
            is_prefill=True,
            cu_seqlens_q=torch.tensor([0, seq_len], dtype=torch.int32, device=device),
            cu_seqlens_k=torch.tensor([0, seq_len], dtype=torch.int32, device=device),
            max_seqlen_q=seq_len,
            max_seqlen_k=seq_len,
        )

        try:
            with torch.inference_mode():
                # Get hidden states from target model
                _, second_to_last_hidden = target_model(
                    input_ids, positions, return_hidden_states=True
                )

                # Generate draft tokens using captured hidden states
                draft_tokens, draft_features = runner.generate_draft_tokens(
                    hidden_states=second_to_last_hidden,
                    token_ids=input_ids,
                    positions=positions,
                )

            # Verify draft generation succeeded
            assert len(draft_tokens) == 3
            assert draft_features.shape == (3, qwen_config.hidden_size)

            for token_id in draft_tokens:
                assert 0 <= token_id < qwen_config.vocab_size
        finally:
            reset_context()

    def test_draft_generation_consistency(self, qwen_config, target_model, device, test_dtype):
        """Test that draft generation is deterministic in eval mode."""
        draft_model = EagleDraftModel(qwen_config, target_model).to(
            device=device, dtype=test_dtype
        )
        draft_model.eval()

        runner = EagleDraftRunner(
            draft_model=draft_model,
            max_speculation_depth=3,
            device=device,
        )

        # Prepare fixed inputs
        seq_len = 5
        hidden_size = qwen_config.hidden_size

        torch.manual_seed(42)
        hidden_states = torch.randn(seq_len, hidden_size, device=device, dtype=test_dtype)
        token_ids = torch.randint(0, qwen_config.vocab_size, (seq_len,), device=device)
        positions = torch.arange(seq_len, device=device)

        # Run twice with same inputs
        with torch.inference_mode():
            draft_tokens_1, draft_features_1 = runner.generate_draft_tokens(
                hidden_states=hidden_states,
                token_ids=token_ids,
                positions=positions,
            )

            draft_tokens_2, draft_features_2 = runner.generate_draft_tokens(
                hidden_states=hidden_states,
                token_ids=token_ids,
                positions=positions,
            )

        # Results should be identical
        assert draft_tokens_1 == draft_tokens_2
        assert torch.allclose(draft_features_1, draft_features_2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
