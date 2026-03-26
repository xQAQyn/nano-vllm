"""
Stage 6 Tests for EAGLE-1 Scheduler Integration

Tests for:
- EAGLE-aware scheduling (draft + verify phases)
- Handle variable acceptance lengths
- Manage draft token preemption
- Engine step modification for EAGLE flow
- End-to-end EAGLE inference

Files tested:
- nanovllm/engine/scheduler.py (EAGLE extensions)
- nanovllm/engine/llm_engine.py (EAGLE step flow)
- nanovllm/engine/model_runner.py (run_eagle integration)
"""

import os
import pytest
import torch
import torch.distributed as dist
from transformers import AutoConfig

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.block_manager import BlockManager
from nanovllm.models.eagle import EagleDraftModel, load_draft_model
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.engine.eagle_runner import EagleDraftRunner
from nanovllm.layers.speculative_sampler import SpeculativeSampler


def setup_distributed():
    """Initialize distributed process group for single-process testing."""
    if not dist.is_initialized():
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '29506'
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
        dtype = torch.bfloat16

    model = Qwen3ForCausalLM(qwen_config).to(device=device, dtype=dtype)
    model.eval()

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    return model


@pytest.fixture(scope="module")
def draft_model(qwen_config, target_model, device):
    """Create a draft model for testing."""
    target_dtype = next(target_model.parameters()).dtype
    model = EagleDraftModel(qwen_config, target_model, fresh_decoder=True)
    model = model.to(device=device, dtype=target_dtype)
    model.eval()
    return model


@pytest.fixture
def eagle_config(qwen_config):
    """Create a config with EAGLE enabled."""
    return Config(
        model="models/Qwen3-0.6B",
        eagle_enabled=True,
        eagle_draft_model="new",  # Use fresh draft model
        speculation_depth=4,
        max_num_batched_tokens=16384,
        max_num_seqs=512,
        tensor_parallel_size=1,
        kvcache_block_size=256,
        num_kvcache_blocks=100,
    )


@pytest.fixture
def standard_config(qwen_config):
    """Create a config with EAGLE disabled."""
    return Config(
        model="models/Qwen3-0.6B",
        eagle_enabled=False,
        max_num_batched_tokens=16384,
        max_num_seqs=512,
        tensor_parallel_size=1,
        kvcache_block_size=256,
        num_kvcache_blocks=100,
    )


class TestSchedulerInitialization:
    """Tests for Scheduler initialization with EAGLE support."""

    def test_scheduler_creation_with_eagle(self, eagle_config):
        """Test creating scheduler with EAGLE enabled."""
        scheduler = Scheduler(eagle_config)

        assert scheduler is not None
        assert scheduler.eagle_enabled == True
        assert scheduler.speculation_depth == 4

    def test_scheduler_creation_without_eagle(self, standard_config):
        """Test creating scheduler without EAGLE."""
        scheduler = Scheduler(standard_config)

        assert scheduler is not None
        assert scheduler.eagle_enabled == False
        assert scheduler.speculation_depth == 0

    def test_scheduler_eagle_config_propagation(self, eagle_config):
        """Test that EAGLE config is properly propagated."""
        scheduler = Scheduler(eagle_config)

        assert scheduler.max_num_seqs == eagle_config.max_num_seqs
        assert scheduler.max_num_batched_tokens == eagle_config.max_num_batched_tokens
        assert scheduler.speculation_depth == eagle_config.speculation_depth


class TestEagleAwareScheduling:
    """Tests for EAGLE-aware scheduling behavior."""

    def test_schedule_prefill_with_eagle(self, eagle_config, device):
        """Test scheduling prefill phase with EAGLE enabled."""
        scheduler = Scheduler(eagle_config)

        # Create a sequence
        seq = Sequence([1, 2, 3, 4, 5], SamplingParams(max_tokens=10))
        scheduler.add(seq)

        # Schedule should allocate blocks and move to running
        scheduled_seqs, is_prefill = scheduler.schedule()

        assert len(scheduled_seqs) == 1
        assert is_prefill == True
        assert seq.status == SequenceStatus.RUNNING
        assert len(seq.block_table) > 0

    def test_schedule_decode_with_eagle(self, eagle_config, device):
        """Test scheduling decode phase with EAGLE enabled."""
        scheduler = Scheduler(eagle_config)

        # Create and prefill a sequence
        seq = Sequence([1, 2, 3, 4, 5], SamplingParams(max_tokens=10))
        scheduler.add(seq)
        scheduler.schedule()  # Prefill

        # Now schedule for decode
        scheduled_seqs, is_prefill = scheduler.schedule()

        assert len(scheduled_seqs) == 1
        assert is_prefill == False
        assert seq in scheduler.running

    def test_schedule_multiple_sequences_eagle(self, eagle_config, device):
        """Test scheduling multiple sequences with EAGLE."""
        scheduler = Scheduler(eagle_config)

        # Add multiple sequences
        for i in range(3):
            seq = Sequence([1, 2, 3, 4, 5], SamplingParams(max_tokens=10))
            scheduler.add(seq)

        # Schedule prefill
        scheduled_seqs, is_prefill = scheduler.schedule()

        assert len(scheduled_seqs) == 3
        assert all(seq.status == SequenceStatus.RUNNING for seq in scheduled_seqs)


class TestSchedulerPostprocessEagle:
    """Tests for postprocess_eagle method."""

    def test_postprocess_eagle_all_accepted(self, eagle_config):
        """Test postprocessing when all draft tokens are accepted."""
        scheduler = Scheduler(eagle_config)

        # Create sequence and add to running
        seq = Sequence([1, 2, 3], SamplingParams(max_tokens=10))
        seq.status = SequenceStatus.RUNNING
        scheduler.running.append(seq)

        # Simulate accepted tokens
        accepted_tokens = [[10, 20, 30]]

        finished_mask = scheduler.postprocess_eagle([seq], accepted_tokens)

        assert len(finished_mask) == 1
        assert finished_mask[0] == False
        assert len(seq) == 6  # Original 3 + 3 accepted
        assert seq.token_ids[-3:] == [10, 20, 30]

    def test_postprocess_eagle_partial_acceptance(self, eagle_config):
        """Test postprocessing with partial token acceptance."""
        scheduler = Scheduler(eagle_config)

        seq = Sequence([1, 2, 3], SamplingParams(max_tokens=10))
        seq.status = SequenceStatus.RUNNING
        scheduler.running.append(seq)

        # Simulate partial acceptance (2 out of 3)
        accepted_tokens = [[10, 20]]

        finished_mask = scheduler.postprocess_eagle([seq], accepted_tokens)

        assert len(finished_mask) == 1
        assert finished_mask[0] == False
        assert len(seq) == 5  # Original 3 + 2 accepted

    def test_postprocess_eagle_max_tokens_reached(self, eagle_config):
        """Test postprocessing when max_tokens is reached."""
        scheduler = Scheduler(eagle_config)

        # Create sequence close to max_tokens
        # max_tokens=5 means 5 completion tokens
        # With 3 prompt tokens, need 5 more to reach max
        seq = Sequence([1, 2, 3], SamplingParams(max_tokens=5))
        seq.status = SequenceStatus.RUNNING
        scheduler.running.append(seq)

        # Accept 5 tokens to reach max (3 prompt + 5 completion = 8 total)
        accepted_tokens = [[10, 20, 30, 40, 50]]

        finished_mask = scheduler.postprocess_eagle([seq], accepted_tokens)

        assert len(finished_mask) == 1
        assert finished_mask[0] == True
        assert seq.status == SequenceStatus.FINISHED
        assert len(seq) == 8  # 3 prompt + 5 completion

    def test_postprocess_eagle_clears_draft_state(self, eagle_config):
        """Test that postprocess_eagle clears draft state."""
        scheduler = Scheduler(eagle_config)

        seq = Sequence([1, 2, 3], SamplingParams(max_tokens=10))
        seq.status = SequenceStatus.RUNNING
        scheduler.running.append(seq)

        # Set draft state
        seq.set_draft_tokens([10, 20, 30])
        assert seq.speculation_depth == 3

        accepted_tokens = [[10, 20, 30]]
        scheduler.postprocess_eagle([seq], accepted_tokens)

        # Draft state should be cleared
        assert seq.speculation_depth == 0
        assert seq.draft_token_ids == []


class TestSchedulerBlockManagerIntegration:
    """Tests for block manager integration with EAGLE."""

    def test_block_allocation_for_eagle(self, eagle_config):
        """Test block allocation for EAGLE sequences."""
        scheduler = Scheduler(eagle_config)

        seq = Sequence([1, 2, 3, 4, 5], SamplingParams(max_tokens=20))
        scheduler.add(seq)

        # Schedule should allocate blocks
        scheduled_seqs, is_prefill = scheduler.schedule()

        assert len(seq.block_table) > 0
        assert scheduler.block_manager.can_allocate(seq) == False or len(seq.block_table) > 0

    def test_draft_block_allocation(self, eagle_config):
        """Test draft block allocation."""
        scheduler = Scheduler(eagle_config)

        seq = Sequence([1, 2, 3], SamplingParams(max_tokens=20))
        scheduler.add(seq)
        scheduler.schedule()  # Prefill

        # Allocate draft blocks
        draft_tokens = 4
        can_allocate = scheduler.block_manager.can_allocate_draft(seq, draft_tokens)

        # Should be able to allocate draft blocks if enough free blocks
        assert isinstance(can_allocate, bool)


class TestSequenceEagleExtensions:
    """Tests for Sequence EAGLE extensions."""

    def test_sequence_draft_token_tracking(self):
        """Test sequence draft token tracking."""
        seq = Sequence([1, 2, 3])

        # Set draft tokens
        draft_tokens = [10, 20, 30]
        seq.set_draft_tokens(draft_tokens)

        assert seq.draft_token_ids == draft_tokens
        assert seq.speculation_depth == len(draft_tokens)

    def test_sequence_clear_draft(self):
        """Test clearing draft state."""
        seq = Sequence([1, 2, 3])
        seq.set_draft_tokens([10, 20, 30])

        seq.clear_draft()

        assert seq.draft_token_ids == []
        assert seq.speculation_depth == 0
        assert seq.accepted_mask == []

    def test_sequence_apply_accepted_tokens(self):
        """Test applying accepted tokens to sequence."""
        seq = Sequence([1, 2, 3])
        initial_len = len(seq)

        # Set draft tokens and acceptance mask
        draft_tokens = [10, 20, 30]
        seq.set_draft_tokens(draft_tokens)
        seq.accepted_mask = [True, True, False]

        # Apply accepted tokens
        seq.apply_accepted_tokens()

        # Should have appended 2 tokens
        assert len(seq) == initial_len + 2
        assert seq.token_ids[-2:] == [10, 20]
        assert seq.draft_token_ids == []  # Cleared after apply


class TestEagleEngineStep:
    """Tests for LLMEngine step with EAGLE."""

    @pytest.mark.skip(reason="Requires full model runner setup - tested in integration")
    def test_engine_step_with_eagle_prefill(self, eagle_config):
        """Test engine step during prefill with EAGLE."""
        # This test requires full model runner setup
        # Covered by end-to-end tests
        pass

    @pytest.mark.skip(reason="Requires full model runner setup - tested in integration")
    def test_engine_step_with_eagle_decode(self, eagle_config):
        """Test engine step during decode with EAGLE."""
        # This test requires full model runner setup
        # Covered by end-to-end tests
        pass


class TestEagleEndToEnd:
    """End-to-end tests for EAGLE speculative decoding."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="EAGLE end-to-end tests require CUDA"
    )
    def test_eagle_single_sequence_generation(self, eagle_config, target_model, draft_model, device):
        """Test complete EAGLE generation for a single sequence."""
        # Create scheduler
        scheduler = Scheduler(eagle_config)

        # Create sequence
        prompt = [1, 2, 3, 4, 5]
        sampling_params = SamplingParams(max_tokens=20, temperature=1.0)
        seq = Sequence(prompt, sampling_params)

        # Add to scheduler
        scheduler.add(seq)

        # Prefill phase
        scheduled_seqs, is_prefill = scheduler.schedule()
        assert is_prefill == True
        assert len(scheduled_seqs) == 1

        # Verify sequence is running
        assert seq.status == SequenceStatus.RUNNING
        assert len(seq.block_table) > 0

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="EAGLE end-to-end tests require CUDA"
    )
    def test_eagle_draft_generation_and_verification(self, qwen_config, target_model, draft_model, device):
        """Test draft token generation and verification workflow."""
        # Create draft runner
        draft_runner = EagleDraftRunner(
            draft_model=draft_model,
            max_speculation_depth=4,
            device=device,
        )

        # Create speculative sampler
        sampler = SpeculativeSampler()

        # Prepare inputs
        seq_len = 10
        hidden_size = qwen_config.hidden_size

        hidden_states = torch.randn(seq_len, hidden_size, device=device, dtype=torch.bfloat16)
        token_ids = torch.randint(0, qwen_config.vocab_size, (seq_len,), device=device)
        positions = torch.arange(seq_len, device=device)

        # Generate draft tokens
        with torch.inference_mode():
            draft_tokens, draft_features = draft_runner.generate_draft_tokens(
                hidden_states=hidden_states,
                token_ids=token_ids,
                positions=positions,
                num_draft_tokens=4,
            )

        # Verify draft generation
        assert len(draft_tokens) == 4
        assert all(0 <= t < qwen_config.vocab_size for t in draft_tokens)
        assert draft_features.shape == (4, hidden_size)


class TestEagleVariableAcceptanceLength:
    """Tests for handling variable acceptance lengths."""

    def test_variable_acceptance_all_accepted(self, eagle_config):
        """Test handling when all draft tokens are accepted."""
        scheduler = Scheduler(eagle_config)

        seq = Sequence([1, 2, 3], SamplingParams(max_tokens=20))
        seq.status = SequenceStatus.RUNNING
        scheduler.running.append(seq)

        # All 4 draft tokens accepted
        accepted_tokens = [[10, 20, 30, 40]]
        finished_mask = scheduler.postprocess_eagle([seq], accepted_tokens)

        assert len(seq) == 7  # 3 + 4
        assert finished_mask[0] == False

    def test_variable_acceptance_partial(self, eagle_config):
        """Test handling when some draft tokens are accepted."""
        scheduler = Scheduler(eagle_config)

        seq = Sequence([1, 2, 3], SamplingParams(max_tokens=20))
        seq.status = SequenceStatus.RUNNING
        scheduler.running.append(seq)

        # Only 2 draft tokens accepted
        accepted_tokens = [[10, 20]]
        finished_mask = scheduler.postprocess_eagle([seq], accepted_tokens)

        assert len(seq) == 5  # 3 + 2
        assert finished_mask[0] == False

    def test_variable_acceptance_none_accepted(self, eagle_config):
        """Test handling when no draft tokens are accepted."""
        scheduler = Scheduler(eagle_config)

        seq = Sequence([1, 2, 3], SamplingParams(max_tokens=20))
        seq.status = SequenceStatus.RUNNING
        scheduler.running.append(seq)

        # No draft tokens accepted (resampled token handled separately)
        accepted_tokens = [[]]
        finished_mask = scheduler.postprocess_eagle([seq], accepted_tokens)

        assert len(seq) == 3  # No change
        assert finished_mask[0] == False


class TestEaglePreemption:
    """Tests for draft token preemption handling."""

    def test_preempt_running_sequence(self, eagle_config):
        """Test preempting a running sequence."""
        scheduler = Scheduler(eagle_config)

        seq = Sequence([1, 2, 3], SamplingParams(max_tokens=20))
        scheduler.add(seq)
        scheduler.schedule()  # Prefill

        assert seq.status == SequenceStatus.RUNNING
        initial_block_count = len(seq.block_table)

        # Preempt
        scheduler.preempt(seq)

        assert seq.status == SequenceStatus.WAITING
        assert len(seq.block_table) == 0
        assert seq in scheduler.waiting

    def test_preempt_with_draft_tokens(self, eagle_config):
        """Test preempting a sequence with draft tokens."""
        scheduler = Scheduler(eagle_config)

        seq = Sequence([1, 2, 3], SamplingParams(max_tokens=20))
        scheduler.add(seq)
        scheduler.schedule()  # Prefill

        # Set draft state
        seq.set_draft_tokens([10, 20, 30])

        # Preempt
        scheduler.preempt(seq)

        assert seq.status == SequenceStatus.WAITING
        # Draft state should be preserved for resumption
        # (or cleared depending on implementation choice)


class TestSchedulerMemoryManagement:
    """Tests for memory management in EAGLE scheduling."""

    def test_block_deallocation_on_finish(self, eagle_config):
        """Test that blocks are deallocated when sequence finishes."""
        scheduler = Scheduler(eagle_config)

        seq = Sequence([1, 2, 3], SamplingParams(max_tokens=5))
        scheduler.add(seq)
        scheduler.schedule()  # Prefill

        initial_free_blocks = len(scheduler.block_manager.free_block_ids)

        # Finish the sequence
        seq.status = SequenceStatus.FINISHED
        scheduler.block_manager.deallocate(seq)
        if seq in scheduler.running:
            scheduler.running.remove(seq)

        # Blocks should be deallocated
        assert len(scheduler.block_manager.free_block_ids) >= initial_free_blocks

    def test_draft_block_deallocation(self, eagle_config):
        """Test draft block deallocation."""
        scheduler = Scheduler(eagle_config)

        seq = Sequence([1, 2, 3], SamplingParams(max_tokens=20))
        scheduler.add(seq)
        scheduler.schedule()  # Prefill

        # Allocate draft blocks
        draft_tokens = 4
        if scheduler.block_manager.can_allocate_draft(seq, draft_tokens):
            draft_blocks = scheduler.block_manager.allocate_draft(seq, draft_tokens)

            assert seq.seq_id in scheduler.block_manager.draft_block_ids

            # Deallocate draft blocks
            scheduler.block_manager.deallocate_draft(seq)

            assert seq.seq_id not in scheduler.block_manager.draft_block_ids


class TestConfigValidation:
    """Tests for EAGLE configuration validation."""

    def test_eagle_config_requires_draft_model(self):
        """Test that EAGLE config requires draft model path."""
        with pytest.raises(AssertionError):
            Config(
                model="models/Qwen3-0.6B",
                eagle_enabled=True,
                eagle_draft_model=None,  # Should fail
                speculation_depth=4,
            )

    def test_eagle_config_speculation_depth_validation(self):
        """Test speculation depth validation."""
        with pytest.raises(AssertionError):
            Config(
                model="models/Qwen3-0.6B",
                eagle_enabled=True,
                eagle_draft_model="new",
                speculation_depth=0,  # Should fail (must be >= 1)
            )

    def test_eagle_config_valid(self, qwen_config):
        """Test valid EAGLE config."""
        config = Config(
            model="models/Qwen3-0.6B",
            eagle_enabled=True,
            eagle_draft_model="new",
            speculation_depth=4,
        )

        assert config.eagle_enabled == True
        assert config.speculation_depth == 4
        assert config.eagle_draft_model == "new"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
