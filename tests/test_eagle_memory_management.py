"""
Stage 3 Tests for EAGLE-1 Sequence State Extensions

Tests for:
- Sequence can track draft tokens separately from accepted tokens
- Block allocation handles speculative tokens correctly
- Rollback works on token rejection
"""

import pytest
import torch

from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager
from nanovllm.sampling_params import SamplingParams


class TestSequenceDraftTokenTracking:
    """Tests for Sequence draft token tracking extensions."""

    @pytest.fixture
    def sequence(self):
        """Create a test sequence."""
        token_ids = [1, 2, 3, 4, 5]
        return Sequence(token_ids)

    def test_sequence_initial_draft_state(self, sequence):
        """Test that sequence initializes with empty draft state."""
        assert sequence.draft_token_ids == []
        assert sequence.draft_features is None
        assert sequence.accepted_mask == []
        assert sequence.speculation_depth == 0

    def test_append_draft_token_single(self, sequence):
        """Test appending a single draft token."""
        sequence.append_draft_token(100)
        
        assert sequence.draft_token_ids == [100]
        assert sequence.speculation_depth == 1
        assert sequence.accepted_mask == []

    def test_append_draft_token_with_feature(self, sequence):
        """Test appending draft token with feature tensor."""
        feature = torch.randn(512)
        sequence.append_draft_token(100, feature)
        
        assert sequence.draft_token_ids == [100]
        assert sequence.draft_features is not None
        assert len(sequence.draft_features) == 1
        assert torch.allclose(sequence.draft_features[0], feature)

    def test_append_multiple_draft_tokens(self, sequence):
        """Test appending multiple draft tokens."""
        sequence.append_draft_token(100)
        sequence.append_draft_token(101)
        sequence.append_draft_token(102)
        
        assert sequence.draft_token_ids == [100, 101, 102]
        assert sequence.speculation_depth == 3

    def test_set_draft_tokens(self, sequence):
        """Test setting multiple draft tokens at once."""
        token_ids = [100, 101, 102, 103]
        sequence.set_draft_tokens(token_ids)
        
        assert sequence.draft_token_ids == token_ids
        assert sequence.speculation_depth == 4
        assert sequence.accepted_mask == []

    def test_set_draft_tokens_with_features(self, sequence):
        """Test setting draft tokens with feature tensor."""
        token_ids = [100, 101, 102]
        features = torch.randn(3, 512)
        sequence.set_draft_tokens(token_ids, features)
        
        assert sequence.draft_token_ids == token_ids
        assert sequence.draft_features is features
        assert sequence.speculation_depth == 3

    def test_clear_draft(self, sequence):
        """Test clearing draft state."""
        sequence.append_draft_token(100)
        sequence.append_draft_token(101)
        sequence.accepted_mask = [True, False]
        
        sequence.clear_draft()
        
        assert sequence.draft_token_ids == []
        assert sequence.draft_features is None
        assert sequence.accepted_mask == []
        assert sequence.speculation_depth == 0

    def test_rollback_draft_partial(self, sequence):
        """Test rolling back some rejected draft tokens."""
        sequence.append_draft_token(100)
        sequence.append_draft_token(101)
        sequence.append_draft_token(102)
        sequence.accepted_mask = [True, False, False]
        
        sequence.rollback_draft(1)
        
        assert sequence.draft_token_ids == [100, 101]
        assert sequence.speculation_depth == 2

    def test_rollback_draft_all(self, sequence):
        """Test rolling back all draft tokens."""
        sequence.append_draft_token(100)
        sequence.append_draft_token(101)
        
        sequence.rollback_draft(2)
        
        assert sequence.draft_token_ids == []
        assert sequence.speculation_depth == 0

    def test_rollback_draft_zero(self, sequence):
        """Test rolling back zero tokens (no-op)."""
        sequence.append_draft_token(100)
        sequence.append_draft_token(101)
        
        sequence.rollback_draft(0)
        
        assert sequence.draft_token_ids == [100, 101]
        assert sequence.speculation_depth == 2

    def test_apply_accepted_tokens_all(self, sequence):
        """Test applying all accepted draft tokens."""
        initial_tokens = sequence.token_ids.copy()
        sequence.set_draft_tokens([100, 101, 102])
        sequence.accepted_mask = [True, True, True]
        
        sequence.apply_accepted_tokens()
        
        assert sequence.token_ids == initial_tokens + [100, 101, 102]
        assert sequence.num_tokens == len(initial_tokens) + 3
        assert sequence.draft_token_ids == []
        assert sequence.speculation_depth == 0

    def test_apply_accepted_tokens_partial(self, sequence):
        """Test applying partially accepted draft tokens."""
        initial_tokens = sequence.token_ids.copy()
        sequence.set_draft_tokens([100, 101, 102])
        sequence.accepted_mask = [True, False, True]
        
        sequence.apply_accepted_tokens()
        
        assert sequence.token_ids == initial_tokens + [100, 102]
        assert sequence.num_tokens == len(initial_tokens) + 2
        assert sequence.draft_token_ids == []

    def test_apply_accepted_tokens_none(self, sequence):
        """Test when no draft tokens are accepted."""
        initial_tokens = sequence.token_ids.copy()
        sequence.set_draft_tokens([100, 101, 102])
        sequence.accepted_mask = [False, False, False]
        
        sequence.apply_accepted_tokens()
        
        assert sequence.token_ids == initial_tokens
        assert sequence.draft_token_ids == []

    def test_draft_token_ids_independent_from_token_ids(self, sequence):
        """Test that draft tokens are tracked separately from main token IDs."""
        initial_tokens = sequence.token_ids.copy()
        sequence.append_draft_token(100)
        sequence.append_draft_token(101)
        
        # Main token IDs should not include draft tokens
        assert sequence.token_ids == initial_tokens
        assert sequence.draft_token_ids == [100, 101]


class TestSequenceWithSamplingParams:
    """Tests for Sequence draft state with different sampling parameters."""

    def test_draft_state_with_custom_sampling_params(self):
        """Test draft state initialization with custom sampling params."""
        sampling_params = SamplingParams(temperature=0.8, max_tokens=128)
        sequence = Sequence([1, 2, 3], sampling_params)
        
        assert sequence.temperature == 0.8
        assert sequence.max_tokens == 128
        assert sequence.draft_token_ids == []
        assert sequence.speculation_depth == 0


class TestBlockManagerDraftAllocation:
    """Tests for BlockManager draft block allocation extensions."""

    @pytest.fixture
    def block_manager(self):
        """Create a test block manager."""
        return BlockManager(num_blocks=100, block_size=256)

    @pytest.fixture
    def sequence(self):
        """Create a test sequence."""
        return Sequence([1, 2, 3, 4, 5])

    def test_can_allocate_draft_empty(self, block_manager, sequence):
        """Test checking draft allocation for zero tokens."""
        assert block_manager.can_allocate_draft(sequence, 0) is True

    def test_can_allocate_draft_few_tokens(self, block_manager, sequence):
        """Test checking draft allocation for few tokens."""
        assert block_manager.can_allocate_draft(sequence, 10) is True

    def test_can_allocate_draft_many_tokens(self, block_manager, sequence):
        """Test checking draft allocation for many tokens."""
        # 100 blocks * 256 tokens/block = 25600 tokens max
        assert block_manager.can_allocate_draft(sequence, 25600) is True

    def test_can_allocate_draft_exceeds_capacity(self, block_manager, sequence):
        """Test checking draft allocation exceeding capacity."""
        # More tokens than available blocks
        assert block_manager.can_allocate_draft(sequence, 25700) is False

    def test_allocate_draft_basic(self, block_manager, sequence):
        """Test basic draft block allocation."""
        draft_blocks = block_manager.allocate_draft(sequence, 10)
        
        assert len(draft_blocks) == 1  # 10 tokens fit in 1 block
        assert block_manager.draft_block_ids[sequence.seq_id] == draft_blocks
        assert len(block_manager.free_block_ids) == 99  # One block allocated

    def test_allocate_draft_multiple_blocks(self, block_manager, sequence):
        """Test draft allocation requiring multiple blocks."""
        draft_blocks = block_manager.allocate_draft(sequence, 300)
        
        assert len(draft_blocks) == 2  # 300 tokens need 2 blocks
        assert block_manager.draft_block_ids[sequence.seq_id] == draft_blocks
        assert len(block_manager.free_block_ids) == 98  # Two blocks allocated

    def test_allocate_draft_exact_block_boundary(self, block_manager, sequence):
        """Test draft allocation at exact block boundary."""
        draft_blocks = block_manager.allocate_draft(sequence, 256)
        
        assert len(draft_blocks) == 1  # Exactly one block
        assert len(block_manager.free_block_ids) == 99

    def test_allocate_draft_just_over_boundary(self, block_manager, sequence):
        """Test draft allocation just over block boundary."""
        draft_blocks = block_manager.allocate_draft(sequence, 257)
        
        assert len(draft_blocks) == 2  # Need second block for 1 token
        assert len(block_manager.free_block_ids) == 98

    def test_deallocate_draft(self, block_manager, sequence):
        """Test deallocating draft blocks."""
        draft_blocks = block_manager.allocate_draft(sequence, 100)
        initial_free_count = len(block_manager.free_block_ids)
        
        block_manager.deallocate_draft(sequence)
        
        assert sequence.seq_id not in block_manager.draft_block_ids
        assert len(block_manager.free_block_ids) == initial_free_count + 1

    def test_deallocate_draft_multiple_blocks(self, block_manager, sequence):
        """Test deallocating multiple draft blocks."""
        draft_blocks = block_manager.allocate_draft(sequence, 300)
        initial_free_count = len(block_manager.free_block_ids)
        
        block_manager.deallocate_draft(sequence)
        
        assert sequence.seq_id not in block_manager.draft_block_ids
        assert len(block_manager.free_block_ids) == initial_free_count + 2

    def test_deallocate_draft_idempotent(self, block_manager, sequence):
        """Test that deallocating twice doesn't cause errors."""
        block_manager.allocate_draft(sequence, 100)
        block_manager.deallocate_draft(sequence)
        
        # Second deallocation should be a no-op
        block_manager.deallocate_draft(sequence)
        
        assert sequence.seq_id not in block_manager.draft_block_ids

    def test_commit_draft(self, block_manager, sequence):
        """Test committing draft blocks to main block table."""
        # First allocate main blocks for the sequence
        block_manager.allocate(sequence)
        initial_block_count = len(sequence.block_table)
        
        # Allocate and commit draft blocks
        draft_blocks = block_manager.allocate_draft(sequence, 100)
        block_manager.commit_draft(sequence)
        
        assert sequence.seq_id not in block_manager.draft_block_ids
        assert len(sequence.block_table) == initial_block_count + 1
        assert draft_blocks[0] in sequence.block_table

    def test_commit_draft_without_main_allocation(self, block_manager, sequence):
        """Test committing draft blocks without main allocation."""
        draft_blocks = block_manager.allocate_draft(sequence, 100)
        block_manager.commit_draft(sequence)
        
        assert sequence.seq_id not in block_manager.draft_block_ids
        assert draft_blocks[0] in sequence.block_table

    def test_rollback_draft_blocks_partial(self, block_manager, sequence):
        """Test rolling back some draft blocks."""
        draft_blocks = block_manager.allocate_draft(sequence, 300)  # 2 blocks
        initial_free_count = len(block_manager.free_block_ids)
        
        block_manager.rollback_draft_blocks(sequence, 100)
        
        # Should still have draft blocks, but freed one block
        assert sequence.seq_id in block_manager.draft_block_ids
        assert len(block_manager.draft_block_ids[sequence.seq_id]) == 1
        assert len(block_manager.free_block_ids) == initial_free_count + 1

    def test_rollback_draft_blocks_all(self, block_manager, sequence):
        """Test rolling back all draft blocks."""
        draft_blocks = block_manager.allocate_draft(sequence, 100)
        initial_free_count = len(block_manager.free_block_ids)
        
        block_manager.rollback_draft_blocks(sequence, 100)
        
        assert sequence.seq_id not in block_manager.draft_block_ids
        assert len(block_manager.free_block_ids) == initial_free_count + 1

    def test_rollback_draft_blocks_zero(self, block_manager, sequence):
        """Test rolling back zero draft blocks."""
        draft_blocks = block_manager.allocate_draft(sequence, 100)
        initial_free_count = len(block_manager.free_block_ids)
        
        block_manager.rollback_draft_blocks(sequence, 0)
        
        assert sequence.seq_id in block_manager.draft_block_ids
        assert len(block_manager.draft_block_ids[sequence.seq_id]) == len(draft_blocks)
        assert len(block_manager.free_block_ids) == initial_free_count


class TestBlockManagerMultipleSequences:
    """Tests for BlockManager handling multiple sequences with drafts."""

    @pytest.fixture
    def block_manager(self):
        """Create a test block manager."""
        return BlockManager(num_blocks=100, block_size=256)

    @pytest.fixture
    def sequences(self):
        """Create multiple test sequences."""
        return [
            Sequence([1, 2, 3])
            for _ in range(3)
        ]

    def test_allocate_draft_multiple_sequences(self, block_manager, sequences):
        """Test allocating draft blocks for multiple sequences."""
        for seq in sequences:
            block_manager.allocate_draft(seq, 100)
        
        assert len(block_manager.draft_block_ids) == 3
        for seq in sequences:
            assert seq.seq_id in block_manager.draft_block_ids

    def test_deallocate_draft_multiple_sequences(self, block_manager, sequences):
        """Test deallocating draft blocks for multiple sequences."""
        for seq in sequences:
            block_manager.allocate_draft(seq, 100)
        
        # Deallocate first sequence
        block_manager.deallocate_draft(sequences[0])
        
        assert len(block_manager.draft_block_ids) == 2
        assert sequences[0].seq_id not in block_manager.draft_block_ids
        assert sequences[1].seq_id in block_manager.draft_block_ids
        assert sequences[2].seq_id in block_manager.draft_block_ids


class TestBlockManagerDraftBlockReuse:
    """Tests for draft block deallocation and reuse."""

    @pytest.fixture
    def block_manager(self):
        """Create a test block manager with limited blocks."""
        return BlockManager(num_blocks=10, block_size=256)

    def test_draft_block_reuse_after_deallocate(self, block_manager):
        """Test that draft blocks can be reused after deallocation."""
        sequence = Sequence([1, 2, 3])
        
        # First allocation
        draft_blocks_1 = block_manager.allocate_draft(sequence, 100)
        initial_free_count = len(block_manager.free_block_ids)
        block_manager.deallocate_draft(sequence)
        after_dealloc_free_count = len(block_manager.free_block_ids)
        
        # Second allocation should succeed (block is available for reuse)
        draft_blocks_2 = block_manager.allocate_draft(sequence, 100)
        
        # The important thing is that allocation succeeds and block count is correct
        assert len(draft_blocks_2) == len(draft_blocks_1) == 1
        # Free block count should be restored after deallocation
        assert after_dealloc_free_count == initial_free_count + 1

    def test_draft_block_reuse_after_commit(self, block_manager):
        """Test that committed draft blocks can be reused."""
        sequence = Sequence([1, 2, 3])
        
        # Allocate main blocks first
        block_manager.allocate(sequence)
        
        # Allocate and commit draft blocks
        draft_blocks_1 = block_manager.allocate_draft(sequence, 100)
        block_manager.commit_draft(sequence)
        
        # Deallocate sequence (frees all blocks)
        block_manager.deallocate(sequence)
        
        # Re-allocate sequence
        sequence.block_table = []
        block_manager.allocate(sequence)
        
        # Allocate draft blocks again
        draft_blocks_2 = block_manager.allocate_draft(sequence, 100)
        
        # Should be able to allocate successfully
        assert len(draft_blocks_2) == 1


class TestIntegration:
    """Integration tests for Sequence and BlockManager working together."""

    @pytest.fixture
    def block_manager(self):
        """Create a test block manager."""
        return BlockManager(num_blocks=50, block_size=256)

    @pytest.fixture
    def sequence(self):
        """Create a test sequence."""
        return Sequence([1, 2, 3, 4, 5])

    def test_full_draft_lifecycle_allocate_generate_commit(self, block_manager, sequence):
        """Test full draft lifecycle: allocate, generate, commit."""
        # Allocate main blocks
        block_manager.allocate(sequence)
        initial_block_count = len(sequence.block_table)
        
        # Generate draft tokens
        draft_token_ids = [100, 101, 102, 103, 104]
        sequence.set_draft_tokens(draft_token_ids)
        
        # Allocate draft blocks
        assert block_manager.can_allocate_draft(sequence, len(draft_token_ids))
        block_manager.allocate_draft(sequence, len(draft_token_ids))
        
        # Accept all draft tokens
        sequence.accepted_mask = [True] * len(draft_token_ids)
        sequence.apply_accepted_tokens()
        
        # Commit draft blocks
        block_manager.commit_draft(sequence)
        
        assert sequence.draft_token_ids == []
        assert sequence.token_ids[-5:] == draft_token_ids
        assert len(sequence.block_table) == initial_block_count + 1

    def test_full_draft_lifecycle_with_rejection(self, block_manager, sequence):
        """Test full draft lifecycle with token rejection."""
        # Allocate main blocks
        block_manager.allocate(sequence)
        initial_token_count = len(sequence.token_ids)
        
        # Generate draft tokens
        draft_token_ids = [100, 101, 102, 103, 104]
        sequence.set_draft_tokens(draft_token_ids)
        
        # Allocate draft blocks
        block_manager.allocate_draft(sequence, len(draft_token_ids))
        
        # Accept only first 2 tokens
        sequence.accepted_mask = [True, True, False, False, False]
        
        # Rollback rejected tokens
        sequence.rollback_draft(3)
        block_manager.rollback_draft_blocks(sequence, 3)
        
        # Apply accepted tokens
        sequence.apply_accepted_tokens()
        
        # Commit remaining draft blocks
        block_manager.commit_draft(sequence)
        
        assert sequence.token_ids[-2:] == [100, 101]
        assert sequence.draft_token_ids == []

    def test_full_draft_lifecycle_all_rejected(self, block_manager, sequence):
        """Test full draft lifecycle with all tokens rejected."""
        # Allocate main blocks
        block_manager.allocate(sequence)
        initial_token_count = len(sequence.token_ids)
        
        # Generate draft tokens
        draft_token_ids = [100, 101, 102]
        sequence.set_draft_tokens(draft_token_ids)
        
        # Allocate draft blocks
        block_manager.allocate_draft(sequence, len(draft_token_ids))
        
        # Reject all tokens
        sequence.accepted_mask = [False, False, False]
        sequence.rollback_draft(3)
        block_manager.rollback_draft_blocks(sequence, 3)
        
        # Apply (nothing to apply)
        sequence.apply_accepted_tokens()
        
        assert sequence.token_ids == list(range(1, 6))  # Unchanged
        assert sequence.draft_token_ids == []


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
