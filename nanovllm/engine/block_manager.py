from collections import deque
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence


class Block:

    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update(self, hash: int, token_ids: list[int]):
        self.hash = hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []


class BlockManager:

    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        self.free_block_ids: deque[int] = deque(range(num_blocks))
        self.used_block_ids: set[int] = set()
        
        # EAGLE speculative decoding extensions (Stage 3)
        # Track draft blocks separately for speculative tokens
        self.draft_block_ids: dict[int, list[int]] = {}  # seq_id -> list of draft block ids

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix: int = -1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        assert not seq.block_table
        h = -1
        cache_miss = False
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0]
                block = self._allocate_block(block_id)
            else:
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    block = self._allocate_block(block_id)
            if h != -1:
                block.update(h, token_ids)
                self.hash_to_block_id[h] = block_id
            seq.block_table.append(block_id)

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1

    def truncate(self, seq: Sequence, rollback_num_tokens: int):
        last_block_num_tokens = seq.last_block_num_tokens
        if rollback_num_tokens < last_block_num_tokens:
            return
        num_blocks_to_free = (rollback_num_tokens - last_block_num_tokens) // self.block_size + 1
        for _ in range(num_blocks_to_free):
            block_id = seq.block_table.pop()
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)

    # EAGLE speculative decoding methods (Stage 3)
    
    def can_allocate_draft(self, seq: Sequence, draft_num_tokens: int) -> bool:
        """Check if we can allocate blocks for draft tokens."""
        draft_num_blocks = (draft_num_tokens + self.block_size - 1) // self.block_size
        return len(self.free_block_ids) >= draft_num_blocks

    def allocate_draft(self, seq: Sequence, draft_num_tokens: int):
        """Allocate blocks for draft tokens (speculative blocks)."""
        assert seq.seq_id not in self.draft_block_ids or len(self.draft_block_ids[seq.seq_id]) == 0
        draft_num_blocks = (draft_num_tokens + self.block_size - 1) // self.block_size
        draft_blocks = []
        
        for _ in range(draft_num_blocks):
            block_id = self.free_block_ids[0]
            block = self._allocate_block(block_id)
            draft_blocks.append(block_id)
        
        self.draft_block_ids[seq.seq_id] = draft_blocks
        return draft_blocks

    def deallocate_draft(self, seq: Sequence):
        """Deallocate draft blocks on rejection or after acceptance."""
        if seq.seq_id not in self.draft_block_ids:
            return
        
        for block_id in self.draft_block_ids[seq.seq_id]:
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        
        del self.draft_block_ids[seq.seq_id]

    def commit_draft(self, seq: Sequence):
        """Commit draft blocks to the sequence's main block table."""
        if seq.seq_id not in self.draft_block_ids:
            return
        
        draft_blocks = self.draft_block_ids.pop(seq.seq_id)
        seq.block_table.extend(draft_blocks)

    def rollback_draft_blocks(self, seq: Sequence, num_rejected_tokens: int):
        """Rollback draft blocks when tokens are rejected."""
        if seq.seq_id not in self.draft_block_ids:
            return
        
        draft_blocks = self.draft_block_ids[seq.seq_id]
        num_blocks_to_free = (num_rejected_tokens + self.block_size - 1) // self.block_size
        
        # Free blocks from the end
        for _ in range(min(num_blocks_to_free, len(draft_blocks))):
            block_id = draft_blocks.pop()
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        
        if not draft_blocks:
            del self.draft_block_ids[seq.seq_id]
