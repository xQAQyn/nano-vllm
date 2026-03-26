from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:
    """Scheduler for managing sequence execution with EAGLE speculative decoding support.

    The scheduler handles:
    1. Prefill phase: Allocate blocks and schedule new sequences
    2. Decode phase: Schedule running sequences for token generation
    3. EAGLE phase: Handle draft token generation and verification (Stage 6)
    """

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()

        # EAGLE speculative decoding configuration (Stage 6)
        self.eagle_enabled = config.eagle_enabled
        self.speculation_depth = config.speculation_depth if self.eagle_enabled else 0

    def is_finished(self):
        return not self.waiting and not self.running

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs.append(seq)
        assert scheduled_seqs
        self.running.extendleft(reversed(scheduled_seqs))
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        """Postprocess sequences after token generation.

        For standard decoding: append tokens and check for completion.
        For EAGLE: tokens are already appended during speculative verification.

        Args:
            seqs: List of sequences to postprocess
            token_ids: Generated token IDs

        Returns:
            List of booleans indicating if each sequence is finished
        """
        finished_mask = []
        for seq, token_id in zip(seqs, token_ids):
            # For EAGLE, tokens may already be appended in run_eagle
            # Check if this is a standard decode step (not EAGLE)
            if not self.eagle_enabled or len(seq.draft_token_ids) == 0:
                seq.append_token(token_id)

            # Check for completion
            is_finished = (not seq.ignore_eos and token_id == self.eos) or \
                         seq.num_completion_tokens >= seq.max_tokens

            if is_finished:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)

            finished_mask.append(is_finished)

        return finished_mask

    def postprocess_eagle(
        self,
        seqs: list[Sequence],
        accepted_token_ids: list[list[int]],
        finished_mask: list[bool] | None = None,
    ) -> list[bool]:
        """Postprocess sequences after EAGLE speculative decoding.

        In EAGLE, multiple tokens may be accepted in a single step.
        This method handles:
        1. Appending accepted tokens to sequence
        2. Clearing draft state
        3. Checking for completion

        Args:
            seqs: List of sequences to postprocess
            accepted_token_ids: List of accepted token IDs for each sequence
            finished_mask: Optional mask indicating which sequences finished

        Returns:
            List of booleans indicating if each sequence is finished
        """
        if finished_mask is None:
            finished_mask = []

        for seq, tokens in zip(seqs, accepted_token_ids):
            # Append accepted tokens to sequence
            for token_id in tokens:
                seq.append_token(token_id)

            # Clear draft state
            seq.clear_draft()

            # Check if sequence finished
            is_finished = seq.status == SequenceStatus.FINISHED or \
                         seq.num_completion_tokens >= seq.max_tokens

            if is_finished and seq.status != SequenceStatus.FINISHED:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                if seq in self.running:
                    self.running.remove(seq)

            finished_mask.append(is_finished)

        return finished_mask
