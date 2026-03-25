from copy import copy
from enum import Enum, auto
from itertools import count

import torch

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams()):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        
        # EAGLE speculative decoding extensions (Stage 3)
        self.draft_token_ids: list[int] = []
        self.draft_features: torch.Tensor | None = None
        self.accepted_mask: list[bool] = []
        self.speculation_depth: int = 0

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1

    def pop_token(self):
        token_id = self.token_ids.pop()
        self.num_tokens -= 1
        self.last_token = self.token_ids[-1] if self.token_ids else None
        return token_id
    
    def truncate(self, rollback_num_tokens: int):
        for _ in range(rollback_num_tokens):
            self.pop_token()

    # EAGLE speculative decoding methods (Stage 3)
    
    def append_draft_token(self, token_id: int, feature: torch.Tensor | None = None):
        """Append a draft token predicted by the draft model."""
        self.draft_token_ids.append(token_id)
        self.speculation_depth = len(self.draft_token_ids)
        if feature is not None:
            if self.draft_features is None:
                self.draft_features = []
            self.draft_features.append(feature)

    def set_draft_tokens(self, token_ids: list[int], features: torch.Tensor | None = None):
        """Set multiple draft tokens at once."""
        self.draft_token_ids = token_ids.copy()
        self.draft_features = features
        self.speculation_depth = len(token_ids)
        self.accepted_mask = []

    def clear_draft(self):
        """Clear all draft token state."""
        self.draft_token_ids.clear()
        self.draft_features = None
        self.accepted_mask.clear()
        self.speculation_depth = 0

    def rollback_draft(self, num_rejected: int):
        """Rollback rejected draft tokens from the end."""
        if num_rejected > 0:
            self.draft_token_ids = self.draft_token_ids[:-num_rejected] if num_rejected < len(self.draft_token_ids) else []
            if self.draft_features is not None and isinstance(self.draft_features, list):
                self.draft_features = self.draft_features[:-num_rejected] if num_rejected < len(self.draft_features) else []
            self.accepted_mask = self.accepted_mask[:-num_rejected] if num_rejected < len(self.accepted_mask) else []
            self.speculation_depth = len(self.draft_token_ids)

    def apply_accepted_tokens(self):
        """Apply accepted draft tokens to the main token sequence."""
        for i, token_id in enumerate(self.draft_token_ids):
            if i < len(self.accepted_mask) and self.accepted_mask[i]:
                self.append_token(token_id)
        self.clear_draft()

    def __getstate__(self):
        return (self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]
