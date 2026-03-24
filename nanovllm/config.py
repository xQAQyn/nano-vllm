import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    # EAGLE speculative decoding options
    eagle_draft_model: str | None = None
    speculation_depth: int = 4
    eagle_enabled: bool = False

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        assert self.max_num_batched_tokens >= self.max_model_len
        if self.eagle_enabled:
            assert self.eagle_draft_model is not None, "eagle_draft_model must be specified when eagle_enabled=True"
            # Allow draft model path to be a directory (for loading) or None (for creating new)
            if self.eagle_draft_model is not None and not os.path.isdir(self.eagle_draft_model):
                # Check if it's a special value like "new" for creating a new draft model
                if self.eagle_draft_model != "new":
                    raise AssertionError(f"Draft model path does not exist: {self.eagle_draft_model}")
            assert self.speculation_depth >= 1, "speculation_depth must be at least 1"
