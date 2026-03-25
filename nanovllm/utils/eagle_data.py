"""Training data preparation for EAGLE draft model.

Collects hidden states + token sequences from target model and creates
(feature_sequence, shifted_token_sequence, next_feature_target) tuples.

Uses lazy loading - model runs on-demand during __getitem__ to avoid
high memory/disk usage from pre-collecting all data.
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Iterator, Optional
import hashlib
import os

from nanovllm.config import Config
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.utils.context import set_context, reset_context


class EagleTrainingSample:
    """A single training sample for EAGLE draft model.
    
    Contains:
    - token_ids: Token IDs (shifted by 1 for prediction)
    - hidden_states: Hidden states from target model's second-to-last layer
    - positions: Position IDs
    - target_features: Target hidden states to predict (next position)
    - target_tokens: Target token IDs for token prediction loss
    """
    
    def __init__(
        self,
        token_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        target_features: torch.Tensor,
        target_tokens: torch.Tensor,
    ):
        self.token_ids = token_ids
        self.hidden_states = hidden_states
        self.positions = positions
        self.target_features = target_features
        self.target_tokens = target_tokens


class EagleDataset(Dataset):
    """Dataset for EAGLE draft model training.

    Collects hidden states from target model and creates training pairs.
    Uses lazy loading - model runs on-demand during __getitem__ to avoid
    high memory/disk usage.
    """

    def __init__(
        self,
        target_model: Qwen3ForCausalLM,
        tokenizer: AutoTokenizer,
        prompts: list[str],
        max_seq_len: int = 2048,
        noise_std: float = 0.1,
        device: str | torch.device = 'cpu',
        dtype: torch.dtype = torch.float32,
        cache_dir: Optional[str] = None,
        use_cache: bool = False,
    ):
        """
        Args:
            target_model: Target model (frozen, eval mode)
            tokenizer: Tokenizer
            prompts: List of prompts for data generation
            max_seq_len: Maximum sequence length
            noise_std: Standard deviation of noise to add for data augmentation
            device: Device to run target model on
            dtype: Data type for collected data
            cache_dir: Directory to cache processed data (optional)
            use_cache: Whether to use disk caching (requires cache_dir)
        """
        self.target_model = target_model
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.noise_std = noise_std
        self.device = torch.device(device)
        self.dtype = dtype
        self.prompts = prompts
        self.use_cache = use_cache
        self.cache_dir = cache_dir

        # Move model to device if not already
        self.target_model.to(self.device)
        self.target_model.eval()

        # Create cache directory if needed
        if self.use_cache and self.cache_dir:
            os.makedirs(self.cache_dir, exist_ok=True)

        # Pre-tokenize all prompts (lightweight operation)
        self.tokenized_prompts: list[tuple[str, torch.Tensor]] = []
        self._valid_indices: list[int] = []
        self._pretokenize_prompts()

    def _pretokenize_prompts(self):
        """Pre-tokenize all prompts to avoid repeated tokenization."""
        for idx, prompt in enumerate(self.prompts):
            input_ids = self.tokenizer.encode(
                prompt, 
                return_tensors='pt', 
                truncation=True, 
                max_length=self.max_seq_len
            )
            input_ids = input_ids.squeeze(0)  # [seq_len]

            if len(input_ids) >= 2:
                self._valid_indices.append(idx)
                self.tokenized_prompts.append((prompt, input_ids))

    def _get_cache_path(self, prompt: str) -> str:
        """Get cache file path for a prompt."""
        prompt_hash = hashlib.md5(prompt.encode()).hexdigest()
        return os.path.join(self.cache_dir, f"{prompt_hash}.pt")

    def _load_from_cache(self, prompt: str) -> Optional[EagleTrainingSample]:
        """Load sample from cache if available."""
        if not self.use_cache or not self.cache_dir:
            return None
        
        cache_path = self._get_cache_path(prompt)
        if os.path.exists(cache_path):
            try:
                data = torch.load(cache_path, weights_only=True)
                return EagleTrainingSample(
                    token_ids=data['token_ids'],
                    hidden_states=data['hidden_states'],
                    positions=data['positions'],
                    target_features=data['target_features'],
                    target_tokens=data['target_tokens'],
                )
            except Exception:
                # Cache corrupted, will regenerate
                pass
        return None

    def _save_to_cache(self, prompt: str, sample: EagleTrainingSample):
        """Save sample to cache."""
        if not self.use_cache or not self.cache_dir:
            return
        
        cache_path = self._get_cache_path(prompt)
        torch.save({
            'token_ids': sample.token_ids,
            'hidden_states': sample.hidden_states,
            'positions': sample.positions,
            'target_features': sample.target_features,
            'target_tokens': sample.target_tokens,
        }, cache_path)

    def _create_sample_from_input_ids(self, input_ids: torch.Tensor) -> Optional[EagleTrainingSample]:
        """Create a training sample from tokenized input."""
        seq_len = len(input_ids) - 1  # Need at least 2 tokens
        if seq_len < 1:
            return None

        # Run target model and collect hidden states
        with torch.no_grad():
            hidden_states = self._get_hidden_states(input_ids)

        # Token IDs (excluding last token, as we predict next)
        token_ids = input_ids[:-1]

        # Hidden states (excluding last, as we predict next)
        hidden_states_input = hidden_states[:-1]

        # Target hidden states (shifted by 1)
        target_features = hidden_states[1:]

        # Target tokens (shifted by 1)
        target_tokens = input_ids[1:]

        # Positions
        positions = torch.arange(len(token_ids), dtype=torch.int64)

        # Add noise for data augmentation
        if self.noise_std > 0:
            noise = torch.randn_like(hidden_states_input) * self.noise_std
            hidden_states_input = hidden_states_input + noise

        return EagleTrainingSample(
            token_ids=token_ids,
            hidden_states=hidden_states_input,
            positions=positions,
            target_features=target_features,
            target_tokens=target_tokens,
        )

    def _get_hidden_states(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Get hidden states from second-to-last layer of target model.

        This requires modifying the forward pass to expose intermediate states.
        Note: We run in prefill mode without KV cache for data collection.
        """
        # Move inputs to device
        input_ids = input_ids.to(self.device)

        # Prepare inputs
        positions = torch.arange(len(input_ids), dtype=torch.int64, device=self.device)

        # Set up context for prefill mode (no KV cache)
        seq_len = len(input_ids)
        cu_seqlens_q = torch.tensor([0, seq_len], dtype=torch.int32, device=self.device)
        cu_seqlens_k = torch.tensor([0, seq_len], dtype=torch.int32, device=self.device)
        slot_mapping = torch.full((seq_len,), -1, dtype=torch.int32, device=self.device)

        set_context(
            is_prefill=True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=seq_len,
            max_seqlen_k=seq_len,
            slot_mapping=slot_mapping,
            context_lens=None,
            block_tables=None,
        )

        try:
            # Run model up to second-to-last layer
            hidden_states = self.target_model.model.embed_tokens(input_ids)
            residual = None

            # Run through all layers except the last one
            num_layers = len(self.target_model.model.layers)
            for i, layer in enumerate(self.target_model.model.layers[:-1]):
                hidden_states, residual = layer(positions, hidden_states, residual)

            # Apply final norm
            hidden_states, _ = self.target_model.model.norm(hidden_states, residual)
        finally:
            reset_context()

        # Convert to target dtype and move to CPU for storage
        return hidden_states.to(self.dtype).cpu()

    def __len__(self) -> int:
        return len(self._valid_indices)

    def __getitem__(self, idx: int) -> EagleTrainingSample:
        """Load or generate a training sample on-demand.
        
        This uses lazy loading - the model is run only when a sample is requested,
        avoiding high memory/disk usage from pre-collecting all data.
        """
        valid_idx = self._valid_indices[idx]
        prompt, input_ids = self.tokenized_prompts[idx]

        # Try to load from cache first
        sample = self._load_from_cache(prompt)
        if sample is not None:
            return sample

        # Generate sample on-demand
        sample = self._create_sample_from_input_ids(input_ids)
        
        if sample is None:
            # Fallback: create a minimal dummy sample
            raise ValueError(f"Failed to create sample for prompt at index {idx}")

        # Save to cache for future use
        self._save_to_cache(prompt, sample)

        return sample


def create_training_dataloader(
    target_model: Qwen3ForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    batch_size: int = 4,
    max_seq_len: int = 2048,
    noise_std: float = 0.1,
    shuffle: bool = True,
    device: str | torch.device = 'cpu',
    dtype: torch.dtype = torch.float32,
    cache_dir: Optional[str] = None,
    use_cache: bool = False,
    num_workers: int = 0,
) -> DataLoader:
    """Create a DataLoader for EAGLE training.

    Args:
        target_model: Target model (frozen)
        tokenizer: Tokenizer
        prompts: Training prompts
        batch_size: Batch size
        max_seq_len: Maximum sequence length
        noise_std: Noise standard deviation for augmentation
        shuffle: Whether to shuffle data
        device: Device to run target model on
        dtype: Data type for collected data
        cache_dir: Directory to cache processed data (optional)
        use_cache: Whether to use disk caching (requires cache_dir)
        num_workers: Number of DataLoader workers (0 = main process)

    Returns:
        DataLoader yielding batches of training samples
    """
    dataset = EagleDataset(
        target_model=target_model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_seq_len=max_seq_len,
        noise_std=noise_std,
        device=device,
        dtype=dtype,
        cache_dir=cache_dir,
        use_cache=use_cache,
    )

    def collate_fn(batch: list[EagleTrainingSample]) -> dict[str, torch.Tensor]:
        """Collate function for batching samples.

        Since samples may have different lengths, we pad them.
        """
        max_len = max(len(s.token_ids) for s in batch)

        token_ids = torch.zeros(len(batch), max_len, dtype=torch.int64)
        hidden_states = torch.zeros(len(batch), max_len, batch[0].hidden_states.size(-1))
        positions = torch.zeros(len(batch), max_len, dtype=torch.int64)
        target_features = torch.zeros(len(batch), max_len, batch[0].target_features.size(-1))
        target_tokens = torch.zeros(len(batch), max_len, dtype=torch.int64)
        lengths = torch.zeros(len(batch), dtype=torch.int64)  # Store actual lengths

        for i, sample in enumerate(batch):
            seq_len = len(sample.token_ids)
            token_ids[i, :seq_len] = sample.token_ids
            hidden_states[i, :seq_len] = sample.hidden_states
            positions[i, :seq_len] = sample.positions
            target_features[i, :seq_len] = sample.target_features
            target_tokens[i, :seq_len] = sample.target_tokens
            lengths[i] = seq_len

        return {
            'token_ids': token_ids,
            'hidden_states': hidden_states,
            'positions': positions,
            'target_features': target_features,
            'target_tokens': target_tokens,
            'lengths': lengths,  # Add lengths to batch
        }

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=collate_fn,
        num_workers=num_workers,
    )
