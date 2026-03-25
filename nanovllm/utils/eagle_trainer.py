"""EAGLE draft model trainer.

Implements the training loop for the EAGLE draft model with:
- L_reg: SmoothL1 loss for feature prediction
- L_cls: CrossEntropy loss for token prediction (weight=0.1)
- Acceptance rate evaluation for speculative decoding performance
"""

import os
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
from typing import Callable

from nanovllm.models.eagle import EagleDraftModel
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from transformers import AutoTokenizer


class EagleTrainer:
    """Trainer for EAGLE draft model.

    Training objectives:
    1. Feature prediction loss (SmoothL1)
    2. Token prediction loss (CrossEntropy, weighted by 0.1)
    3. Acceptance rate evaluation for speculative decoding
    """

    def __init__(
        self,
        draft_model: EagleDraftModel,
        target_model: Qwen3ForCausalLM | None = None,
        tokenizer: AutoTokenizer | None = None,
        lr: float = 1e-4,
        weight_decay: float = 0.01,
        feature_loss_weight: float = 1.0,
        token_loss_weight: float = 0.1,
        max_grad_norm: float = 1.0,
        device: str = 'cuda',
        dtype: torch.dtype | None = None,
        disable_compile: bool = True,
    ):
        """
        Args:
            draft_model: Draft model to train
            target_model: Target model for acceptance rate evaluation (optional)
            tokenizer: Tokenizer for acceptance rate evaluation (optional)
            lr: Learning rate
            weight_decay: Weight decay for AdamW
            feature_loss_weight: Weight for feature prediction loss
            token_loss_weight: Weight for token prediction loss
            max_grad_norm: Maximum gradient norm for clipping
            device: Device to train on
            dtype: Data type for training (None = use model's dtype)
            disable_compile: Disable torch.compile for training (avoids in-place op issues)
        """
        self.draft_model = draft_model.to(device)
        if dtype is not None:
            self.draft_model = self.draft_model.to(dtype)
        self.device = torch.device(device)
        self.device_type = self.device.type  # Store device type for autocast
        self.dtype = dtype or next(draft_model.parameters()).dtype
        self.feature_loss_weight = feature_loss_weight
        self.token_loss_weight = token_loss_weight
        self.max_grad_norm = max_grad_norm
        self.target_model = target_model
        self.tokenizer = tokenizer

        # Disable torch.compile for layers that use in-place operations
        if disable_compile:
            self._disable_compilation()

        # Optimizer - only optimize trainable parameters
        trainable_params = draft_model.get_trainable_params()
        self.optimizer = AdamW(
            trainable_params,
            lr=lr,
            weight_decay=weight_decay,
        )

        # Loss functions
        self.feature_loss_fn = nn.SmoothL1Loss()
        self.token_loss_fn = nn.CrossEntropyLoss()

        # Training state
        self.global_step = 0
        self.best_loss = float('inf')
    
    def _disable_compilation(self):
        """Disable torch.compile for layers with in-place operations."""
        # Disable compilation for the entire draft model during training
        torch._dynamo.disable(self.draft_model)
        
        # Also disable RMSNorm compilation in decoder layer
        def disable_rmsnorm_compile(module):
            if hasattr(module, 'disable_compile'):
                module.disable_compile()
            for child in module.children():
                disable_rmsnorm_compile(child)
        
        if hasattr(self.draft_model, 'decoder_layer'):
            disable_rmsnorm_compile(self.draft_model.decoder_layer)
    
    def compute_loss(
        self,
        predicted_features: torch.Tensor,
        predicted_logits: torch.Tensor,
        target_features: torch.Tensor,
        target_tokens: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute combined loss.

        Args:
            predicted_features: Predicted hidden features, [num_valid_tokens, hidden_size]
            predicted_logits: Predicted logits, [num_valid_tokens, vocab_size]
            target_features: Target hidden features, [num_valid_tokens, hidden_size]
            target_tokens: Target token IDs, [num_valid_tokens]

        Returns:
            tuple:
                - total_loss: Combined loss
                - loss_dict: Dictionary with individual losses
        """
        # Features and logits are already flattened (valid tokens only)
        # Targets are also flattened

        # Feature prediction loss (SmoothL1)
        feature_loss = self.feature_loss_fn(predicted_features, target_features)

        # Token prediction loss (CrossEntropy)
        token_loss = self.token_loss_fn(predicted_logits, target_tokens)

        # Combined loss
        total_loss = (
            self.feature_loss_weight * feature_loss +
            self.token_loss_weight * token_loss
        )

        loss_dict = {
            'total_loss': total_loss.item(),
            'feature_loss': feature_loss.item(),
            'token_loss': token_loss.item(),
        }

        return total_loss, loss_dict
    
    def train_step(
        self,
        batch: dict[str, torch.Tensor],
    ) -> dict[str, float]:
        """Single training step.

        Args:
            batch: Dictionary with batch data

        Returns:
            Dictionary with loss values
        """
        from nanovllm.utils.context import set_context, reset_context

        self.draft_model.train()
        self.optimizer.zero_grad()

        # Move batch to device and convert to correct dtype
        token_ids = batch['token_ids'].to(self.device)
        hidden_states = batch['hidden_states'].to(self.device).to(self.dtype)
        positions = batch['positions'].to(self.device)
        target_features = batch['target_features'].to(self.device).to(self.dtype)
        target_tokens = batch['target_tokens'].to(self.device)
        lengths = batch.get('lengths')
        if lengths is not None:
            lengths = lengths.to(self.device)  # Move lengths to device

        batch_size, max_seq_len = token_ids.shape

        # Create masks for valid positions (exclude padding)
        if lengths is not None:
            # Create position indices and compare with lengths to get valid mask
            position_indices = torch.arange(max_seq_len, device=self.device).unsqueeze(0)
            mask = position_indices < lengths.unsqueeze(1)

            # Only process valid tokens
            token_ids_flat = token_ids[mask]
            hidden_states_flat = hidden_states[mask]
            positions_flat = positions[mask]
            total_len = len(token_ids_flat)
        else:
            token_ids_flat = token_ids.reshape(-1)
            hidden_states_flat = hidden_states.reshape(-1, hidden_states.size(-1))
            positions_flat = positions.reshape(-1)
            total_len = len(token_ids_flat)

        # Set up context for prefill mode (training uses prefill mode without KV cache)
        cu_seqlens_q = torch.tensor([0, total_len], dtype=torch.int32, device=self.device)
        cu_seqlens_k = torch.tensor([0, total_len], dtype=torch.int32, device=self.device)
        slot_mapping = torch.full((total_len,), -1, dtype=torch.int32, device=self.device)

        set_context(
            is_prefill=True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=total_len,
            max_seqlen_k=total_len,
            slot_mapping=slot_mapping,
            context_lens=None,
            block_tables=None,
        )

        try:
            # Forward pass through draft model
            # Use torch.autocast to ensure correct dtype for mixed precision
            with torch.autocast(device_type=self.device_type, dtype=self.dtype, enabled=False):
                predicted_features, predicted_logits = self.draft_model(
                    token_ids=token_ids_flat,
                    hidden_states=hidden_states_flat,
                    positions=positions_flat,
                )
                
                # Check for NaN in forward pass outputs
                if torch.isnan(predicted_features).any() or torch.isnan(predicted_logits).any():
                    print(f"Warning: NaN detected in forward pass outputs")
                    print(f"  predicted_features NaN: {torch.isnan(predicted_features).any().item()}")
                    print(f"  predicted_logits NaN: {torch.isnan(predicted_logits).any().item()}")
        finally:
            reset_context()

        # For loss computation, we need to handle variable length sequences
        # Flatten targets using the same mask
        if lengths is not None:
            target_features_flat = target_features[mask]
            target_tokens_flat = target_tokens[mask]
        else:
            target_features_flat = target_features.reshape(-1, target_features.size(-1))
            target_tokens_flat = target_tokens.reshape(-1)

        # Check for NaN in targets
        if torch.isnan(target_features_flat).any() or torch.isnan(target_tokens_flat).any():
            print(f"Warning: NaN detected in target data")
            print(f"  target_features NaN: {torch.isnan(target_features_flat).any().item()}")

        # Compute loss on valid tokens only
        # Note: predicted_features and predicted_logits are already [num_valid_tokens, ...]

        total_loss, loss_dict = self.compute_loss(
            predicted_features=predicted_features,
            predicted_logits=predicted_logits,
            target_features=target_features_flat,
            target_tokens=target_tokens_flat,
        )

        # Check for NaN in loss
        if torch.isnan(total_loss):
            print(f"Warning: NaN loss detected!")
            print(f"  feature_loss: {loss_dict['feature_loss']}")
            print(f"  token_loss: {loss_dict['token_loss']}")
            print(f"  predicted_features stats: mean={predicted_features.mean().item():.6f}, std={predicted_features.std().item():.6f}")
            print(f"  predicted_logits stats: mean={predicted_logits.mean().item():.6f}, std={predicted_logits.std().item():.6f}")
            print(f"  target_features stats: mean={target_features_flat.mean().item():.6f}, std={target_features_flat.std().item():.6f}")
            # Return loss dict without backward pass to avoid NaN gradient propagation
            return loss_dict

        # Backward pass
        total_loss.backward()

        # Check for NaN gradients
        has_nan_grad = False
        for name, param in self.draft_model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                print(f"Warning: NaN gradient in {name}")
                has_nan_grad = True
        
        if has_nan_grad:
            print("Warning: NaN gradients detected, skipping optimizer step")
            return loss_dict

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.draft_model.parameters(),
            self.max_grad_norm,
        )

        # Optimizer step
        self.optimizer.step()

        self.global_step += 1

        return loss_dict

    @torch.no_grad()
    def evaluate_acceptance_rate(
        self,
        val_prompts: list[str],
        max_samples: int = 100,
        max_tokens: int = 64,
        top_k: list[int] | None = None,
    ) -> dict[str, float]:
        """Evaluate draft model acceptance rate on validation prompts.

        The acceptance rate measures how often the draft model's predictions
        would be accepted by the target model during speculative decoding.
        The mean accept sequence length measures the average length of consecutive
        accepted tokens before a rejection (important for EAGLE performance).

        Args:
            val_prompts: List of validation prompts
            max_samples: Maximum number of prompts to evaluate
            max_tokens: Maximum tokens to generate per prompt
            top_k: List of k values for top-k acceptance rate evaluation.
                   Default: [1, 2, 3, 4, 5, 10]

        Returns:
            Dictionary with:
                - acceptance_rate: Overall acceptance rate (0.0 to 1.0) (top-1)
                - mean_accept_seq_len: Mean accept sequence length
                - top_k_acceptance_rate_k: Acceptance rate for top-k (for each k in top_k)
        """
        if self.target_model is None or self.tokenizer is None:
            return {'acceptance_rate': 0.0, 'mean_accept_seq_len': 0.0}

        if top_k is None:
            top_k = [1, 2, 3, 4, 5, 10]

        self.draft_model.eval()
        self.target_model.eval()

        total_accepted = 0
        total_predicted = 0
        num_errors = 0

        # For mean accept sequence length
        total_accept_seq_len = 0.0
        num_accept_sequences = 0

        # For top-k acceptance rates: track cumulative matches at each k
        top_k_matches = {k: 0 for k in top_k}
        top_k_total = {k: 0 for k in top_k}

        # Limit samples
        val_prompts = val_prompts[:max_samples]

        for idx, prompt in enumerate(val_prompts):
            # Tokenize prompt
            input_ids = self.tokenizer.encode(
                prompt,
                return_tensors='pt',
                truncation=True,
                max_length=512,
            ).squeeze(0).to(self.device)

            if len(input_ids) < 2:
                continue

            # Run target model to get hidden states from penultimate layer
            try:
                target_outputs = self._get_target_hidden_states_and_logits(input_ids)
                if target_outputs is None:
                    num_errors += 1
                    continue
                target_hidden, _ = target_outputs  # We only use hidden states, not logits
            except Exception as e:
                num_errors += 1
                if idx < 3:  # Print first 3 errors
                    print(f"    Warning: Error getting target hidden states: {e}")
                continue

            # Run draft model to get predictions
            try:
                draft_outputs = self._get_draft_predictions(input_ids, target_hidden)
                if draft_outputs is None:
                    num_errors += 1
                    continue
                draft_logits = draft_outputs
            except Exception as e:
                num_errors += 1
                if idx < 3:  # Print first 3 errors
                    print(f"    Warning: Error getting draft predictions: {e}")
                continue

            # Compute acceptance rate for this sequence
            # Target tokens: derived from penultimate layer hidden states (what draft is predicting)
            # Draft tokens: predicted by draft model
            target_logits_from_hidden = self.target_model.lm_head(target_hidden[:-1])  # Exclude last position
            target_tokens = target_logits_from_hidden.argmax(dim=-1)
            draft_tokens = draft_logits.argmax(dim=-1)

            # Get top-k draft predictions at each position
            draft_top_k_tokens = draft_logits.topk(max(top_k), dim=-1).indices  # [seq_len, max_k]

            # Count matches and compute accept sequence lengths
            seq_len = min(len(target_tokens), len(draft_tokens))
            current_accept_seq_len = 0
            for i in range(seq_len):
                target_token = target_tokens[i].item()
                
                # Top-1 acceptance (original behavior)
                total_predicted += 1
                draft_top1_token = draft_tokens[i].item()
                if target_token == draft_top1_token:
                    total_accepted += 1
                    current_accept_seq_len += 1
                else:
                    # Rejection occurred, record the accept sequence length
                    if current_accept_seq_len > 0:
                        total_accept_seq_len += current_accept_seq_len
                        num_accept_sequences += 1
                    current_accept_seq_len = 0

                # Top-k acceptance: check if target is in top-k draft predictions
                for k in top_k:
                    top_k_total[k] += 1
                    draft_top_k = draft_top_k_tokens[i, :k].tolist()
                    if target_token in draft_top_k:
                        top_k_matches[k] += 1

            # Don't forget the last accept sequence if it ends without rejection
            if current_accept_seq_len > 0:
                total_accept_seq_len += current_accept_seq_len
                num_accept_sequences += 1

        self.draft_model.train()

        # Print debug info
        print(f"    Acceptance eval: {total_accepted}/{total_predicted} matches, {num_errors} errors")

        acceptance_rate = total_accepted / total_predicted if total_predicted > 0 else 0.0
        mean_accept_seq_len = total_accept_seq_len / num_accept_sequences if num_accept_sequences > 0 else 0.0

        print(f"    Mean accept sequence length: {mean_accept_seq_len:.2f}")

        result = {
            'acceptance_rate': acceptance_rate,
            'mean_accept_seq_len': mean_accept_seq_len,
        }

        # Add top-k acceptance rates
        for k in top_k:
            top_k_rate = top_k_matches[k] / top_k_total[k] if top_k_total[k] > 0 else 0.0
            result[f'top{k}_acceptance_rate'] = top_k_rate
            print(f"    Top-{k} acceptance rate: {top_k_rate:.4f}")

        return result

    def _get_target_hidden_states_and_logits(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Get hidden states from penultimate layer and logits from target model.

        Returns:
            tuple: (hidden_states_from_penultimate_layer, logits_from_last_layer)
        """
        from nanovllm.utils.context import set_context, reset_context

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
            hidden_states = self.target_model.model.embed_tokens(input_ids)
            residual = None

            # Run through all layers except the last one
            for layer in self.target_model.model.layers[:-1]:
                hidden_states, residual = layer(
                    positions=torch.arange(seq_len, device=self.device),
                    hidden_states=hidden_states,
                    residual=residual,
                )

            # Apply final norm to get penultimate layer output
            hidden_states_norm, _ = self.target_model.model.norm(hidden_states, residual)

            # Get logits from the penultimate layer hidden states
            # (This is what the draft model is trained to predict)
            logits = self.target_model.lm_head(hidden_states_norm)

            return hidden_states_norm, logits
        finally:
            reset_context()

    def _get_draft_predictions(
        self,
        input_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        """Get logits from draft model."""
        from nanovllm.utils.context import set_context, reset_context

        seq_len = len(input_ids)

        # Use all but last token as input (since we predict next)
        token_ids = input_ids[:-1]
        hidden_input = hidden_states[:-1]
        positions = torch.arange(seq_len - 1, device=self.device)
        
        # Set up context for prefill mode
        draft_seq_len = len(token_ids)
        cu_seqlens_q = torch.tensor([0, draft_seq_len], dtype=torch.int32, device=self.device)
        cu_seqlens_k = torch.tensor([0, draft_seq_len], dtype=torch.int32, device=self.device)
        slot_mapping = torch.full((draft_seq_len,), -1, dtype=torch.int32, device=self.device)

        set_context(
            is_prefill=True,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=draft_seq_len,
            max_seqlen_k=draft_seq_len,
            slot_mapping=slot_mapping,
            context_lens=None,
            block_tables=None,
        )

        try:
            _, predicted_logits = self.draft_model(
                token_ids=token_ids,
                hidden_states=hidden_input,
                positions=positions,
            )
        finally:
            reset_context()

        return predicted_logits

    def train_epoch(
        self,
        dataloader,
        progress_bar: bool = True,
        val_interval: int = 4000,
        val_prompts: list[str] | None = None,
        top_k: list[int] | None = None,
        save_path: str | None = None,
        primary_k: int = 2,
    ) -> dict[str, float]:
        """Train for one epoch.

        Args:
            dataloader: DataLoader with training data
            progress_bar: Whether to show progress bar
            val_interval: Validation interval in steps (default: 4000)
            val_prompts: Validation prompts for acceptance rate evaluation
            top_k: List of k values for top-k acceptance rate evaluation
            save_path: Path to save best model

        Returns:
            Dictionary with average losses for the epoch
        """
        self.draft_model.train()

        total_losses = {
            'total_loss': 0.0,
            'feature_loss': 0.0,
            'token_loss': 0.0,
        }
        num_batches = 0

        iterator = dataloader
        if progress_bar:
            iterator = tqdm(dataloader, desc=f"Epoch {self.global_step // len(dataloader) + 1}")

        for batch in iterator:
            loss_dict = self.train_step(batch)

            for key in total_losses:
                total_losses[key] += loss_dict.get(key, 0.0)
            num_batches += 1

            # Validation every val_interval steps
            if val_prompts is not None and self.target_model is not None and save_path is not None:
                if self.global_step % val_interval == 0:
                    print(f"\n  Running validation at step {self.global_step}...")
                    eval_metrics = self.evaluate_acceptance_rate(
                        val_prompts=val_prompts,
                        max_samples=50,
                        top_k=top_k,
                    )
                    
                    # Save best model based on validation metrics
                    if top_k is None:
                        top_k = [1, 2, 3, 4, 5, 10]
                    primary_metric_key = f'top{primary_k}_acceptance_rate'
                    current_primary = eval_metrics[primary_metric_key]
                    best_primary = getattr(self, 'best_primary_metric', 0)
                    
                    if current_primary > best_primary:
                        self.best_primary_metric = current_primary
                        self.best_mean_accept_seq_len = eval_metrics['mean_accept_seq_len']
                        self.save_checkpoint(save_path)
                        print(f"  Saved best model with {primary_metric_key}={self.best_primary_metric:.4f}, "
                              f"mean_accept_seq_len={self.best_mean_accept_seq_len:.2f}")
                    elif current_primary == best_primary:
                        if eval_metrics['mean_accept_seq_len'] > getattr(self, 'best_mean_accept_seq_len', 0):
                            self.best_mean_accept_seq_len = eval_metrics['mean_accept_seq_len']
                            self.save_checkpoint(save_path)
                            print(f"  Saved best model with {primary_metric_key}={current_primary:.4f} (tie), "
                                  f"mean_accept_seq_len={self.best_mean_accept_seq_len:.2f}")

        # Compute averages
        avg_losses = {
            key: value / num_batches for key, value in total_losses.items()
        }

        return avg_losses
    
    def train(
        self,
        dataloader,
        num_epochs: int = 10,
        eval_fn: Callable | None = None,
        save_path: str | None = None,
        log_interval: int = 10,
        save_per_epoch: bool = False,
        val_prompts: list[str] | None = None,
        val_interval: int = 4000,
        top_k: list[int] | None = None,
        primary_k: int = 2,
    ) -> list[dict[str, float]]:
        """Full training loop.

        Args:
            dataloader: DataLoader with training data
            num_epochs: Number of epochs
            eval_fn: Optional evaluation function
            save_path: Path to save best model (base path, epoch will be appended)
            log_interval: Log every N steps
            save_per_epoch: Whether to save weights at the end of each epoch
            val_prompts: Validation prompts for acceptance rate evaluation
            val_interval: Validation interval in steps (default: 4000)
            top_k: List of k values for top-k acceptance rate evaluation
            primary_k: Which k value to use as primary metric for saving best model (default: 2)

        Returns:
            List of epoch loss dictionaries
        """
        history = []

        if top_k is None:
            top_k = [1, 2, 3, 4, 5, 10]

        for epoch in range(num_epochs):
            # Train for one epoch with step-based validation
            epoch_losses = self.train_epoch(
                dataloader,
                progress_bar=True,
                val_interval=val_interval,
                val_prompts=val_prompts,
                top_k=top_k,
                save_path=save_path,
                primary_k=primary_k,
            )
            epoch_losses['epoch'] = epoch + 1

            # Evaluation
            if eval_fn is not None:
                eval_losses = eval_fn(self.draft_model)
                epoch_losses.update(eval_losses)

            # Acceptance rate evaluation at end of epoch (optional, for logging)
            if val_prompts is not None and self.target_model is not None:
                eval_metrics = self.evaluate_acceptance_rate(
                    val_prompts=val_prompts,
                    max_samples=50,
                    top_k=top_k,
                )
                epoch_losses['acceptance_rate'] = eval_metrics['acceptance_rate']
                epoch_losses['mean_accept_seq_len'] = eval_metrics['mean_accept_seq_len']
                # Add top-k acceptance rates to epoch losses
                for k in top_k:
                    epoch_losses[f'top{k}_acceptance_rate'] = eval_metrics[f'top{k}_acceptance_rate']
                print(f"  Acceptance rate: {eval_metrics['acceptance_rate']:.4f}")
                print(f"  Mean accept seq len: {eval_metrics['mean_accept_seq_len']:.2f}")

            history.append(epoch_losses)

            # Logging
            if (epoch + 1) % log_interval == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}")
                for key, value in epoch_losses.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")

            # Save per epoch
            if save_per_epoch and save_path is not None:
                # Extract base path and add epoch suffix
                base_path, ext = os.path.splitext(save_path)
                epoch_path = f"{base_path}_epoch{epoch + 1}{ext}"
                self.save_checkpoint(epoch_path)
                print(f"  Saved checkpoint for epoch {epoch + 1} to {epoch_path}")

        return history
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint.
        
        Args:
            path: Path to save checkpoint
        """
        # Only save trainable parameters
        state_dict = {
            name: param.cpu()
            for name, param in self.draft_model.named_parameters()
            if name.startswith('fusion_layer') or name.startswith('decoder_layer')
        }
        torch.save(state_dict, path)
        print(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint.
        
        Args:
            path: Path to checkpoint
        """
        state_dict = torch.load(path, map_location='cpu', weights_only=True)
        self.draft_model.load_state_dict(state_dict, strict=False)
        print(f"Loaded checkpoint from {path}")
