"""EAGLE-1 Draft Token Generation Runner.

This module implements feature-level autoregressive draft generation for EAGLE speculative decoding.
The draft model predicts hidden features (not tokens) from the target model, and uses the target
model's LM head to convert features to tokens.

Key components:
1. EagleDraftRunner: Manages draft token generation workflow
2. prepare_draft_input(): Concatenates hidden states + token embeddings
3. generate_draft_tokens(): Autoregressively predicts K draft features/tokens
"""

import torch
from torch import nn
from typing import Optional

from nanovllm.models.eagle import EagleDraftModel
from nanovllm.utils.context import set_context, reset_context


class EagleDraftRunner:
    """EAGLE-1 Draft Token Generation Runner.

    Manages the draft token generation workflow:
    1. Prepare draft input by concatenating hidden states with token embeddings
    2. Autoregressively generate K draft tokens using the draft model
    3. Use target model's LM head to convert predicted features to tokens

    Args:
        draft_model: The EAGLE draft model
        max_speculation_depth: Maximum number of draft tokens to generate
        device: Device to run draft generation on
    """

    def __init__(
        self,
        draft_model: EagleDraftModel,
        max_speculation_depth: int = 4,
        device: Optional[torch.device] = None,
    ):
        self.draft_model = draft_model
        self.max_speculation_depth = max_speculation_depth
        self.device = device or next(draft_model.parameters()).device
        self.dtype = next(draft_model.parameters()).dtype

        # Set draft model to eval mode for inference
        self.draft_model.eval()

    @torch.inference_mode()
    def prepare_draft_input(
        self,
        hidden_states: torch.Tensor,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Prepare draft model input by concatenating hidden states with token embeddings.

        The draft model takes as input:
        - Hidden states from target model's second-to-last layer
        - Token embeddings (shifted by 1 position for autoregressive prediction)

        Args:
            hidden_states: Hidden states from target model, shape [seq_len, hidden_size]
            token_ids: Token IDs for embedding lookup, shape [seq_len]
            positions: Position IDs, shape [seq_len]

        Returns:
            tuple:
                - token_embeds: Token embeddings, shape [seq_len, hidden_size]
                - hidden_states: Hidden states (unchanged), shape [seq_len, hidden_size]
                - positions: Position IDs (unchanged), shape [seq_len]
        """
        # Get token embeddings using draft model's embedding layer
        token_embeds = self.draft_model.embed_tokens(token_ids)

        # Ensure same dtype and device as draft model
        token_embeds = token_embeds.to(dtype=self.dtype, device=self.device)
        hidden_states = hidden_states.to(dtype=self.dtype, device=self.device)
        positions = positions.to(device=self.device)

        return token_embeds, hidden_states, positions

    @torch.inference_mode()
    def generate_draft_tokens(
        self,
        hidden_states: torch.Tensor,
        token_ids: torch.Tensor,
        positions: torch.Tensor,
        num_draft_tokens: Optional[int] = None,
    ) -> tuple[list[int], torch.Tensor]:
        """Autoregressively generate draft tokens using the draft model.

        The draft model predicts hidden features autoregressively:
        1. At each step, concatenate current hidden state with token embedding
        2. Pass through fusion layer + decoder layer to get predicted features
        3. Use target model's LM head to convert features to token logits
        4. Sample/argmax the next token
        5. Use predicted features + new token for next step

        Note: The draft model expects 1D inputs (single sequence).

        Args:
            hidden_states: Initial hidden states from target model,
                          shape [seq_len, hidden_size] or [batch_size, seq_len, hidden_size]
            token_ids: Initial token IDs, shape [seq_len] or [batch_size, seq_len]
            positions: Initial position IDs, shape [seq_len] or [batch_size, seq_len]
            num_draft_tokens: Number of draft tokens to generate (default: max_speculation_depth)

        Returns:
            tuple:
                - draft_token_ids: List of generated draft token IDs (length = num_draft_tokens)
                - draft_features: Tensor of predicted hidden features,
                                 shape [num_draft_tokens, hidden_size]
        """
        num_draft_tokens = num_draft_tokens or self.max_speculation_depth

        # Ensure 2D input (seq_len, hidden_size) for hidden_states
        if hidden_states.dim() == 3:
            # Take first sequence from batch
            hidden_states = hidden_states[0]  # [seq_len, hidden_size]
        assert hidden_states.dim() == 2, f"Expected 2D hidden_states, got {hidden_states.dim()}D"

        # Ensure 1D input (seq_len,) for token_ids and positions
        if token_ids.dim() == 2:
            token_ids = token_ids[0]  # [seq_len]
        if positions.dim() == 2:
            positions = positions[0]  # [seq_len]
        assert token_ids.dim() == 1, f"Expected 1D token_ids, got {token_ids.dim()}D"
        assert positions.dim() == 1, f"Expected 1D positions, got {positions.dim()}D"

        hidden_size = hidden_states.shape[-1]

        # Track generated tokens and features
        draft_token_ids: list[int] = []
        draft_features_list: list[torch.Tensor] = []

        # Initialize current state
        current_hidden = hidden_states  # [seq_len, hidden_size]
        current_tokens = token_ids  # [seq_len]
        current_positions = positions  # [seq_len]

        for step in range(num_draft_tokens):
            # Prepare input: get token embeddings for current tokens
            token_embeds, hidden_in, pos_in = self.prepare_draft_input(
                hidden_states=current_hidden,
                token_ids=current_tokens,
                positions=current_positions,
            )

            # Set up context for FlashAttention
            # For draft generation, we use prefill mode with simple attention
            seq_len = current_tokens.shape[0]
            cu_seqlens = torch.tensor([0, seq_len], dtype=torch.int32, device=self.device)
            max_seqlen = seq_len
            
            set_context(
                is_prefill=True,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
            )

            # Run draft model forward pass (expects 1D inputs)
            # Returns: predicted_features [seq_len, hidden_size],
            #          predicted_logits [seq_len, vocab_size]
            try:
                predicted_features, predicted_logits = self.draft_model(
                    token_ids=current_tokens,
                    hidden_states=hidden_in,
                    positions=pos_in,
                )
            finally:
                # Reset context after forward pass
                reset_context()

            # Get the last position's prediction (next token prediction)
            last_pos_logits = predicted_logits[-1, :]  # [vocab_size]

            # Generate next token using argmax (greedy decoding for draft)
            next_token = last_pos_logits.argmax(dim=-1)  # scalar
            next_token_id = next_token.item()

            draft_token_ids.append(next_token_id)

            # Store predicted features for this step
            predicted_feature = predicted_features[-1, :]  # [hidden_size]
            draft_features_list.append(predicted_feature)

            # Prepare for next step:
            # 1. Append new token to token sequence
            # 2. Use predicted features as new hidden state
            # 3. Increment positions

            # Create new token tensor with appended token
            next_token_tensor = next_token.unsqueeze(0)  # [1]
            current_tokens = torch.cat([current_tokens, next_token_tensor], dim=0)

            # Use predicted features as hidden state for next step
            predicted_feature_expanded = predicted_feature.unsqueeze(0)  # [1, hidden_size]
            current_hidden = torch.cat([current_hidden, predicted_feature_expanded], dim=0)

            # Increment positions
            next_position = current_positions[-1:] + 1  # [1]
            current_positions = torch.cat([current_positions, next_position], dim=0)

        # Stack draft features: [num_draft_tokens, hidden_size]
        draft_features = torch.stack(draft_features_list, dim=0)

        return draft_token_ids, draft_features

    @torch.inference_mode()
    def generate_single_step(
        self,
        hidden_states: torch.Tensor,
        token_id: int,
        position: int,
    ) -> tuple[int, torch.Tensor]:
        """Generate a single draft token.

        This is a convenience method for generating one draft token at a time,
        useful for incremental generation.

        Args:
            hidden_states: Hidden state from target model, shape [seq_len, hidden_size]
                          or [hidden_size]
            token_id: Current token ID
            position: Current position

        Returns:
            tuple:
                - next_token_id: Generated draft token ID
                - predicted_feature: Predicted hidden feature, shape [hidden_size]
        """
        # Reshape inputs for proper processing
        if hidden_states.dim() == 1:
            hidden_states = hidden_states.unsqueeze(0)  # [1, hidden_size]
        elif hidden_states.dim() == 3:
            hidden_states = hidden_states[0]  # [seq_len, hidden_size]

        token_ids = torch.tensor([token_id], device=self.device)
        positions = torch.tensor([position], device=self.device)

        # Prepare input
        token_embeds, hidden_in, pos_in = self.prepare_draft_input(
            hidden_states=hidden_states,
            token_ids=token_ids,
            positions=positions,
        )

        # Set up context for FlashAttention
        set_context(
            is_prefill=True,
            cu_seqlens_q=torch.tensor([0, 1], dtype=torch.int32, device=self.device),
            cu_seqlens_k=torch.tensor([0, 1], dtype=torch.int32, device=self.device),
            max_seqlen_q=1,
            max_seqlen_k=1,
        )

        try:
            # Run draft model
            predicted_features, predicted_logits = self.draft_model(
                token_ids=token_ids,
                hidden_states=hidden_in,
                positions=pos_in,
            )
        finally:
            reset_context()

        # Get prediction
        last_pos_logits = predicted_logits[-1, :]
        next_token = last_pos_logits.argmax(dim=-1)
        next_token_id = next_token.item()

        # Get predicted feature
        predicted_feature = predicted_features[-1, :]

        return next_token_id, predicted_feature


def create_draft_runner(
    draft_model: EagleDraftModel,
    max_speculation_depth: int = 4,
) -> EagleDraftRunner:
    """Create an EAGLE draft runner.

    Args:
        draft_model: The EAGLE draft model
        max_speculation_depth: Maximum number of draft tokens to generate

    Returns:
        EagleDraftRunner instance
    """
    return EagleDraftRunner(
        draft_model=draft_model,
        max_speculation_depth=max_speculation_depth,
    )
