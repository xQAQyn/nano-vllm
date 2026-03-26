"""Speculative Sampling for EAGLE-1 Speculative Decoding.

This module implements the speculative sampling algorithm used to verify draft tokens
from the EAGLE draft model against the target model's probability distribution.

The key insight is that we can accept draft tokens with probability min(1, p(target)/p(draft)),
which preserves the target model's output distribution while potentially accepting multiple
tokens in a single forward pass.

Algorithm (Multi-round Speculative Sampling):
1. Draft model generates K tokens autoregressively
2. Target model computes probabilities for all K draft tokens in one forward pass
3. For each draft token i:
   - Compute acceptance probability: min(1, p_target(token_i) / p_draft(token_i))
   - Accept with this probability
   - On rejection, sample from normalized residual distribution
4. Continue until a token is rejected or K tokens are accepted
"""

import torch
from torch import nn


class SpeculativeSampler(nn.Module):
    """Speculative sampler for verifying draft tokens.

    Implements multi-round speculative sampling:
    1. Compute acceptance probabilities for draft tokens
    2. Accept/reject based on min(1, p_target/p_draft)
    3. Handle rejection by sampling from residual distribution
    """

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(
        self,
        target_logits: torch.Tensor,
        draft_logits: torch.Tensor,
        draft_token_ids: list[int],
        temperature: float = 1.0,
    ) -> tuple[list[int], list[bool], torch.Tensor | None]:
        """Perform speculative sampling to verify draft tokens.

        Args:
            target_logits: Logits from target model for draft token positions,
                          shape [K, vocab_size] where K is number of draft tokens
            draft_logits: Logits from draft model for draft token positions,
                         shape [K, vocab_size]
            draft_token_ids: List of draft token IDs to verify, length K
            temperature: Sampling temperature (default: 1.0)

        Returns:
            tuple:
                - accepted_tokens: List of accepted token IDs
                - accepted_mask: Boolean mask indicating which draft tokens were accepted
                - resampled_token: Token sampled from residual distribution on first rejection,
                                  or None if all tokens accepted
        """
        K = len(draft_token_ids)
        vocab_size = target_logits.shape[-1]

        # Convert logits to probabilities with temperature scaling
        target_probs = torch.softmax(target_logits.float() / temperature, dim=-1)  # [K, vocab_size]
        draft_probs = torch.softmax(draft_logits.float() / temperature, dim=-1)  # [K, vocab_size]

        accepted_tokens = []
        accepted_mask = []
        resampled_token = None

        for i in range(K):
            draft_token = draft_token_ids[i]
            p_target = target_probs[i, draft_token]
            p_draft = draft_probs[i, draft_token]

            # Compute acceptance probability: min(1, p_target / p_draft)
            if p_draft > 0:
                acceptance_prob = torch.clamp(p_target / p_draft, max=1.0)
            else:
                # If draft probability is 0 but target > 0, always reject
                # (draft made an impossible prediction)
                acceptance_prob = torch.tensor(0.0, device=target_probs.device)

            # Sample to decide acceptance
            rand = torch.rand_like(acceptance_prob)
            accept = rand < acceptance_prob

            if accept:
                # Accept the draft token
                accepted_tokens.append(draft_token)
                accepted_mask.append(True)
            else:
                # Reject: sample from residual distribution
                # Residual = max(0, p_target - p_draft), normalized
                residual = torch.maximum(
                    target_probs[i] - draft_probs[i],
                    torch.tensor(0.0, device=target_probs.device)
                )
                residual_sum = residual.sum()

                if residual_sum > 1e-10:
                    # Normalize and sample from residual
                    residual_probs = residual / residual_sum
                    resampled_token = torch.multinomial(residual_probs, num_samples=1)
                else:
                    # Fallback: sample from target distribution
                    resampled_token = torch.multinomial(target_probs[i], num_samples=1)

                accepted_mask.append(False)
                # Stop after first rejection
                break

        return accepted_tokens, accepted_mask, resampled_token

    def verify_tokens(
        self,
        target_logits: torch.Tensor,
        draft_logits: torch.Tensor,
        draft_token_ids: list[int],
        temperature: float = 1.0,
    ) -> tuple[list[int], list[bool], int | None]:
        """Verify draft tokens using speculative sampling.

        This is a convenience wrapper that returns Python types.
        Uses the non-compiled implementation to avoid torch.compile issues.

        Args:
            target_logits: Logits from target model, shape [K, vocab_size]
            draft_logits: Logits from draft model, shape [K, vocab_size]
            draft_token_ids: List of draft token IDs to verify
            temperature: Sampling temperature

        Returns:
            tuple:
                - accepted_tokens: List of accepted token IDs
                - accepted_mask: Boolean mask (True = accepted, False = rejected)
                - resampled_token: Resampled token on rejection, or None
        """
        # Use non-compiled version to avoid torch.compile issues
        accepted_tokens, accepted_mask, resampled_token = self._sample_no_compile(
            target_logits, draft_logits, draft_token_ids, temperature
        )
        resampled_id = int(resampled_token.item()) if resampled_token is not None else None
        return accepted_tokens, accepted_mask, resampled_id
    
    def _sample_no_compile(
        self,
        target_logits: torch.Tensor,
        draft_logits: torch.Tensor,
        draft_token_ids: list[int],
        temperature: float = 1.0,
    ) -> tuple[list[int], list[bool], torch.Tensor | None]:
        """Non-compiled speculative sampling implementation.
        
        Args:
            target_logits: Logits from target model, shape [K, vocab_size]
            draft_logits: Logits from draft model, shape [K, vocab_size]
            draft_token_ids: List of draft token IDs to verify
            temperature: Sampling temperature
            
        Returns:
            tuple:
                - accepted_tokens: List of accepted token IDs
                - accepted_mask: Boolean mask
                - resampled_token: Resampled token tensor or None
        """
        K = len(draft_token_ids)

        # Convert logits to probabilities with temperature scaling
        target_probs = torch.softmax(target_logits.float() / temperature, dim=-1)
        draft_probs = torch.softmax(draft_logits.float() / temperature, dim=-1)

        accepted_tokens = []
        accepted_mask = []
        resampled_token = None

        for i in range(K):
            draft_token = draft_token_ids[i]
            p_target = target_probs[i, draft_token]
            p_draft = draft_probs[i, draft_token]

            # Compute acceptance probability: min(1, p_target / p_draft)
            if p_draft > 1e-10:  # Avoid division by zero
                acceptance_prob = min(1.0, (p_target / p_draft).item())
            else:
                acceptance_prob = 0.0

            # Sample to decide acceptance
            rand = torch.rand(1, device=target_probs.device).item()
            accept = rand < acceptance_prob

            if accept:
                accepted_tokens.append(draft_token)
                accepted_mask.append(True)
            else:
                # Reject: sample from residual distribution
                residual = torch.maximum(
                    target_probs[i] - draft_probs[i],
                    torch.tensor(0.0, device=target_probs.device)
                )
                residual_sum = residual.sum().item()

                if residual_sum > 1e-10:
                    residual_probs = (residual / residual_sum).cpu()
                    resampled_token = torch.multinomial(residual_probs, num_samples=1)
                else:
                    target_probs_cpu = target_probs[i].cpu()
                    resampled_token = torch.multinomial(target_probs_cpu, num_samples=1)

                accepted_mask.append(False)
                break

        return accepted_tokens, accepted_mask, resampled_token


class SpeculativeSamplerNoCompile(nn.Module):
    """Non-compiled version of speculative sampler for debugging.

    Same algorithm as SpeculativeSampler but without torch.compile.
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        target_logits: torch.Tensor,
        draft_logits: torch.Tensor,
        draft_token_ids: list[int],
        temperature: float = 1.0,
    ) -> tuple[list[int], list[bool], torch.Tensor | None]:
        """Perform speculative sampling to verify draft tokens.

        Args:
            target_logits: Logits from target model for draft token positions,
                          shape [K, vocab_size]
            draft_logits: Logits from draft model for draft token positions,
                         shape [K, vocab_size]
            draft_token_ids: List of draft token IDs to verify, length K
            temperature: Sampling temperature (default: 1.0)

        Returns:
            tuple:
                - accepted_tokens: List of accepted token IDs
                - accepted_mask: Boolean mask indicating which draft tokens were accepted
                - resampled_token: Token sampled from residual distribution on first rejection,
                                  or None if all tokens accepted
        """
        K = len(draft_token_ids)

        # Convert logits to probabilities with temperature scaling
        target_probs = torch.softmax(target_logits.float() / temperature, dim=-1)
        draft_probs = torch.softmax(draft_logits.float() / temperature, dim=-1)

        accepted_tokens = []
        accepted_mask = []
        resampled_token = None

        for i in range(K):
            draft_token = draft_token_ids[i]
            p_target = target_probs[i, draft_token]
            p_draft = draft_probs[i, draft_token]

            # Compute acceptance probability: min(1, p_target / p_draft)
            if p_draft > 1e-10:  # Avoid division by zero
                acceptance_prob = min(1.0, (p_target / p_draft).item())
            else:
                acceptance_prob = 0.0

            # Sample to decide acceptance
            rand = torch.rand(1, device=target_probs.device).item()
            accept = rand < acceptance_prob

            if accept:
                accepted_tokens.append(draft_token)
                accepted_mask.append(True)
            else:
                # Reject: sample from residual distribution
                residual = torch.maximum(
                    target_probs[i] - draft_probs[i],
                    torch.tensor(0.0, device=target_probs.device)
                )
                residual_sum = residual.sum().item()

                if residual_sum > 1e-10:
                    residual_probs = (residual / residual_sum).cpu()
                    resampled_token = torch.multinomial(residual_probs, num_samples=1)
                else:
                    target_probs_cpu = target_probs[i].cpu()
                    resampled_token = torch.multinomial(target_probs_cpu, num_samples=1)

                accepted_mask.append(False)
                break

        return accepted_tokens, accepted_mask, resampled_token


def create_speculative_sampler(use_compile: bool = True) -> SpeculativeSampler | SpeculativeSamplerNoCompile:
    """Create a speculative sampler.

    Args:
        use_compile: If True, use torch.compile for faster inference

    Returns:
        SpeculativeSampler or SpeculativeSamplerNoCompile instance
    """
    if use_compile:
        return SpeculativeSampler()
    else:
        return SpeculativeSamplerNoCompile()
