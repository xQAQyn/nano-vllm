#!/usr/bin/env python3
"""Debug acceptance rate evaluation."""

import argparse
import torch
from transformers import AutoTokenizer, AutoConfig

from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.eagle import EagleDraftModel, load_draft_model
from nanovllm.utils.loader import load_model
from nanovllm.utils.context import set_context, reset_context


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/Qwen3-0.6B")
    parser.add_argument("--draft-model", type=str, default="drafts/qwen3_eagle.pt")
    args = parser.parse_args()

    # Initialize distributed environment
    import os
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '29503')
    import torch.distributed as dist
    dist.init_process_group(backend="gloo", rank=0, world_size=1)

    print("=" * 60)
    print("Acceptance Rate Debug")
    print("=" * 60)

    # Load target model
    print("\n1. Loading target model...")
    config = AutoConfig.from_pretrained(args.model)
    torch.set_default_dtype(config.torch_dtype)
    
    target_model = Qwen3ForCausalLM(config)
    load_model(target_model, args.model)
    target_model = target_model.cuda()
    target_model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"   Model loaded")

    # Load draft model
    print("\n2. Loading draft model...")
    draft_model = load_draft_model(config, target_model, args.draft_model, fresh_decoder=True)
    draft_model = draft_model.cuda()
    draft_model.eval()
    print(f"   Draft model loaded from {args.draft_model}")

    # Test prompt
    test_prompt = "Hello, how are you?"
    print(f"\n3. Testing with prompt: '{test_prompt}'")
    
    input_ids = tokenizer.encode(
        test_prompt,
        return_tensors='pt',
        truncation=True,
        max_length=512,
    ).squeeze(0).cuda()
    
    print(f"   Input tokens: {input_ids.shape}")
    print(f"   Input token IDs: {input_ids.tolist()}")

    # Get target hidden states and logits
    print("\n4. Getting target model outputs...")
    seq_len = len(input_ids)
    
    cu_seqlens_q = torch.tensor([0, seq_len], dtype=torch.int32, device='cuda')
    cu_seqlens_k = torch.tensor([0, seq_len], dtype=torch.int32, device='cuda')
    slot_mapping = torch.full((seq_len,), -1, dtype=torch.int32, device='cuda')

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
        # Get hidden states from second-to-last layer
        hidden_states = target_model.model.embed_tokens(input_ids)
        residual = None

        # Run through all layers except the last one
        for layer in target_model.model.layers[:-1]:
            hidden_states, residual = layer(
                positions=torch.arange(seq_len, device='cuda'),
                hidden_states=hidden_states,
                residual=residual,
            )

        # Apply final norm
        hidden_states, _ = target_model.model.norm(hidden_states, residual)
        print(f"   Hidden states (penultimate layer): {hidden_states.shape}")
        print(f"   Hidden stats: mean={hidden_states.mean().item():.6f}, std={hidden_states.std().item():.6f}")

        # Get logits from last layer
        last_layer_hidden, _ = target_model.model.layers[-1](
            positions=torch.arange(seq_len, device='cuda'),
            hidden_states=hidden_states,
            residual=None,
        )
        # When residual is None, norm returns single tensor
        last_layer_hidden_norm = target_model.model.norm(last_layer_hidden)
        target_logits = target_model.lm_head(last_layer_hidden_norm)
        print(f"   Target logits: {target_logits.shape}")
        
        target_tokens = target_logits.argmax(dim=-1)
        print(f"   Target predicted tokens: {target_tokens.tolist()}")
        
        # Also get target tokens from penultimate layer (for comparison)
        penultimate_logits = target_model.lm_head(hidden_states)
        penultimate_tokens = penultimate_logits.argmax(dim=-1)
        print(f"   Penultimate layer tokens: {penultimate_tokens.tolist()}")
        
    finally:
        reset_context()

    # Get draft model predictions
    print("\n5. Getting draft model predictions...")
    
    # Use all but last token as input
    token_ids = input_ids[:-1]
    hidden_input = hidden_states[:-1]  # Exclude last position
    positions = torch.arange(seq_len - 1, device='cuda')
    
    print(f"   Draft input token_ids: {token_ids.shape}")
    print(f"   Draft input hidden: {hidden_input.shape}")
    
    # Set context for draft model prefill
    draft_seq_len = len(token_ids)
    cu_seqlens_q = torch.tensor([0, draft_seq_len], dtype=torch.int32, device='cuda')
    cu_seqlens_k = torch.tensor([0, draft_seq_len], dtype=torch.int32, device='cuda')
    slot_mapping = torch.full((draft_seq_len,), -1, dtype=torch.int32, device='cuda')

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
        with torch.no_grad():
            predicted_features, predicted_logits = draft_model(
                token_ids=token_ids,
                hidden_states=hidden_input,
                positions=positions,
            )
    finally:
        reset_context()
    
    print(f"   Draft predicted logits: {predicted_logits.shape}")
    draft_tokens = predicted_logits.argmax(dim=-1)
    print(f"   Draft predicted tokens: {draft_tokens.tolist()}")
    
    # Compare
    print("\n6. Comparing predictions...")
    
    # Target tokens from penultimate layer (excluding last position)
    target_logits_from_hidden = target_model.lm_head(hidden_states)
    target_tokens_from_hidden = target_logits_from_hidden.argmax(dim=-1)
    
    print(f"   Target tokens (from penultimate, 0 to {seq_len-2}): {target_tokens_from_hidden[:-1].tolist()}")
    print(f"   Draft tokens (0 to {seq_len-2}): {draft_tokens.tolist()}")
    
    matches = (target_tokens_from_hidden[:-1] == draft_tokens).sum().item()
    total = len(draft_tokens)
    print(f"\n   Matches: {matches}/{total} = {matches/total:.4f}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
