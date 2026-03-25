#!/usr/bin/env python3
"""Debug script to trace NaN source in EAGLE training."""

import argparse
import torch
from transformers import AutoTokenizer, AutoConfig

from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.eagle import EagleDraftModel, load_draft_model
from nanovllm.utils.eagle_data import EagleDataset
from nanovllm.utils.loader import load_model
from nanovllm.utils.context import set_context, reset_context


def check_tensor(name, tensor):
    """Check tensor for NaN/Inf."""
    if tensor is None:
        print(f"  {name}: None")
        return
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    if tensor.dtype.is_floating_point:
        print(f"  {name}: shape={tensor.shape}, dtype={tensor.dtype}, "
              f"mean={tensor.mean().item():.6f}, std={tensor.std().item():.6f}, "
              f"has_nan={has_nan}, has_inf={has_inf}")
    else:
        print(f"  {name}: shape={tensor.shape}, dtype={tensor.dtype}, "
              f"has_nan={has_nan}, has_inf={has_inf}")
    if has_nan or has_inf:
        print(f"    WARNING: {name} has NaN/Inf!")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/Qwen3-0.6B")
    parser.add_argument("--data-path", type=str, 
                        default="data/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=512)
    args = parser.parse_args()

    # Initialize distributed environment
    import os
    os.environ.setdefault('MASTER_ADDR', 'localhost')
    os.environ.setdefault('MASTER_PORT', '29502')
    import torch.distributed as dist
    dist.init_process_group(backend="gloo", rank=0, world_size=1)

    print("=" * 60)
    print("EAGLE Training NaN Debug Script")
    print("=" * 60)

    # Load target model
    print("\n1. Loading target model...")
    config = AutoConfig.from_pretrained(args.model)
    torch.set_default_dtype(config.torch_dtype)
    
    target_model = Qwen3ForCausalLM(config)
    load_model(target_model, args.model)
    target_model = target_model.cuda()
    target_model.eval()
    
    for param in target_model.parameters():
        param.requires_grad = False
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    print(f"   Model loaded: {sum(p.numel() for p in target_model.parameters()):,} params")

    # Load draft model
    print("\n2. Creating draft model...")
    draft_model = load_draft_model(config, target_model, fresh_decoder=True)
    draft_model = draft_model.cuda()
    draft_model.train()
    print(f"   Draft model created")

    # Create dataset
    print("\n3. Creating dataset...")
    test_prompts = [
        "Hello, how are you?",
        "The quick brown fox jumps over the lazy dog.",
        "Once upon a time, there was a brave knight.",
    ]
    
    dataset = EagleDataset(
        target_model=target_model,
        tokenizer=tokenizer,
        prompts=test_prompts,
        max_seq_len=args.max_seq_len,
        noise_std=0.1,
        device='cuda',
        dtype=config.torch_dtype,
    )
    print(f"   Dataset created with {len(dataset)} samples")

    # Get a sample and trace through the pipeline
    print("\n4. Testing data collection (target model forward)...")
    sample = dataset[0]
    
    check_tensor("token_ids", sample.token_ids)
    check_tensor("hidden_states (input)", sample.hidden_states)
    check_tensor("positions", sample.positions)
    check_tensor("target_features", sample.target_features)
    check_tensor("target_tokens", sample.target_tokens)

    # Test draft model forward
    print("\n5. Testing draft model forward pass...")
    
    # Get first few tokens for testing
    test_len = min(32, len(sample.token_ids))
    token_ids = sample.token_ids[:test_len].cuda().unsqueeze(0)  # [1, seq_len]
    hidden_states = sample.hidden_states[:test_len].cuda().unsqueeze(0)  # [1, seq_len, hidden]
    positions = sample.positions[:test_len].cuda().unsqueeze(0)  # [1, seq_len]
    
    print(f"   Input shapes:")
    check_tensor("token_ids", token_ids)
    check_tensor("hidden_states", hidden_states)
    check_tensor("positions", positions)
    
    # Flatten for draft model
    token_ids_flat = token_ids.reshape(-1)
    hidden_states_flat = hidden_states.reshape(-1, hidden_states.size(-1))
    positions_flat = positions.reshape(-1)
    
    print(f"   Flattened shapes:")
    check_tensor("token_ids_flat", token_ids_flat)
    check_tensor("hidden_states_flat", hidden_states_flat)
    check_tensor("positions_flat", positions_flat)
    
    # Set context
    total_len = len(token_ids_flat)
    cu_seqlens_q = torch.tensor([0, total_len], dtype=torch.int32, device='cuda')
    cu_seqlens_k = torch.tensor([0, total_len], dtype=torch.int32, device='cuda')
    slot_mapping = torch.full((total_len,), -1, dtype=torch.int32, device='cuda')
    
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
        # Forward through draft model
        print("\n6. Running draft model forward...")
        with torch.no_grad():
            # Step through draft model manually
            print("   Step 1: Token embedding")
            token_embeds = draft_model.embed_tokens(token_ids_flat)
            check_tensor("token_embeds", token_embeds)
            
            print("   Step 2: Fusion layer")
            fused_features = draft_model.fusion_layer(token_embeds, hidden_states_flat)
            check_tensor("fused_features", fused_features)
            
            print("   Step 3: Decoder layer input layernorm")
            normalized = draft_model.decoder_layer.input_layernorm(fused_features)
            check_tensor("normalized (after input_layernorm)", normalized)
            
            print("   Step 4: Self attention (using full Qwen3Attention.forward)")
            # Debug attention inputs
            print(f"      positions_flat shape: {positions_flat.shape}")
            print(f"      normalized shape: {normalized.shape}")
            print(f"      normalized stats: mean={normalized.mean().item():.6f}, std={normalized.std().item():.6f}")
            
            # Use the full attention module forward (includes qkv_proj, rotary, flash_attn, o_proj)
            self_attn = draft_model.decoder_layer.self_attn
            attn_output = self_attn(positions_flat, normalized)
            check_tensor("attn_output (full attention)", attn_output)
            
            print("   Step 5: Residual add")
            residual = fused_features
            post_attn = attn_output + residual
            check_tensor("post_attn (residual added)", post_attn)
            
            print("   Step 6: Post-attention layernorm")
            post_norm = draft_model.decoder_layer.post_attention_layernorm(post_attn)
            check_tensor("post_norm", post_norm)
            
            print("   Step 7: MLP")
            mlp_output = draft_model.decoder_layer.mlp(post_norm)
            check_tensor("mlp_output", mlp_output)
            
            print("   Step 8: Final residual")
            final_output = mlp_output + post_norm
            check_tensor("final_output (decoder output)", final_output)
            
            print("   Step 9: LM head")
            predicted_logits = torch.nn.functional.linear(final_output, draft_model.lm_head.weight)
            check_tensor("predicted_logits", predicted_logits)
            
            # Now test the full forward
            print("\n7. Testing full draft model forward()...")
            predicted_features, predicted_logits = draft_model(
                token_ids=token_ids_flat,
                hidden_states=hidden_states_flat,
                positions=positions_flat,
            )
            check_tensor("predicted_features (full forward)", predicted_features)
            check_tensor("predicted_logits (full forward)", predicted_logits)
            
    finally:
        reset_context()
    
    # Test loss computation
    print("\n8. Testing loss computation...")
    target_features_flat = sample.target_features[:test_len].cuda().reshape(-1, sample.target_features.size(-1))
    target_tokens_flat = sample.target_tokens[:test_len].cuda().reshape(-1)
    
    check_tensor("target_features_flat", target_features_flat)
    check_tensor("target_tokens_flat", target_tokens_flat)
    
    feature_loss = torch.nn.functional.smooth_l1_loss(predicted_features, target_features_flat)
    token_loss = torch.nn.functional.cross_entropy(predicted_logits, target_tokens_flat)
    
    print(f"   feature_loss: {feature_loss.item():.6f}")
    print(f"   token_loss: {token_loss.item():.6f}")
    
    total_loss = feature_loss + 0.1 * token_loss
    print(f"   total_loss: {total_loss.item():.6f}")
    
    print("\n" + "=" * 60)
    print("Debug complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
