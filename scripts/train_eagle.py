#!/usr/bin/env python3
"""Training script for EAGLE draft model.

Usage:
    python scripts/train_eagle.py \
        --model models/Qwen3-0.6B \
        --output drafts/qwen3_0.6b_eagle.pt \
        --epochs 10 \
        --batch-size 4 \
        --data-path data/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json

This script:
1. Loads the target model in eval mode
2. Loads training data from ShareGPT dataset
3. Trains the draft model to predict hidden features
4. Evaluates acceptance rate per epoch
5. Saves the trained draft model weights per epoch
"""

import argparse
import os
import torch
from transformers import AutoTokenizer, AutoConfig

from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.eagle import EagleDraftModel, load_draft_model
from nanovllm.utils.eagle_data import create_training_dataloader
from nanovllm.utils.eagle_trainer import EagleTrainer
from nanovllm.utils.loader import load_model
from nanovllm.utils.sharegpt_loader import create_train_val_split


def load_training_data(
    data_path: str,
    train_samples: int = 2000,
    val_samples: int = 200,
    seed: int = 42,
) -> tuple[list[str], list[str]]:
    """Load training and validation prompts from ShareGPT dataset.

    Args:
        data_path: Path to the ShareGPT JSON file
        train_samples: Number of training samples to use
        val_samples: Number of validation samples to use
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_prompts, val_prompts)
    """
    print(f"Loading training data from {data_path}...")
    train_prompts, val_prompts = create_train_val_split(
        file_path=data_path,
        train_samples=train_samples,
        val_samples=val_samples,
        seed=seed,
    )
    return train_prompts, val_prompts


def load_target_model(model_path: str, tensor_parallel_size: int = 1):
    """Load target model in eval mode.
    
    Returns:
        tuple: (model, config)
    """
    print(f"Loading target model from {model_path}...")
    
    # Initialize distributed environment for tensor parallelism
    if tensor_parallel_size > 1:
        # Multi-GPU case - requires external process group initialization
        # This script currently only supports single GPU training
        raise ValueError(
            "Multi-GPU training (tensor_parallel_size > 1) is not supported in this script. "
            "Please use tensor_parallel_size=1."
        )
    else:
        # Single GPU case - initialize a single-process process group
        import torch.distributed as dist
        if not dist.is_initialized():
            # Initialize a single-process process group for CPU-only operations
            # This is needed for VocabParallelEmbedding to work
            # Use localhost with explicit port to avoid env variable requirements
            import os
            os.environ.setdefault('MASTER_ADDR', 'localhost')
            os.environ.setdefault('MASTER_PORT', '29501')
            dist.init_process_group(backend="gloo", rank=0, world_size=1)
    
    config = AutoConfig.from_pretrained(model_path)
    
    # Set dtype
    torch.set_default_dtype(config.torch_dtype)
    
    # Load model on CPU first, then move to CUDA
    model = Qwen3ForCausalLM(config)
    load_model(model, model_path)
    
    # Move to CUDA if available
    if torch.cuda.is_available():
        model = model.cuda()
    
    model.eval()

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    print(f"Target model loaded with {sum(p.numel() for p in model.parameters()):,} parameters")
    return model, config


def create_draft_model(target_model: Qwen3ForCausalLM, config, draft_model_path: str | None = None):
    """Create or load draft model."""
    print("Creating EAGLE draft model...")

    draft_model = load_draft_model(config, target_model, draft_model_path)

    num_trainable = draft_model.count_trainable_params()
    print(f"Draft model created with {num_trainable:,} trainable parameters")
    print(f"  (Fusion layer + Decoder layer)")

    return draft_model


def evaluate_model(draft_model, dataloader, device='cuda'):
    """Evaluate draft model on validation data."""
    draft_model.eval()
    total_feature_loss = 0.0
    total_token_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            token_ids = batch['token_ids'].to(device)
            hidden_states = batch['hidden_states'].to(device)
            positions = batch['positions'].to(device)
            target_features = batch['target_features'].to(device)
            target_tokens = batch['target_tokens'].to(device)
            
            batch_size, seq_len = token_ids.shape
            token_ids_flat = token_ids.view(-1)
            hidden_states_flat = hidden_states.view(-1, hidden_states.size(-1))
            positions_flat = positions.view(-1)
            
            predicted_features, predicted_logits = draft_model(
                token_ids=token_ids_flat,
                hidden_states=hidden_states_flat,
                positions=positions_flat,
            )
            
            predicted_features = predicted_features.view(batch_size, seq_len, -1)
            predicted_logits = predicted_logits.view(batch_size, seq_len, -1)
            
            # Compute losses
            feature_loss = torch.nn.functional.smooth_l1_loss(
                predicted_features.view(-1, predicted_features.size(-1)),
                target_features.view(-1, target_features.size(-1)),
            )
            token_loss = torch.nn.functional.cross_entropy(
                predicted_logits.view(-1, predicted_logits.size(-1)),
                target_tokens.view(-1),
            )
            
            total_feature_loss += feature_loss.item()
            total_token_loss += token_loss.item()
            num_batches += 1
    
    draft_model.train()
    
    return {
        'val_feature_loss': total_feature_loss / num_batches,
        'val_token_loss': total_token_loss / num_batches,
    }


def main():
    parser = argparse.ArgumentParser(description="Train EAGLE draft model")
    parser.add_argument(
        "--model",
        type=str,
        default="models/Qwen3-0.6B",
        help="Path to target model",
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        default=None,
        help="Path to existing draft model weights (optional)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="drafts/eagle_draft.pt",
        help="Output path for trained draft model (best model by acceptance rate)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Training batch size",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=2048,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.1,
        help="Noise standard deviation for data augmentation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json",
        help="Path to ShareGPT training data",
    )
    parser.add_argument(
        "--train-samples",
        type=int,
        default=92000,
        help="Number of training samples to use from ShareGPT",
    )
    parser.add_argument(
        "--val-samples",
        type=int,
        default=2000,
        help="Number of validation samples for acceptance rate evaluation",
    )
    parser.add_argument(
        "--save-per-epoch",
        action="store_true",
        help="Save model weights at the end of each epoch",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory to cache preprocessed training data (optional, speeds up subsequent runs)",
    )
    parser.add_argument(
        "--use-cache",
        action="store_true",
        help="Use cached training data if available (requires --cache-dir)",
    )
    parser.add_argument(
        "--val-interval",
        type=int,
        default=4000,
        help="Validation interval in steps (default: 4000)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 5, 10],
        help="List of k values for top-k acceptance rate evaluation (default: [1, 2, 3, 4, 5, 10])",
    )
    parser.add_argument(
        "--primary-k",
        type=int,
        default=2,
        help="Which top-k value to use as primary metric for saving best model (default: 2)",
    )

    args = parser.parse_args()

    # Create output directory
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    # Load target model and config
    target_model, config = load_target_model(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # Create draft model
    draft_model = create_draft_model(target_model, config, args.draft_model)

    # Load training and validation data from ShareGPT
    train_prompts, val_prompts = load_training_data(
        data_path=args.data_path,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        seed=args.seed,
    )

    print(f"Using {len(train_prompts)} prompts for training data generation")
    print(f"Using {len(val_prompts)} prompts for validation")

    # Create training dataloader
    print("Collecting training data from target model...")
    print("(Using lazy loading - data will be processed on-demand)")
    if args.use_cache and args.cache_dir:
        print(f"Caching enabled: {args.cache_dir}")
    
    train_loader = create_training_dataloader(
        target_model=target_model,
        tokenizer=tokenizer,
        prompts=train_prompts,
        batch_size=args.batch_size,
        max_seq_len=args.max_seq_len,
        noise_std=args.noise_std,
        shuffle=True,
        device=args.device,  # Use CUDA device
        cache_dir=args.cache_dir,
        use_cache=args.use_cache,
    )
    print(f"Created training dataloader with {len(train_loader.dataset)} samples")

    # Create trainer with target model and tokenizer for acceptance rate evaluation
    trainer = EagleTrainer(
        draft_model=draft_model,
        target_model=target_model,
        tokenizer=tokenizer,
        lr=args.lr,
        device=args.device,
    )

    # Train
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Train samples: {len(train_prompts)}")
    print(f"Val samples: {len(val_prompts)}")
    print(f"Validation interval: {args.val_interval} steps")
    print(f"Top-k values: {args.top_k}")
    print(f"Primary metric: top-{args.primary_k} acceptance rate")
    print("-" * 50)

    history = trainer.train(
        dataloader=train_loader,
        num_epochs=args.epochs,
        save_path=args.output,
        log_interval=1,
        save_per_epoch=args.save_per_epoch,
        val_prompts=val_prompts,
        val_interval=args.val_interval,
        top_k=args.top_k,
        primary_k=args.primary_k,
    )

    # Print final results
    print("\n" + "=" * 50)
    print("Training completed!")
    if hasattr(trainer, 'best_primary_metric'):
        print(f"Best primary metric (top-k acceptance rate): {trainer.best_primary_metric:.4f}")
    if hasattr(trainer, 'best_mean_accept_seq_len'):
        print(f"Best mean accept sequence length: {trainer.best_mean_accept_seq_len:.2f}")
    elif hasattr(trainer, 'best_acceptance_rate'):
        print(f"Best acceptance rate: {trainer.best_acceptance_rate:.4f}")
    print(f"Best model saved to: {args.output}")

    # Print training history summary
    print("\nTraining history:")
    for epoch_data in history:
        epoch = epoch_data.get('epoch', '?')
        total_loss = epoch_data.get('total_loss', 0)
        acceptance_rate = epoch_data.get('acceptance_rate', None)
        mean_accept_seq_len = epoch_data.get('mean_accept_seq_len', None)
        # Print top-k acceptance rates if available
        top_k_rates = {k: epoch_data.get(f'top{k}_acceptance_rate') for k in args.top_k}
        
        if mean_accept_seq_len is not None:
            print(f"  Epoch {epoch}: total_loss = {total_loss:.4f}, acceptance_rate = {acceptance_rate:.4f}, mean_accept_seq_len = {mean_accept_seq_len:.2f}")
        elif acceptance_rate is not None:
            print(f"  Epoch {epoch}: total_loss = {total_loss:.4f}, acceptance_rate = {acceptance_rate:.4f}")
        else:
            print(f"  Epoch {epoch}: total_loss = {total_loss:.4f}")
        
        # Print top-k acceptance rates if available
        if any(v is not None for v in top_k_rates.values()):
            top_k_str = ", ".join([f"top{k}={v:.4f}" for k, v in top_k_rates.items() if v is not None])
            print(f"    {top_k_str}")


if __name__ == "__main__":
    main()
