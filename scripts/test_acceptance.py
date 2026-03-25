#!/usr/bin/env python3
"""Quick test for acceptance rate evaluation."""

import os
os.environ.setdefault('MASTER_ADDR', 'localhost')
os.environ.setdefault('MASTER_PORT', '29504')
import torch.distributed as dist
dist.init_process_group(backend="gloo", rank=0, world_size=1)

import torch
from transformers import AutoTokenizer, AutoConfig

from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.eagle import load_draft_model
from nanovllm.utils.eagle_trainer import EagleTrainer
from nanovllm.utils.loader import load_model

print("Loading models...")
config = AutoConfig.from_pretrained("models/Qwen3-0.6B")
torch.set_default_dtype(config.torch_dtype)

target_model = Qwen3ForCausalLM(config)
load_model(target_model, "models/Qwen3-0.6B")
target_model = target_model.cuda()
target_model.eval()

tokenizer = AutoTokenizer.from_pretrained("models/Qwen3-0.6B")

draft_model = load_draft_model(config, target_model, "drafts/qwen3_eagle_epoch10.pt", fresh_decoder=True)
draft_model = draft_model.cuda()
draft_model.eval()

print("Creating trainer...")
trainer = EagleTrainer(
    draft_model=draft_model,
    target_model=target_model,
    tokenizer=tokenizer,
    lr=1e-4,
    device='cuda',
)

# Test with a few prompts
test_prompts = [
    "Hello, how are you?",
    "The quick brown fox jumps over the lazy dog.",
    "Once upon a time, there was a brave knight.",
]

print("Testing acceptance rate evaluation...")
acceptance_rate = trainer.evaluate_acceptance_rate(
    val_prompts=test_prompts,
    max_samples=3,
)
print(f"\nAcceptance rate: {acceptance_rate:.4f}")
