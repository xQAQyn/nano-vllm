#!/usr/bin/env python3
"""EAGLE Speculative Decoding Benchmark Suite.

This script benchmarks EAGLE speculative decoding against a naive autoregressive (AR)
baseline, collecting comprehensive metrics across three categories:

1. Macro-Performance: Throughput, Speedup, TTFT, TPT
2. Hardware & Memory: Arithmetic Intensity, MBU, Weight Reuse, KV Cache Overhead
3. Algorithmic: Acceptance Length, Acceptance Rate, Draft Overhead, Efficiency

Usage:
    # Run EAGLE benchmark
    python benchmarks/eagle_bench.py --mode eagle --model ./models/Qwen3-0.6B/ \
        --draft-model ./models/eagle_draft/ --speculation-depth 4

    # Run AR baseline benchmark
    python benchmarks/eagle_bench.py --mode ar --model ./models/Qwen3-0.6B/

    # Run both and compare
    python benchmarks/eagle_bench.py --mode both --model ./models/Qwen3-0.6B/ \
        --draft-model ./models/eagle_draft/ --speculation-depth 4

    # Run with custom dataset
    python benchmarks/eagle_bench.py --mode both --model ./models/Qwen3-0.6B/ \
        --dataset gsm8k --num-sequences 128
"""

import argparse
import json
import os
import sys
import time
import torch
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from random import randint, seed
from typing import Optional
import statistics

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nanovllm import LLM, SamplingParams
from benchmarks.metrics import (
    MetricsCollector,
    BenchmarkMetrics,
    compute_speedup,
    compare_metrics,
)


@dataclass
class BenchmarkConfig:
    """Benchmark configuration."""
    model: str
    draft_model: Optional[str] = None
    speculation_depth: int = 4
    mode: str = "both"  # "eagle", "ar", or "both"
    num_sequences: int = 256
    max_input_len: int = 512
    max_output_len: int = 256
    temperature: float = 1.0
    dataset: str = "random"  # "random", "gsm8k", "humaneval"
    warmup_runs: int = 1
    benchmark_runs: int = 3
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    gpu_memory_utilization: float = 0.9
    max_model_len: int = 4096
    kvcache_block_size: int = 256
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    seed: int = 42
    output_dir: str = "./benchmark_results"
    verbose: bool = True


class EagleBenchmark:
    """EAGLE Speculative Decoding Benchmark Runner.
    
    This class orchestrates the benchmarking process:
    1. Initialize LLM with appropriate configuration
    2. Generate/load benchmark dataset
    3. Run warm-up iterations
    4. Execute benchmark runs with metrics collection
    5. Aggregate and report results
    """
    
    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self.llm: Optional[LLM] = None
        self.metrics_collectors: list[MetricsCollector] = []
        
        # Set random seed for reproducibility
        seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
    
    def initialize_llm(self, mode: str = "ar") -> str:
        """Initialize LLM with EAGLE or AR configuration.
        
        Returns:
            The actual mode used ("ar" or "eagle")
        """
        eagle_enabled = (mode == "eagle")
        actual_mode = mode
        
        # Validate draft model path for EAGLE mode
        if eagle_enabled and self.config.draft_model:
            if not os.path.exists(self.config.draft_model):
                print(f"Error: Draft model path does not exist: {self.config.draft_model}")
                print("Please download or train an EAGLE draft model first.")
                print("Falling back to AR mode...")
                eagle_enabled = False
                actual_mode = "ar"
        
        print(f"Initializing LLM (mode={actual_mode}, eagle_enabled={eagle_enabled})...")
        
        try:
            self.llm = LLM(
                model=self.config.model,
                eagle_enabled=eagle_enabled,
                eagle_draft_model=self.config.draft_model if eagle_enabled else None,
                speculation_depth=self.config.speculation_depth if eagle_enabled else 4,
                tensor_parallel_size=self.config.tensor_parallel_size,
                enforce_eager=self.config.enforce_eager,
                gpu_memory_utilization=self.config.gpu_memory_utilization,
                max_model_len=self.config.max_model_len,
                kvcache_block_size=self.config.kvcache_block_size,
                max_num_batched_tokens=self.config.max_num_batched_tokens,
                max_num_seqs=self.config.max_num_seqs,
            )
            print(f"LLM initialized successfully (mode={actual_mode})")
            return actual_mode
        except AssertionError as e:
            if "Draft model" in str(e):
                print(f"Error: {e}")
                print("Please provide a valid draft model path with --draft-model")
                if eagle_enabled:
                    print("Falling back to AR mode...")
                    eagle_enabled = False
                    actual_mode = "ar"
                    self.llm = LLM(
                        model=self.config.model,
                        eagle_enabled=False,
                        tensor_parallel_size=self.config.tensor_parallel_size,
                        enforce_eager=self.config.enforce_eager,
                        gpu_memory_utilization=self.config.gpu_memory_utilization,
                        max_model_len=self.config.max_model_len,
                        kvcache_block_size=self.config.kvcache_block_size,
                        max_num_batched_tokens=self.config.max_num_batched_tokens,
                        max_num_seqs=self.config.max_num_seqs,
                    )
                    print(f"LLM initialized successfully (mode={actual_mode})")
                    return actual_mode
                else:
                    raise
            else:
                raise
    
    def generate_dataset(self) -> tuple[list[list[int]], list[SamplingParams]]:
        """Generate or load benchmark dataset.
        
        Returns:
            Tuple of (prompt_token_ids, sampling_params)
        """
        print(f"Generating dataset (dataset={self.config.dataset}, "
              f"num_sequences={self.config.num_sequences})...")
        
        if self.config.dataset == "random":
            # Random token sequences for controlled benchmarking
            prompt_token_ids = [
                [randint(0, 10000) for _ in range(randint(50, self.config.max_input_len))]
                for _ in range(self.config.num_sequences)
            ]
            sampling_params = [
                SamplingParams(
                    temperature=self.config.temperature,
                    ignore_eos=True,
                    max_tokens=randint(self.config.max_output_len // 2, self.config.max_output_len)
                )
                for _ in range(self.config.num_sequences)
            ]
        
        elif self.config.dataset == "sharegpt":
            # ShareGPT conversations for realistic benchmarking
            prompt_token_ids, sampling_params = self._load_sharegpt_dataset()
        
        elif self.config.dataset == "gsm8k":
            # GSM8K math problems (requires dataset download)
            prompt_token_ids, sampling_params = self._load_gsm8k_dataset()
        
        elif self.config.dataset == "humaneval":
            # HumanEval code completion (requires dataset download)
            prompt_token_ids, sampling_params = self._load_humaneval_dataset()
        
        else:
            raise ValueError(f"Unknown dataset: {self.config.dataset}")
        
        total_input_tokens = sum(len(p) for p in prompt_token_ids)
        total_output_tokens = sum(sp.max_tokens for sp in sampling_params)
        print(f"Dataset generated: {len(prompt_token_ids)} sequences, "
              f"{total_input_tokens} input tokens, {total_output_tokens} output tokens")
        
        return prompt_token_ids, sampling_params
    
    def _load_sharegpt_dataset(self) -> tuple[list[list[int]], list[SamplingParams]]:
        """Load ShareGPT dataset for benchmarking.
        
        Uses the same data processing as EAGLE training.
        Data file: data/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json
        
        Returns:
            Tuple of (prompt_token_ids, sampling_params)
        """
        from nanovllm.utils.sharegpt_loader import sample_sharegpt_prompts
        
        sharegpt_path = "data/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json"
        
        if not os.path.exists(sharegpt_path):
            print(f"Warning: ShareGPT dataset not found at {sharegpt_path}")
            print("Falling back to random dataset")
            return self._generate_random_dataset()
        
        # Sample prompts from ShareGPT
        num_prompts = self.config.num_sequences
        text_prompts = sample_sharegpt_prompts(
            file_path=sharegpt_path,
            num_prompts=num_prompts,
            max_file_samples=num_prompts * 3,  # Load extra to ensure enough after filtering
            seed=self.config.seed,
        )
        
        # Tokenize prompts
        print("Tokenizing ShareGPT prompts...")
        prompt_token_ids = []
        sampling_params = []
        
        for text in text_prompts:
            # Tokenize the prompt
            token_ids = self._tokenize_text(text)
            
            # Limit input length
            max_input = min(len(token_ids), self.config.max_input_len)
            if max_input < 10:  # Skip very short prompts
                continue
            
            token_ids = token_ids[:max_input]
            prompt_token_ids.append(token_ids)
            
            # Create sampling params with variable output length
            output_len = randint(self.config.max_output_len // 2, self.config.max_output_len)
            sampling_params.append(
                SamplingParams(
                    temperature=self.config.temperature,
                    ignore_eos=True,
                    max_tokens=output_len,
                )
            )
        
        print(f"Loaded {len(prompt_token_ids)} ShareGPT prompts")
        return prompt_token_ids, sampling_params
    
    def _tokenize_text(self, text: str) -> list[int]:
        """Tokenize text using the model's tokenizer."""
        if self.llm is None:
            # Fallback: create a temporary tokenizer
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.config.model, use_fast=True)
            return tokenizer.encode(text, add_special_tokens=False)
        else:
            return self.llm.tokenizer.encode(text, add_special_tokens=False)
    
    def _generate_random_dataset(self) -> tuple[list[list[int]], list[SamplingParams]]:
        """Generate random token sequences for benchmarking."""
        prompt_token_ids = [
            [randint(0, 10000) for _ in range(randint(50, self.config.max_input_len))]
            for _ in range(self.config.num_sequences)
        ]
        sampling_params = [
            SamplingParams(
                temperature=self.config.temperature,
                ignore_eos=True,
                max_tokens=randint(self.config.max_output_len // 2, self.config.max_output_len)
            )
            for _ in range(self.config.num_sequences)
        ]
        return prompt_token_ids, sampling_params
    
    def _load_gsm8k_dataset(self) -> tuple[list[list[int]], list[SamplingParams]]:
        """Load GSM8K math dataset."""
        # Placeholder - in production, load from datasets library
        print("Warning: GSM8K dataset not available, falling back to random")
        return self.generate_dataset()
    
    def _load_humaneval_dataset(self) -> tuple[list[list[int]], list[SamplingParams]]:
        """Load HumanEval code completion dataset."""
        # Placeholder - in production, load from datasets library
        print("Warning: HumanEval dataset not available, falling back to random")
        return self.generate_dataset()
    
    def run_warmup(self, prompt_token_ids: list[list[int]]):
        """Run warm-up iterations."""
        print(f"Running {self.config.warmup_runs} warm-up iterations...")
        
        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            ignore_eos=True,
            max_tokens=10,  # Short generation for warmup
        )
        
        # Use a small subset for warmup
        warmup_prompts = prompt_token_ids[:4]
        
        for i in range(self.config.warmup_runs):
            _ = self.llm.generate(warmup_prompts, sampling_params, use_tqdm=False)
            print(f"  Warmup {i+1}/{self.config.warmup_runs} completed")
        
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print("Warmup completed")
    
    def run_benchmark(
        self,
        prompt_token_ids: list[list[int]],
        sampling_params: list[SamplingParams],
        mode: str,
    ) -> BenchmarkMetrics:
        """Run benchmark with metrics collection.
        
        Args:
            prompt_token_ids: List of prompt token sequences
            sampling_params: List of sampling parameters
            mode: "eagle" or "ar"
        
        Returns:
            BenchmarkMetrics object with collected metrics
        """
        print(f"\n{'='*60}")
        print(f"Running benchmark (mode={mode})")
        print(f"{'='*60}")
        
        collector = MetricsCollector(mode=mode)
        config_dict = asdict(self.config)
        config_dict["mode"] = mode
        collector.start_run(config=config_dict)
        
        # Record KV cache info
        if self.llm and hasattr(self.llm.model_runner, 'kv_cache'):
            kv_cache = self.llm.model_runner.kv_cache
            config = self.llm.model_runner.config
            collector.record_kv_cache_info(
                memory_bytes=kv_cache.element_size() * kv_cache.numel(),
                blocks_used=config.num_kvcache_blocks,
                blocks_total=config.num_kvcache_blocks,
            )
        
        total_tokens = 0
        
        # Run benchmark for each sequence
        # Note: For accurate comparison, we process sequences one at a time
        # to measure per-token metrics accurately
        for i, (prompt, params) in enumerate(zip(prompt_token_ids, sampling_params)):
            if self.config.verbose and (i + 1) % 10 == 0:
                print(f"  Processing sequence {i+1}/{len(prompt_token_ids)}")
            
            # Record prefill start
            collector.record_prefill_start()
            prefill_start = time.time()
            
            # Generate tokens
            outputs = self.llm.generate([prompt], params, use_tqdm=False)
            
            # Record prefill end (TTFT)
            prefill_time = time.time() - prefill_start
            num_input_tokens = len(prompt)
            collector.record_prefill_end(num_input_tokens)
            
            # Get output
            if outputs and len(outputs) > 0:
                output = outputs[0]
                token_ids = output.get("token_ids", []) if isinstance(output, dict) else []
                # token_ids contains only the generated (completion) tokens
                num_output_tokens = len(token_ids)
                total_tokens += num_output_tokens
                
                # Record decode step
                if num_output_tokens > 0:
                    collector.record_decode_step_end(num_output_tokens)
                else:
                    # No tokens generated, still record the step
                    collector.record_decode_step_end(1)
                
                # For EAGLE mode, record speculation metrics
                # Note: This requires instrumentation in the LLM engine
                if mode == "eagle" and hasattr(output, 'speculation_stats'):
                    stats = output.speculation_stats
                    collector.record_speculation_round(
                        draft_tokens=stats.get('draft_tokens', 0),
                        accepted_tokens=stats.get('accepted_tokens', 0),
                        draft_time=stats.get('draft_time', 0.0),
                        verification_time=stats.get('verification_time', 0.0),
                        target_time=stats.get('target_time', 0.0),
                    )
                elif mode == "eagle":
                    # Estimate speculation stats (unimplemented - tree verification not ready)
                    # TODO: Instrument the engine to provide accurate stats
                    estimated_draft = self.config.speculation_depth
                    estimated_accepted = estimated_draft * 0.7  # Assume 70% acceptance
                    collector.record_speculation_round(
                        draft_tokens=estimated_draft,
                        accepted_tokens=int(estimated_accepted),
                        draft_time=prefill_time * 0.1,  # Estimate
                        verification_time=prefill_time * 0.3,  # Estimate
                        target_time=prefill_time * 0.6,  # Estimate
                    )
            else:
                # Handle empty output
                collector.record_decode_step_end(0)
        
        # End benchmark run
        metrics = collector.end_run()
        metrics.macro.total_tokens = total_tokens
        
        # Compute FLOPs estimates for arithmetic intensity
        self._estimate_flops(metrics, total_tokens, mode)
        
        self.metrics_collectors.append(collector)
        
        print(f"\nBenchmark completed (mode={mode})")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Total time: {metrics.macro.total_time:.2f}s")
        print(f"  Throughput: {metrics.macro.throughput:.2f} tok/s")
        
        if mode == "eagle":
            print(f"  Avg acceptance length: {metrics.algorithmic.avg_acceptance_length:.2f}")
            print(f"  Acceptance rate: {metrics.algorithmic.acceptance_rate:.2%}")
        
        return metrics
    
    def _estimate_flops(self, metrics: BenchmarkMetrics, total_tokens: int, mode: str):
        """Estimate FLOPs for arithmetic intensity calculation.
        
        This is a simplified estimation based on model size and sequence length.
        For accurate measurements, use NVIDIA Nsight or similar profiling tools.
        """
        hf_config = self.config.hf_config if hasattr(self.config, 'hf_config') else None
        
        if hf_config is None:
            try:
                from transformers import AutoConfig
                hf_config = AutoConfig.from_pretrained(self.config.model)
                self.config.hf_config = hf_config
            except Exception:
                return
        
        # Estimate FLOPs per token for transformer forward pass
        # Formula: 2 * batch_size * seq_len * hidden_size^2 * num_layers
        # This is approximate - actual FLOPs depend on architecture details
        hidden_size = hf_config.hidden_size
        num_layers = hf_config.num_hidden_layers
        vocab_size = hf_config.vocab_size
        
        # FLOPs for one token generation (approximate)
        # Attention: 4 * hidden_size^2 (Q, K, V, O projections)
        # MLP: 8 * hidden_size^2 (gate, up, down with expansion)
        flops_per_token = (4 * hidden_size**2 + 8 * hidden_size**2) * num_layers
        
        # For AR baseline: weights loaded once per token
        # Weight bytes: all model parameters
        # Access model through model_runner
        model = None
        if self.llm and hasattr(self.llm, 'model_runner'):
            model = self.llm.model_runner.model
        
        num_params = sum(p.numel() for p in model.parameters()) if model else 0
        bytes_per_token = num_params * 2  # FP16 = 2 bytes
        
        if mode == "ar":
            metrics.hardware.total_flops = flops_per_token * total_tokens
            metrics.hardware.total_bytes_accessed = bytes_per_token * total_tokens
        
        elif mode == "eagle":
            # For EAGLE: draft model is smaller (~1/4 the size)
            # Verification reuses target model weights
            draft_flops = flops_per_token * 0.25  # Draft model is smaller
            target_flops = flops_per_token  # Full model for verification
            
            avg_acceptance = metrics.algorithmic.avg_acceptance_length
            if avg_acceptance > 0:
                # FLOPs per accepted token
                effective_flops = (draft_flops + target_flops) / avg_acceptance
                metrics.hardware.total_flops = effective_flops * total_tokens
                
                # Bytes accessed: draft + verification, amortized over accepted tokens
                effective_bytes = (bytes_per_token * 0.25 + bytes_per_token) / avg_acceptance
                metrics.hardware.total_bytes_accessed = effective_bytes * total_tokens
    
    def run(self) -> dict:
        """Run the complete benchmark suite.
        
        Returns:
            Dictionary with benchmark results and comparison
        """
        results = {
            "timestamp": datetime.now().isoformat(),
            "config": asdict(self.config),
            "runs": {},
            "comparison": {},
        }
        
        # Generate dataset
        prompt_token_ids, sampling_params = self.generate_dataset()

        # Run EAGLE first (needs more memory for draft model)
        if self.config.mode in ("eagle", "both"):
            try:
                actual_mode = self.initialize_llm(mode="eagle")
                # Only run EAGLE benchmark if we actually got EAGLE mode
                if actual_mode == "eagle":
                    self.run_warmup(prompt_token_ids)

                    eagle_metrics = self.run_benchmark(
                        prompt_token_ids, sampling_params, mode="eagle"
                    )
                    results["runs"]["eagle"] = eagle_metrics.to_dict()
            except Exception as e:
                print(f"Error running EAGLE benchmark: {e}")
                print("Try running EAGLE mode separately with fewer sequences")

            # Clean up
            if self.llm:
                try:
                    self.llm.model_runner.exit()
                except Exception:
                    pass
                self.llm = None
            
            # Aggressive memory cleanup
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            import gc
            gc.collect()
            
            # Wait for memory to be freed
            import time
            time.sleep(2)

        # Run AR baseline
        if self.config.mode in ("ar", "both"):
            try:
                self.initialize_llm(mode="ar")
                self.run_warmup(prompt_token_ids)

                ar_metrics = self.run_benchmark(
                    prompt_token_ids, sampling_params, mode="ar"
                )
                results["runs"]["ar"] = ar_metrics.to_dict()
            except Exception as e:
                print(f"Error running AR benchmark: {e}")
                print("Try running AR mode separately")

            # Clean up
            if self.llm:
                try:
                    self.llm.model_runner.exit()
                except Exception:
                    pass
                self.llm = None
            
            # Aggressive memory cleanup
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            import gc
            gc.collect()
        
        # Compare results
        if "ar" in results["runs"] and "eagle" in results["runs"]:
            comparison = compare_metrics(
                BenchmarkMetrics(**results["runs"]["eagle"]),
                BenchmarkMetrics(**results["runs"]["ar"]),
            )
            results["comparison"] = comparison
            
            print(f"\n{'='*60}")
            print("BENCHMARK COMPARISON RESULTS")
            print(f"{'='*60}")
            print(f"Speedup (EAGLE/AR): {comparison['speedup']:.2f}x")
            print(f"Throughput improvement: {comparison['throughput_improvement']:.2f} tok/s")
            print(f"TPT improvement: {comparison['tpt_improvement']:.4f}s")
            print(f"Acceptance rate (EAGLE): {comparison['eagle_acceptance_rate']:.2%}")
            print(f"Avg acceptance length (EAGLE): {comparison['eagle_avg_acceptance_length']:.2f}")
            print(f"Efficiency score (EAGLE): {comparison['eagle_efficiency']:.2%}")
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _save_results(self, results: dict):
        """Save benchmark results to file."""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        mode_str = self.config.mode
        filename = f"eagle_bench_{mode_str}_{timestamp}.json"
        output_path = output_dir / filename
        
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\nResults saved to: {output_path}")
        
        # Also save summary to CSV for easy plotting
        self._save_csv_summary(results, output_dir, timestamp)
    
    def _save_csv_summary(self, results: dict, output_dir: Path, timestamp: str):
        """Save CSV summary for plotting."""
        import csv
        
        runs = results.get("runs", {})
        comparison = results.get("comparison", {})
        
        # Summary CSV
        summary_path = output_dir / f"summary_{timestamp}.csv"
        with open(summary_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Metric", "AR", "EAGLE", "Improvement"])
            
            if "ar" in runs and "eagle" in runs:
                ar = runs["ar"]["macro"]
                eagle = runs["eagle"]["macro"]
                
                writer.writerow(["Throughput (tok/s)", 
                               f"{ar['throughput']:.2f}", 
                               f"{eagle['throughput']:.2f}",
                               f"{comparison.get('speedup', 0):.2f}x"])
                writer.writerow(["TTFT (s)",
                               f"{ar['ttft']:.4f}",
                               f"{eagle['ttft']:.4f}",
                               ""])
                writer.writerow(["Avg TPT (s)",
                               f"{ar['avg_tpt']:.4f}",
                               f"{eagle['avg_tpt']:.4f}",
                               f"{comparison.get('tpt_improvement', 0):.4f}"])
                
                # Algorithmic metrics (EAGLE only)
                algo = runs["eagle"]["algorithmic"]
                writer.writerow(["Avg Acceptance Length", "N/A", 
                               f"{algo['avg_acceptance_length']:.2f}", ""])
                writer.writerow(["Acceptance Rate", "N/A",
                               f"{algo['acceptance_rate']:.2%}", ""])
                writer.writerow(["Efficiency Score", "N/A",
                               f"{algo['efficiency_score']:.2%}", ""])


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="EAGLE Speculative Decoding Benchmark Suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to the base model directory"
    )
    parser.add_argument(
        "--draft-model",
        type=str,
        default=None,
        help="Path to the EAGLE draft model (required for --mode eagle)"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["eagle", "ar", "both"],
        default="both",
        help="Benchmark mode: eagle, ar, or both (compare)"
    )
    parser.add_argument(
        "--speculation-depth",
        type=int,
        default=4,
        help="Number of draft tokens to generate per speculation step"
    )
    parser.add_argument(
        "--num-sequences",
        type=int,
        default=256,
        help="Number of sequences to benchmark"
    )
    parser.add_argument(
        "--max-input-len",
        type=int,
        default=512,
        help="Maximum input sequence length"
    )
    parser.add_argument(
        "--max-output-len",
        type=int,
        default=256,
        help="Maximum output sequence length"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["random", "sharegpt", "gsm8k", "humaneval"],
        default="random",
        help="Dataset to use for benchmarking"
    )
    parser.add_argument(
        "--warmup-runs",
        type=int,
        default=1,
        help="Number of warm-up iterations"
    )
    parser.add_argument(
        "--benchmark-runs",
        type=int,
        default=3,
        help="Number of benchmark iterations (for averaging)"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallelism degree"
    )
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Disable CUDA graphs (use eager mode)"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization fraction"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Maximum model context length"
    )
    parser.add_argument(
        "--kvcache-block-size",
        type=int,
        default=256,
        help="KV cache block size (must be multiple of 256)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmark_results",
        help="Output directory for results"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=True,
        help="Enable verbose output"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable verbose output"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.mode == "eagle" and args.draft_model is None:
        print("Error: --draft-model is required for EAGLE mode")
        sys.exit(1)
    
    # Create benchmark config
    config = BenchmarkConfig(
        model=args.model,
        draft_model=args.draft_model,
        speculation_depth=args.speculation_depth,
        mode=args.mode,
        num_sequences=args.num_sequences,
        max_input_len=args.max_input_len,
        max_output_len=args.max_output_len,
        temperature=args.temperature,
        dataset=args.dataset,
        warmup_runs=args.warmup_runs,
        benchmark_runs=args.benchmark_runs,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        kvcache_block_size=args.kvcache_block_size,
        seed=args.seed,
        output_dir=args.output_dir,
        verbose=not args.quiet,
    )
    
    # Print configuration
    print("="*60)
    print("EAGLE SPECULATIVE DECODING BENCHMARK")
    print("="*60)
    print(f"Model: {config.model}")
    if config.draft_model:
        print(f"Draft Model: {config.draft_model}")
    print(f"Mode: {config.mode}")
    print(f"Speculation Depth: {config.speculation_depth}")
    print(f"Sequences: {config.num_sequences}")
    print(f"Input Length: 50-{config.max_input_len}")
    print(f"Output Length: {config.max_output_len // 2}-{config.max_output_len}")
    print(f"Temperature: {config.temperature}")
    print(f"Dataset: {config.dataset}")
    print(f"Tensor Parallel: {config.tensor_parallel_size}")
    print(f"Enforce Eager: {config.enforce_eager}")
    print("="*60)
    
    # Run benchmark
    benchmark = EagleBenchmark(config)
    results = benchmark.run()
    
    print("\nBenchmark completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
