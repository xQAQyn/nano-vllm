# EAGLE Speculative Decoding Benchmark Report

## Executive Summary

This document describes the benchmarking methodology, metrics, and usage guide for evaluating **EAGLE (Speculative Decoding)** against a **Naive Autoregressive (AR)** baseline in the nano-vLLM inference engine.

The benchmark suite is designed to:
1. **Quantify end-to-end speed gains** from speculative decoding
2. **Analyze the "Memory Wall" impact** - demonstrating how EAGLE mitigates memory bandwidth bottlenecks
3. **Diagnose speculative efficiency** - measuring draft head performance and tree-structured verification

---

## Table of Contents

- [Benchmark Methodology](#benchmark-methodology)
- [Hardware Environment](#hardware-environment)
- [Metrics Definitions](#metrics-definitions)
- [Usage Guide](#usage-guide)
- [Interpreting Results](#interpreting-results)
- [Troubleshooting](#troubleshooting)

---

## Benchmark Methodology

### Overview

The benchmark follows a rigorous methodology to ensure fair comparison between EAGLE and AR modes:

```
┌─────────────────────────────────────────────────────────────┐
│                    Benchmark Workflow                        │
├─────────────────────────────────────────────────────────────┤
│  1. Initialize LLM (AR mode)                                 │
│  2. Generate/Load Dataset                                    │
│  3. Warm-up (1-3 iterations)                                 │
│  4. Run AR Benchmark → Collect Metrics                       │
│  5. Cleanup & Reset GPU Memory                               │
│  6. Initialize LLM (EAGLE mode)                              │
│  7. Warm-up (1-3 iterations)                                 │
│  8. Run EAGLE Benchmark → Collect Metrics                    │
│  9. Compute Comparison & Speedup                             │
│ 10. Save Results (JSON + CSV)                                │
└─────────────────────────────────────────────────────────────┘
```

### Dataset Configuration

The benchmark supports multiple datasets for comprehensive evaluation:

| Dataset | Description | Use Case |
|---------|-------------|----------|
| `random` | Random token sequences (controlled) | Performance isolation |
| `sharegpt` | ShareGPT conversations (realistic) | Real-world chat workload |
| `gsm8k` | Grade school math problems | Reasoning workload |
| `humaneval` | Code completion tasks | Code generation workload |

**Default Configuration (Random Dataset):**
- **Number of sequences:** 256
- **Input length:** 50-512 tokens (uniform random)
- **Output length:** 128-256 tokens (uniform random)
- **Temperature:** 1.0

**ShareGPT Dataset:**
- **Data file:** `data/ShareGPT_V3_unfiltered_cleaned_split_no_imsorry.json`
- **Input length:** 50-max_input_len tokens (from actual conversations)
- **Output length:** max_output_len/2 to max_output_len tokens
- Uses the same data processing as EAGLE training

### Warm-up Protocol

To eliminate cold-start effects:
1. **1-3 warm-up iterations** with short sequences (10 tokens)
2. **GPU memory reset** between AR and EAGLE runs
3. **CUDA cache clearing** before timing begins

### Iteration Strategy

For statistical significance:
- **Default:** 3 benchmark runs per mode
- **Reported metrics:** Mean ± standard deviation
- **Outlier handling:** Median reported if variance > 10%

---

## Hardware Environment

### System Information Template

Fill in your hardware details below:

```markdown
### GPU Configuration
- **GPU Model:** [e.g., NVIDIA RTX 4070 Laptop]
- **GPU Memory:** [e.g., 8 GB GDDR6X]
- **Memory Bandwidth:** [e.g., 256 GB/s peak]
- **CUDA Cores:** [e.g., 4608]
- **Tensor Cores:** [e.g., 144 (4th gen)]

### Software Stack
- **CUDA Version:** [e.g., 12.1]
- **PyTorch Version:** [e.g., 2.4.0]
- **Driver Version:** [e.g., 535.104.05]
- **Python Version:** [e.g., 3.12]

### Model Configuration
- **Base Model:** [e.g., Qwen3-0.6B]
- **Draft Model:** [e.g., EAGLE-Qwen3-0.6B]
- **Precision:** [e.g., FP16]
- **KV Cache Block Size:** [e.g., 256]
- **GPU Memory Utilization:** [e.g., 90%]
```

### Automatic Hardware Detection

The benchmark script automatically detects and logs:
- GPU name and memory size
- CUDA version
- PyTorch version
- KV cache allocation

Example output:
```
Hardware Info:
  GPU: NVIDIA GeForce RTX 4070 Laptop GPU
  Memory: 8.0 GB
  CUDA: 12.1
  PyTorch: 2.4.0
```

---

## Metrics Definitions

### 1. Macro-Performance Metrics (The "What")

These metrics quantify the observable performance improvements.

#### Throughput (Tokens Per Second)

**Definition:**
```
TPS = Total Generated Tokens / Total Time
```

**Significance:**
- Primary measure of inference performance
- Higher is better
- Directly impacts user experience

**Reported As:** `tok/s`

#### Speedup Ratio

**Definition:**
```
Speedup = TPS_EAGLE / TPS_AR
```

**Significance:**
- Measures relative improvement over baseline
- Speedup > 1.0 indicates EAGLE is faster
- Theoretical maximum depends on acceptance rate

**Interpretation:**
- `1.5x` = 50% faster than AR
- `2.0x` = 2x faster (100% improvement)
- `1.0x` = no improvement

#### TTFT (Time to First Token)

**Definition:**
```
TTFT = Time(prompt input → first token generated)
```

**Significance:**
- Measures prefill stage latency
- Critical for interactive applications
- Lower is better

**Reported As:** `seconds` or `milliseconds`

#### TPT (Time Per Token)

**Definition:**
```
TPT = Σ(inter-token latency) / Number of tokens
```

**Significance:**
- Average decode latency per token
- Lower is better
- Inverse of throughput (approximately)

**Reported As:** `seconds/token` or `milliseconds/token`

---

### 2. Hardware & Memory Metrics (The "Why" - Memory Wall Analysis)

These metrics explain *why* EAGLE achieves speedup by analyzing hardware utilization.

#### Arithmetic Intensity (FLOPs per Byte)

**Definition:**
```
Arithmetic Intensity = Total FLOPs / Total Bytes Accessed
```

**Significance:**
- Measures compute-to-memory ratio
- **Higher values indicate compute-bound behavior**
- **Lower values indicate memory-bound behavior**

**Memory Wall Analysis:**
- AR baseline: Low arithmetic intensity (~1-5 F/B) → memory-bound
- EAGLE: Higher arithmetic intensity (~3-10 F/B) → more compute-bound

**Why it matters:**
Modern GPUs have much higher compute capacity than memory bandwidth. By increasing arithmetic intensity, EAGLE better utilizes GPU compute units and reduces memory bottleneck impact.

#### Memory Bandwidth Utilization (MBU)

**Definition:**
```
MBU = Actual Memory Bandwidth Used / Peak Memory Bandwidth
```

**Significance:**
- Measures how effectively memory bandwidth is utilized
- Reported in `GB/s`
- Helps identify memory bottlenecks

**Typical Values:**
- AR baseline: 60-80% of peak bandwidth
- EAGLE: 70-90% of peak bandwidth (more efficient access patterns)

#### Weight Reuse Factor

**Definition:**
```
Weight Reuse Factor = Weight Loads from VRAM / Successfully Generated Tokens
```

**Significance:**
- **Lower is better**
- Measures how many times weights are loaded per token
- AR baseline: ~1.0 (weights loaded once per token)
- EAGLE: < 1.0 (weights loaded once for multiple accepted tokens)

**Example:**
- If EAGLE accepts 3 tokens per speculation round:
  - Weight Reuse Factor ≈ 1/3 = 0.33
  - 3x better weight reuse than AR

#### KV Cache Overhead

**Definition:**
```
KV Cache Overhead = ΔVRAM_EAGLE - ΔVRAM_AR
```

**Significance:**
- Measures additional VRAM consumed by tree-structured attention
- EAGLE requires storing KV for draft tokens during verification
- Typically small overhead (< 10% of total KV cache)

**Reported As:** `bytes` or `MB`

---

### 3. Algorithmic Metrics (Speculative Diagnostics)

These metrics diagnose the effectiveness of the speculative decoding algorithm.

#### Average Acceptance Length (τ)

**Definition:**
```
τ = Σ(Accepted Tokens per Round) / Number of Speculation Rounds
```

**Significance:**
- **Key metric for EAGLE efficiency**
- Average number of tokens accepted per speculation step
- Higher values → more speedup potential

**Theoretical Speedup:**
```
Max Speedup ≈ τ + 1
```

**Example:**
- τ = 2.5 → Theoretical max speedup ≈ 3.5x
- τ = 3.0 → Theoretical max speedup ≈ 4.0x

**Typical Values:**
- Good draft model: τ = 2.0-4.0
- Excellent draft model: τ = 4.0-6.0

#### Acceptance Rate Distribution

**Definition:**
```
P(n tokens accepted) for n = 0, 1, 2, ..., K
```

**Significance:**
- Histogram showing acceptance frequency
- Reveals draft model quality distribution
- Helps tune speculation depth

**Example Distribution:**
```
n=0 (all rejected):  5%
n=1:                15%
n=2:                30%
n=3:                35%
n=4 (all accepted): 15%
```

**Interpretation:**
- High probability at n=K → increase speculation depth
- High probability at n=0 → draft model needs improvement

#### Draft Head Overhead

**Definition:**
```
Draft Overhead = Draft Head Time / (Draft Head Time + Target Model Time)
```

**Significance:**
- Percentage of time spent in draft model
- Lower is better (draft model should be fast)
- Typical: 10-30%

**Optimization Target:**
- Draft model should be < 25% of total time
- If > 40%, consider smaller draft model

#### Efficiency Score

**Definition:**
```
Efficiency = Accepted Tokens / Total Drafted Tokens
```

**Significance:**
- Overall acceptance rate
- Measures draft model accuracy
- Higher is better (max = 1.0)

**Relationship to Acceptance Rate:**
- Efficiency Score = Average Acceptance Rate across all rounds

**Typical Values:**
- Good: 0.6-0.8
- Excellent: 0.8-0.9

---

## Usage Guide

### Prerequisites

1. **Install dependencies:**
   ```bash
   uv sync
   pip install pynvml  # For GPU metrics
   ```

2. **Download models:**
   ```bash
   # Base model
   huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
     --local-dir models/Qwen3-0.6B/
   
   # EAGLE draft model (if available)
   huggingface-cli download --resume-download <EAGLE-draft-model> \
     --local-dir models/eagle_draft/
   ```

### Basic Usage

**Note:** Due to PyTorch distributed memory management, running both modes sequentially (`--mode both`) may cause OOM errors on GPUs with limited memory. For best results, run each mode separately and compare the JSON results.

#### Run EAGLE Benchmark

```bash
python benchmarks/eagle_bench.py \
  --model ./models/Qwen3-0.6B/ \
  --draft-model ./models/eagle_draft/ \
  --mode eagle \
  --speculation-depth 4 \
  --num-sequences 256
```

#### Run AR Baseline Benchmark

```bash
python benchmarks/eagle_bench.py \
  --model ./models/Qwen3-0.6B/ \
  --mode ar \
  --num-sequences 256
```

#### Run Comparison (Both Modes)

**Recommended:** Run modes separately for memory efficiency:
```bash
# Run EAGLE
python benchmarks/eagle_bench.py \
  --model ./models/Qwen3-0.6B/ \
  --draft-model ./models/eagle_draft/ \
  --mode eagle \
  --speculation-depth 4 \
  --output-dir results/eagle

# Run AR
python benchmarks/eagle_bench.py \
  --model ./models/Qwen3-0.6B/ \
  --mode ar \
  --output-dir results/ar

# Compare results using analysis tool
python benchmarks/analysis.py \
  --results-dir results \
  --report --plots
```

**Alternative:** Run both in single command (may require more GPU memory):
```bash
python benchmarks/eagle_bench.py \
  --model ./models/Qwen3-0.6B/ \
  --draft-model ./models/eagle_draft/ \
  --mode both \
  --speculation-depth 4 \
  --num-sequences 256 \
  --gpu-memory-utilization 0.7
```

### Advanced Configuration

#### CLI Arguments Reference

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | (required) | Path to base model directory |
| `--draft-model` | str | None | Path to EAGLE draft model |
| `--mode` | str | both | Benchmark mode: `eagle`, `ar`, or `both` |
| `--speculation-depth` | int | 4 | Number of draft tokens per step |
| `--num-sequences` | int | 256 | Number of sequences to benchmark |
| `--max-input-len` | int | 512 | Maximum input sequence length |
| `--max-output-len` | int | 256 | Maximum output sequence length |
| `--temperature` | float | 1.0 | Sampling temperature |
| `--dataset` | str | random | Dataset: `random`, `sharegpt`, `gsm8k`, `humaneval` |
| `--warmup-runs` | int | 1 | Number of warm-up iterations |
| `--benchmark-runs` | int | 3 | Number of benchmark runs |
| `--tensor-parallel-size` | int | 1 | Tensor parallelism degree |
| `--enforce-eager` | flag | False | Disable CUDA graphs |
| `--gpu-memory-utilization` | float | 0.9 | GPU memory utilization |
| `--max-model-len` | int | 4096 | Maximum model context length |
| `--kvcache-block-size` | int | 256 | KV cache block size |
| `--seed` | int | 42 | Random seed |
| `--output-dir` | str | ./benchmark_results | Results output directory |
| `--verbose` | flag | True | Enable verbose output |
| `--quiet` | flag | False | Disable verbose output |

#### Example: Tuning Speculation Depth

```bash
# Test different speculation depths
for depth in 2 4 6 8; do
  python benchmarks/eagle_bench.py \
    --model ./models/Qwen3-0.6B/ \
    --draft-model ./models/eagle_draft/ \
    --mode eagle \
    --speculation-depth $depth \
    --output-dir results/depth_$depth
done
```

#### Example: Memory-Constrained Benchmark

```bash
# Run with reduced memory usage
python benchmarks/eagle_bench.py \
  --model ./models/Qwen3-0.6B/ \
  --draft-model ./models/eagle_draft/ \
  --mode both \
  --gpu-memory-utilization 0.7 \
  --max-model-len 2048 \
  --num-sequences 128
```

### Output Files

The benchmark generates the following output files:

```
benchmark_results/
├── eagle_bench_both_20260326_143052.json   # Full results (JSON)
└── summary_20260326_143052.csv              # Summary (CSV for plotting)
```

#### JSON Output Structure

```json
{
  "timestamp": "2026-03-26T14:30:52.123456",
  "config": {
    "model": "./models/Qwen3-0.6B/",
    "draft_model": "./models/eagle_draft/",
    "mode": "both",
    "speculation_depth": 4,
    ...
  },
  "runs": {
    "ar": {
      "mode": "ar",
      "macro": {
        "total_tokens": 65536,
        "total_time": 48.23,
        "throughput": 1358.7,
        "ttft": 0.0234,
        "avg_tpt": 0.000736,
        ...
      },
      "hardware": {
        "arithmetic_intensity": 2.34,
        "avg_memory_bandwidth": 412.5,
        "weight_reuse_factor": 1.0,
        ...
      },
      "algorithmic": { ... }
    },
    "eagle": {
      "mode": "eagle",
      "macro": {
        "throughput": 2145.3,
        ...
      },
      "hardware": {
        "arithmetic_intensity": 6.78,
        "weight_reuse_factor": 0.34,
        ...
      },
      "algorithmic": {
        "avg_acceptance_length": 2.85,
        "acceptance_rate": 0.71,
        "efficiency_score": 0.71,
        "acceptance_distribution": {
          "0": 0.05,
          "1": 0.15,
          "2": 0.30,
          "3": 0.35,
          "4": 0.15
        }
      }
    }
  },
  "comparison": {
    "speedup": 1.58,
    "throughput_improvement": 786.6,
    "tpt_improvement": 0.000234,
    ...
  }
}
```

---

## Interpreting Results

### Expected Performance

**Baseline (AR):**
- Throughput: ~1300-1400 tok/s (Qwen3-0.6B on RTX 4070 Laptop)
- Arithmetic Intensity: ~2-3 F/B (memory-bound)
- Weight Reuse: 1.0

**EAGLE (Speculation Depth = 4):**
- Throughput: ~1800-2200 tok/s (depending on draft model quality)
- Speedup: 1.4x - 1.8x
- Arithmetic Intensity: ~5-8 F/B (more compute-bound)
- Weight Reuse: 0.3-0.5
- Average Acceptance Length: 2.5-3.5

### Performance Analysis Checklist

#### If Speedup < 1.2x:
- [ ] Check acceptance rate (should be > 0.6)
- [ ] Verify draft model is properly trained
- [ ] Try reducing speculation depth
- [ ] Check if workload is compute-bound (not memory-bound)

#### If Acceptance Rate < 0.5:
- [ ] Draft model may be undertrained
- [ ] Draft model architecture mismatch
- [ ] Temperature too high (try 0.7-0.9)
- [ ] Dataset distribution mismatch

#### If Acceptance Rate > 0.9:
- [ ] Increase speculation depth
- [ ] Draft model is very good - can benefit from deeper speculation
- [ ] Consider larger draft model for even better performance

#### If Draft Overhead > 0.4:
- [ ] Draft model is too large
- [ ] Consider using smaller draft model
- [ ] Optimize draft model forward pass

### Memory Wall Analysis

**Key Insight:** EAGLE improves performance by increasing arithmetic intensity, shifting from memory-bound to more compute-bound behavior.

**Evidence to Look For:**
1. **Higher Arithmetic Intensity** in EAGLE vs AR
2. **Lower Weight Reuse Factor** in EAGLE
3. **Correlation** between acceptance length and speedup

**Example Analysis:**
```
AR Baseline:
  - Arithmetic Intensity: 2.3 F/B
  - Weight Reuse: 1.0
  - Memory Bandwidth: 420 GB/s (80% utilization)
  → Memory-bound (limited by weight loading)

EAGLE:
  - Arithmetic Intensity: 6.8 F/B (3x improvement)
  - Weight Reuse: 0.33 (3x better)
  - Memory Bandwidth: 480 GB/s (92% utilization)
  → More compute-bound (better weight reuse)

Conclusion: EAGLE achieves 1.6x speedup by reducing memory bottleneck
```

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM) Error

**Symptoms:**
```
RuntimeError: CUDA out of memory
```

**Solutions:**
- Reduce `--gpu-memory-utilization` (e.g., 0.7)
- Reduce `--max-model-len`
- Reduce `--num-sequences`
- Reduce `--kvcache-block-size`

#### 2. Draft Model Not Found

**Symptoms:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'models/eagle_draft/'
```

**Solutions:**
- Verify draft model path with `--draft-model`
- Download draft model weights
- Use `"new"` for untrained draft model (training required first)

#### 3. Low Acceptance Rate

**Symptoms:**
```
Acceptance rate: 0.35 (expected > 0.6)
```

**Solutions:**
- Ensure draft model is trained on same distribution
- Reduce temperature (try 0.7-0.9)
- Reduce speculation depth
- Check draft model architecture compatibility

#### 4. Speedup < 1.0 (EAGLE slower than AR)

**Symptoms:**
```
Speedup: 0.85x (EAGLE is slower!)
```

**Solutions:**
- Check acceptance rate (should be > 0.5)
- Reduce speculation depth
- Verify draft model overhead is not too high
- Ensure CUDA graphs are enabled (`--enforce-eager` not set)

#### 5. PyNVML Not Available

**Symptoms:**
```
Warning: PyNVML not available, using fallback metrics
```

**Solutions:**
- Install pynvml: `pip install pynvml`
- Hardware metrics will use PyTorch fallback (less accurate)

---

## Appendix: Theoretical Analysis

### Speedup Model

The theoretical speedup of EAGLE can be modeled as:

```
Speedup = (τ + 1) / (1 + α * (τ + 1))
```

Where:
- `τ` = Average acceptance length
- `α` = Draft model overhead ratio (draft time / target time)

**Example:**
- τ = 3.0, α = 0.25
- Speedup = (3 + 1) / (1 + 0.25 * 4) = 4 / 2 = 2.0x

### Memory Bound vs Compute Bound

**Roofline Model:**
```
Performance = min(Compute Peak, Memory Bandwidth × Arithmetic Intensity)
```

**For AR (Memory-Bound):**
- Low arithmetic intensity (~2 F/B)
- Performance limited by memory bandwidth
- Typical: 400-500 GB/s utilization

**For EAGLE (More Compute-Bound):**
- Higher arithmetic intensity (~6 F/B)
- Performance limited by compute capacity
- Better weight reuse amortizes memory cost

---

## References

1. **EAGLE Paper:** "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty" (2024)
2. **vLLM Paper:** "vLLM: Easy, Fast, and Cheap LLM Serving with PagedAttention" (2023)
3. **Roofline Model:** Williams et al., "Roofline: An Insightful Visual Performance Model" (2009)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-03-26 | Initial release |

---

**Note:** Tree-structured verification is marked as **unimplemented** in the current benchmark script. The metrics collection infrastructure is in place, but accurate tree attention metrics will be populated once the feature is implemented.
