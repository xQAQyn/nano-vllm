# Nano-vLLM Project Context

## Project Overview

**Nano-vLLM** is a lightweight, from-scratch implementation of vLLM's core inference engine. It provides fast offline LLM inference with a clean, readable codebase (~1,200 lines of Python). The project is designed for educational purposes and optimization experimentation.

### Key Features
- 🚀 **Fast offline inference** - Comparable or better performance than vLLM
- 📖 **Readable codebase** - Clean implementation in ~1,200 lines
- ⚡ **Optimization Suite** - Prefix caching, Tensor Parallelism, Torch compilation, CUDA graphs

### Architecture
The project mirrors vLLM's architecture with these core components:

```
nanovllm/
├── llm.py              # Main LLM interface (wraps LLMEngine)
├── config.py           # Configuration dataclass
├── sampling_params.py  # Sampling parameters (temperature, max_tokens)
├── engine/
│   ├── llm_engine.py   # Core engine managing generation loop
│   ├── model_runner.py # Model execution with TP support
│   ├── scheduler.py    # Prefill/decode scheduling
│   ├── block_manager.py# Paged KV-cache management with prefix caching
│   └── sequence.py     # Sequence state tracking
├── models/
│   └── qwen3.py        # Qwen3 model implementation
├── layers/
│   ├── attention.py    # FlashAttention + paged KV-cache
│   ├── linear.py       # Tensor-parallel linear layers
│   ├── rotary_embedding.py
│   ├── sampler.py      # Temperature-scaled sampling
│   └── ...
└── utils/
    ├── context.py      # Global context for forward passes
    └── loader.py       # Safetensors model loading
```

## Building and Running

### Prerequisites
- **Python:** 3.12+ (see `.python-version`)
- **Package Manager:** `uv` (project uses `uv.lock`)
- **GPU:** NVIDIA GPU with CUDA support (NCCL for tensor parallelism)

### Installation
```bash
# Using pip
pip install git+https://github.com/GeeeekExplorer/nano-vllm.git

# Local development (using uv)
uv sync
```

### Model Download
```bash
huggingface-cli download --resume-download Qwen/Qwen3-0.6B \
  --local-dir models/Qwen3-0.6B/
```

### Quick Start
```python
from nanovllm import LLM, SamplingParams

llm = LLM("models/Qwen3-0.6B/", enforce_eager=True, tensor_parallel_size=1)
sampling_params = SamplingParams(temperature=0.6, max_tokens=256)
prompts = ["Hello, Nano-vLLM."]
outputs = llm.generate(prompts, sampling_params)
print(outputs[0]["text"])
```

### Example Scripts
| File | Purpose |
|------|---------|
| `example.py` | Basic usage with chat templates |
| `bench.py` | Performance benchmarking |
| `main.py` | Entry point (placeholder) |
| `download.py` | Model download helper |

### Running Examples
```bash
# Run example
python example.py

# Run benchmark
python bench.py
```

## Development Conventions

### Code Style
- **Type hints:** Used throughout (e.g., `torch.Tensor`, `list[int]`)
- **Dataclasses:** Configuration via `@dataclass` (Config, SamplingParams)
- **Naming:** CamelCase for classes, snake_case for functions/variables
- **Modules:** Organized by functionality (engine, layers, models, utils)

### Key Design Patterns
1. **Tensor Parallelism:** Multi-process via `torch.multiprocessing` with NCCL backend
2. **Paged KV-Cache:** Block-based memory management with prefix caching (xxhash)
3. **CUDA Graphs:** Captured for decode phase (batch sizes: 1,2,4,8,16...)
4. **Context Management:** Global `Context` dataclass for forward pass state

### Testing Practices
- No formal test suite in the codebase
- Validation via example scripts and benchmarks
- Assertions in `__post_init__` methods for config validation

### Dependencies
The project uses minimal explicit dependencies in `pyproject.toml`. Key runtime dependencies:
- `torch` - Core tensor operations
- `transformers` - Tokenizer and config loading
- `flash_attn` - FlashAttention for efficient attention
- `triton` - Custom CUDA kernels
- `safetensors` - Model weight loading
- `xxhash` - Fast hashing for prefix cache
- `tqdm` - Progress bars

### Performance Notes
- **Benchmark config:** RTX 4070 Laptop (8GB), Qwen3-0.6B, 256 sequences
- **Results:** Nano-vLLM achieves ~1434 tok/s vs vLLM's ~1362 tok/s
- **Optimizations:** Prefix caching, tensor parallelism, CUDA graphs, torch.compile

## Configuration Options

```python
LLM(
    model="path/to/model",
    max_num_batched_tokens=16384,
    max_num_seqs=512,
    max_model_len=4096,
    gpu_memory_utilization=0.9,
    tensor_parallel_size=1,      # 1-8
    enforce_eager=False,          # Enable CUDA graphs
    kvcache_block_size=256,       # Must be multiple of 256
)
```

## Sampling Parameters

```python
SamplingParams(
    temperature=1.0,    # Must be > 1e-10 (no greedy sampling)
    max_tokens=64,
    ignore_eos=False,
)
```
