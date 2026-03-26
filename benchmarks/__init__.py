"""EAGLE Speculative Decoding Benchmark Suite.

This package provides comprehensive benchmarking tools for evaluating
EAGLE speculative decoding against naive autoregressive baseline.

Modules:
- benchmarks.eagle_bench: Main benchmark runner
- benchmarks.analysis: Results analysis and visualization
- benchmarks.metrics: Metrics collection infrastructure
"""

from benchmarks.eagle_bench import EagleBenchmark, BenchmarkConfig, main as run_benchmark
from benchmarks.analysis import ResultsAggregator, BenchmarkVisualizer, AggregatedResults

__all__ = [
    "EagleBenchmark",
    "BenchmarkConfig",
    "run_benchmark",
    "ResultsAggregator",
    "BenchmarkVisualizer",
    "AggregatedResults",
]
