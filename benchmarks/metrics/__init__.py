"""Metrics collection module for EAGLE benchmark."""

from benchmarks.metrics.collector import (
    MetricsCollector,
    BenchmarkMetrics,
    MacroMetrics,
    HardwareMetrics,
    AlgorithmicMetrics,
    compute_speedup,
    compare_metrics,
)

__all__ = [
    "MetricsCollector",
    "BenchmarkMetrics",
    "MacroMetrics",
    "HardwareMetrics",
    "AlgorithmicMetrics",
    "compute_speedup",
    "compare_metrics",
]
