"""Metrics Collector for EAGLE Speculative Decoding Benchmark.

This module provides comprehensive metrics collection for benchmarking
EAGLE speculative decoding against naive autoregressive (AR) baseline.

Metrics Categories:
1. Macro-Performance: Throughput, Speedup, TTFT, TPT
2. Hardware & Memory: Arithmetic Intensity, MBU, Weight Reuse, KV Cache Overhead
3. Algorithmic: Acceptance Length, Acceptance Rate, Draft Overhead, Efficiency
"""

import time
import torch
import pynvml
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict
import statistics


@dataclass
class HardwareSnapshot:
    """Snapshot of hardware metrics at a point in time."""
    timestamp: float
    gpu_memory_used: int  # bytes
    gpu_memory_total: int  # bytes
    gpu_utilization: int  # percentage
    memory_bandwidth_used: float  # GB/s (if available)
    
    @classmethod
    def capture(cls) -> "HardwareSnapshot":
        """Capture current hardware state using PyNVML."""
        try:
            nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
            utilization = pynvml.nvmlDeviceGetUtilizationRates(nvml_handle)
            
            # Note: Memory bandwidth requires NVML v2+ or estimation
            # We'll estimate based on memory transactions if available
            try:
                # Try to get memory bandwidth (may not be available on all GPUs)
                bandwidth = pynvml.nvmlDeviceGetFieldValues(
                    nvml_handle, 
                    [pynvml.NVML_FI_DEV_MEMORY_CLOCK]
                )[0].value
                # This is memory clock, not actual bandwidth - will be refined
                memory_bandwidth = 0.0
            except Exception:
                memory_bandwidth = 0.0
                
            return cls(
                timestamp=time.time(),
                gpu_memory_used=memory_info.used,
                gpu_memory_total=memory_info.total,
                gpu_utilization=utilization.gpu,
                memory_bandwidth_used=memory_bandwidth,
            )
        except Exception:
            # Fallback if PyNVML not available
            return cls(
                timestamp=time.time(),
                gpu_memory_used=torch.cuda.memory_allocated(),
                gpu_memory_total=torch.cuda.get_device_properties(0).total_memory,
                gpu_utilization=0,
                memory_bandwidth_used=0.0,
            )


@dataclass
class MacroMetrics:
    """Macro-level performance metrics."""
    total_tokens: int = 0
    total_time: float = 0.0
    ttft: float = 0.0  # Time to first token (seconds)
    tpt_sum: float = 0.0  # Sum of time per token
    tpt_count: int = 0
    prefill_tokens: int = 0
    prefill_time: float = 0.0
    decode_tokens: int = 0
    decode_time: float = 0.0
    
    @property
    def throughput(self) -> float:
        """Tokens per second."""
        if self.total_time == 0:
            return 0.0
        return self.total_tokens / self.total_time
    
    @property
    def avg_tpt(self) -> float:
        """Average time per token (seconds)."""
        if self.tpt_count == 0:
            return 0.0
        return self.tpt_sum / self.tpt_count
    
    @property
    def avg_prefill_speed(self) -> float:
        """Prefill tokens per second."""
        if self.prefill_time == 0:
            return 0.0
        return self.prefill_tokens / self.prefill_time
    
    @property
    def avg_decode_speed(self) -> float:
        """Decode tokens per second."""
        if self.decode_time == 0:
            return 0.0
        return self.decode_tokens / self.decode_time


@dataclass
class HardwareMetrics:
    """Hardware and memory-related metrics."""
    # Arithmetic Intensity: FLOPs per Byte
    total_flops: float = 0.0
    total_bytes_accessed: float = 0.0
    
    # Memory Bandwidth Utilization
    memory_bandwidth_samples: list[float] = field(default_factory=list)
    
    # Weight Reuse Factor
    weight_loads: int = 0  # Number of times weights loaded from VRAM
    successful_tokens: int = 0  # Tokens successfully generated
    
    # KV Cache
    kv_cache_memory_bytes: int = 0
    kv_cache_blocks_used: int = 0
    kv_cache_blocks_total: int = 0
    
    @property
    def arithmetic_intensity(self) -> float:
        """FLOPs per Byte - higher means more compute-bound."""
        if self.total_bytes_accessed == 0:
            return 0.0
        return self.total_flops / self.total_bytes_accessed
    
    @property
    def avg_memory_bandwidth(self) -> float:
        """Average memory bandwidth utilization (GB/s)."""
        if not self.memory_bandwidth_samples:
            return 0.0
        return statistics.mean(self.memory_bandwidth_samples)
    
    @property
    def weight_reuse_factor(self) -> float:
        """Weights loaded per successfully generated token.
        
        Lower is better - indicates better weight reuse.
        For AR baseline: ~1.0 (weights loaded once per token)
        For EAGLE: < 1.0 (weights loaded once for multiple tokens)
        """
        if self.successful_tokens == 0:
            return 0.0
        return self.weight_loads / self.successful_tokens
    
    @property
    def kv_cache_utilization(self) -> float:
        """KV cache block utilization."""
        if self.kv_cache_blocks_total == 0:
            return 0.0
        return self.kv_cache_blocks_used / self.kv_cache_blocks_total


@dataclass
class AlgorithmicMetrics:
    """Speculative decoding algorithmic metrics."""
    # Acceptance statistics
    acceptance_lengths: list[int] = field(default_factory=list)
    acceptance_counts: dict[int, int] = field(default_factory=lambda: defaultdict(int))
    total_draft_tokens: int = 0
    total_accepted_tokens: int = 0
    total_rejected_tokens: int = 0
    
    # Draft head overhead
    draft_head_time: float = 0.0
    target_model_time: float = 0.0
    verification_time: float = 0.0
    
    # Speculation rounds
    speculation_rounds: int = 0
    
    @property
    def avg_acceptance_length(self) -> float:
        """Average number of tokens accepted per speculative step (τ)."""
        if not self.acceptance_lengths:
            return 0.0
        return statistics.mean(self.acceptance_lengths)
    
    @property
    def acceptance_rate(self) -> float:
        """Overall acceptance rate."""
        if self.total_draft_tokens == 0:
            return 0.0
        return self.total_accepted_tokens / self.total_draft_tokens
    
    @property
    def efficiency_score(self) -> float:
        """Efficiency: accepted tokens / total drafted tokens."""
        if self.total_draft_tokens == 0:
            return 0.0
        return self.total_accepted_tokens / self.total_draft_tokens
    
    @property
    def draft_overhead_ratio(self) -> float:
        """Percentage of time spent in draft head vs target model."""
        total_time = self.draft_head_time + self.target_model_time
        if total_time == 0:
            return 0.0
        return self.draft_head_time / total_time
    
    def get_acceptance_distribution(self) -> dict[int, float]:
        """Get acceptance rate distribution histogram.
        
        Returns dict mapping n (tokens accepted) -> probability.
        """
        total_rounds = self.speculation_rounds
        if total_rounds == 0:
            return {}
        return {
            n: count / total_rounds 
            for n, count in self.acceptance_counts.items()
        }


@dataclass
class BenchmarkMetrics:
    """Complete benchmark metrics for a single run."""
    mode: str  # "eagle" or "ar"
    macro: MacroMetrics = field(default_factory=MacroMetrics)
    hardware: HardwareMetrics = field(default_factory=HardwareMetrics)
    algorithmic: AlgorithmicMetrics = field(default_factory=AlgorithmicMetrics)
    
    # Metadata
    config: dict = field(default_factory=dict)
    hardware_info: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "mode": self.mode,
            "macro": {
                "total_tokens": self.macro.total_tokens,
                "total_time": self.macro.total_time,
                "throughput": self.macro.throughput,
                "ttft": self.macro.ttft,
                "avg_tpt": self.macro.avg_tpt,
                "prefill_tokens": self.macro.prefill_tokens,
                "prefill_time": self.macro.prefill_time,
                "decode_tokens": self.macro.decode_tokens,
                "decode_time": self.macro.decode_time,
                "avg_decode_speed": self.macro.avg_decode_speed,
            },
            "hardware": {
                "arithmetic_intensity": self.hardware.arithmetic_intensity,
                "avg_memory_bandwidth": self.hardware.avg_memory_bandwidth,
                "weight_reuse_factor": self.hardware.weight_reuse_factor,
                "kv_cache_utilization": self.hardware.kv_cache_utilization,
                "kv_cache_memory_bytes": self.hardware.kv_cache_memory_bytes,
            },
            "algorithmic": {
                "avg_acceptance_length": self.algorithmic.avg_acceptance_length,
                "acceptance_rate": self.algorithmic.acceptance_rate,
                "efficiency_score": self.algorithmic.efficiency_score,
                "draft_overhead_ratio": self.algorithmic.draft_overhead_ratio,
                "total_draft_tokens": self.algorithmic.total_draft_tokens,
                "total_accepted_tokens": self.algorithmic.total_accepted_tokens,
                "total_rejected_tokens": self.algorithmic.total_rejected_tokens,
                "speculation_rounds": self.algorithmic.speculation_rounds,
                "acceptance_distribution": self.algorithmic.get_acceptance_distribution(),
            },
            "config": self.config,
            "hardware_info": self.hardware_info,
        }


class MetricsCollector:
    """Collects and aggregates metrics during benchmark runs.
    
    Usage:
        collector = MetricsCollector(mode="eagle")
        
        # Start benchmark
        collector.start_run()
        
        # Record prefill
        collector.record_prefill_start()
        ... prefill execution ...
        collector.record_prefill_end(num_tokens)
        
        # Record decode steps
        for step in range(num_steps):
            collector.record_decode_step_start()
            ... decode execution ...
            collector.record_decode_step_end(num_tokens)
            
            # For EAGLE mode
            collector.record_speculation_round(
                draft_tokens=4,
                accepted_tokens=3,
                draft_time=0.001,
                verification_time=0.002,
            )
        
        # End benchmark
        metrics = collector.end_run()
    """
    
    def __init__(self, mode: str = "ar"):
        """Initialize metrics collector.
        
        Args:
            mode: "eagle" or "ar" (autoregressive baseline)
        """
        assert mode in ("eagle", "ar"), "Mode must be 'eagle' or 'ar'"
        self.mode = mode
        self.metrics = BenchmarkMetrics(mode=mode)
        
        # Timing state
        self._run_start_time: Optional[float] = None
        self._prefill_start_time: Optional[float] = None
        self._decode_step_start_time: Optional[float] = None
        self._speculation_start_time: Optional[float] = None
        
        # Hardware monitoring state
        self._hw_monitoring = False
        self._hw_snapshots: list[HardwareSnapshot] = []
        
        # Capture hardware info
        self._capture_hardware_info()
    
    def _capture_hardware_info(self):
        """Capture static hardware information."""
        info = {
            "gpu_name": "Unknown",
            "gpu_memory_total_gb": 0,
            "cuda_version": torch.version.cuda or "Unknown",
            "pytorch_version": torch.__version__,
        }
        
        try:
            pynvml.nvmlInit()
            nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            info["gpu_name"] = pynvml.nvmlDeviceGetName(nvml_handle)
            memory_info = pynvml.nvmlDeviceGetMemoryInfo(nvml_handle)
            info["gpu_memory_total_gb"] = memory_info.total / (1024**3)
            pynvml.nvmlShutdown()
        except Exception:
            # Fallback using torch
            try:
                props = torch.cuda.get_device_properties(0)
                info["gpu_name"] = props.name
                info["gpu_memory_total_gb"] = props.total_memory / (1024**3)
            except Exception:
                pass
        
        self.metrics.hardware_info = info
    
    def start_run(self, config: Optional[dict] = None):
        """Start a benchmark run."""
        self._run_start_time = time.time()
        self.metrics.config = config or {}
        self._hw_monitoring = True
        self._start_hw_monitoring()
    
    def end_run(self) -> BenchmarkMetrics:
        """End the benchmark run and return metrics."""
        if self._run_start_time is not None:
            self.metrics.macro.total_time = time.time() - self._run_start_time
        self._run_start_time = None
        self._hw_monitoring = False
        self._stop_hw_monitoring()
        
        # Compute hardware metrics from snapshots
        self._compute_hardware_metrics()
        
        return self.metrics
    
    def record_prefill_start(self):
        """Record the start of prefill phase."""
        self._prefill_start_time = time.time()
    
    def record_prefill_end(self, num_tokens: int):
        """Record the end of prefill phase."""
        if self._prefill_start_time is not None:
            elapsed = time.time() - self._prefill_start_time
            self.metrics.macro.prefill_time = elapsed
            self.metrics.macro.prefill_tokens = num_tokens
            self.metrics.macro.ttft = elapsed
            self._prefill_start_time = None
    
    def record_decode_step_start(self):
        """Record the start of a decode step."""
        self._decode_step_start_time = time.time()
    
    def record_decode_step_end(self, num_tokens: int):
        """Record the end of a decode step."""
        if self._decode_step_start_time is not None:
            elapsed = time.time() - self._decode_step_start_time
            self.metrics.macro.decode_time += elapsed
            self.metrics.macro.decode_tokens += num_tokens
            self.metrics.macro.tpt_sum += elapsed / num_tokens if num_tokens > 0 else 0
            self.metrics.macro.tpt_count += num_tokens
            self._decode_step_start_time = None
    
    def record_speculation_round(
        self,
        draft_tokens: int,
        accepted_tokens: int,
        draft_time: float = 0.0,
        verification_time: float = 0.0,
        target_time: float = 0.0,
    ):
        """Record a speculative decoding round (EAGLE mode only).
        
        Args:
            draft_tokens: Number of tokens drafted
            accepted_tokens: Number of tokens accepted
            draft_time: Time spent in draft model (seconds)
            verification_time: Time spent in verification (seconds)
            target_time: Time spent in target model forward (seconds)
        """
        if self.mode != "eagle":
            return
        
        rejected_tokens = draft_tokens - accepted_tokens
        
        # Update algorithmic metrics
        self.metrics.algorithmic.acceptance_lengths.append(accepted_tokens)
        self.metrics.algorithmic.acceptance_counts[accepted_tokens] += 1
        self.metrics.algorithmic.total_draft_tokens += draft_tokens
        self.metrics.algorithmic.total_accepted_tokens += accepted_tokens
        self.metrics.algorithmic.total_rejected_tokens += rejected_tokens
        self.metrics.algorithmic.speculation_rounds += 1
        
        # Update timing
        self.metrics.algorithmic.draft_head_time += draft_time
        self.metrics.algorithmic.verification_time += verification_time
        self.metrics.algorithmic.target_model_time += target_time
        
        # Update hardware metrics
        self.metrics.hardware.weight_loads += 1  # One target model forward pass
        self.metrics.hardware.successful_tokens += accepted_tokens
    
    def record_ar_token(self, flops: float = 0.0, bytes_accessed: float = 0.0):
        """Record a single AR token generation (baseline mode).
        
        Args:
            flops: FLOPs for this token
            bytes_accessed: Bytes accessed from memory
        """
        if self.mode == "ar":
            self.metrics.hardware.weight_loads += 1
            self.metrics.hardware.successful_tokens += 1
            self.metrics.hardware.total_flops += flops
            self.metrics.hardware.total_bytes_accessed += bytes_accessed
    
    def record_kv_cache_info(
        self,
        memory_bytes: int,
        blocks_used: int,
        blocks_total: int,
    ):
        """Record KV cache information."""
        self.metrics.hardware.kv_cache_memory_bytes = memory_bytes
        self.metrics.hardware.kv_cache_blocks_used = blocks_used
        self.metrics.hardware.kv_cache_blocks_total = blocks_total
    
    def _start_hw_monitoring(self):
        """Start hardware monitoring thread."""
        # Simple implementation - in production, use a background thread
        pass
    
    def _stop_hw_monitoring(self):
        """Stop hardware monitoring."""
        pass
    
    def _compute_hardware_metrics(self):
        """Compute aggregate hardware metrics from snapshots."""
        # For now, use PyNVML to get memory bandwidth estimate
        try:
            pynvml.nvmlInit()
            nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            
            # Get memory clock and bus width for bandwidth estimation
            mem_clock = pynvml.nvmlDeviceGetClockInfo(
                nvml_handle, pynvml.NVML_CLOCK_MEM
            )  # MHz
            
            # Estimate bandwidth based on memory type (simplified)
            # GDDR6X: ~1 GB/s per MHz of memory clock (approximate)
            # This is a rough estimate - actual bandwidth depends on GPU
            estimated_bandwidth = mem_clock * 0.8  # GB/s, rough estimate
            
            self.metrics.hardware.memory_bandwidth_samples.append(estimated_bandwidth)
            pynvml.nvmlShutdown()
        except Exception:
            pass
    
    def record_memory_bandwidth_sample(self, bandwidth_gb_s: float):
        """Record a memory bandwidth sample."""
        self.metrics.hardware.memory_bandwidth_samples.append(bandwidth_gb_s)
    
    def record_flops_and_bytes(self, flops: float, bytes_accessed: float):
        """Record FLOPs and bytes accessed for arithmetic intensity calculation."""
        self.metrics.hardware.total_flops += flops
        self.metrics.hardware.total_bytes_accessed += bytes_accessed


def compute_speedup(eagle_metrics: BenchmarkMetrics, ar_metrics: BenchmarkMetrics) -> float:
    """Compute speedup ratio: EAGLE throughput / AR throughput."""
    if ar_metrics.macro.throughput == 0:
        return 0.0
    return eagle_metrics.macro.throughput / ar_metrics.macro.throughput


def compare_metrics(eagle: BenchmarkMetrics, ar: BenchmarkMetrics) -> dict:
    """Compare EAGLE and AR metrics and return comparison report."""
    return {
        "speedup": compute_speedup(eagle, ar),
        "throughput_improvement": (
            eagle.macro.throughput - ar.macro.throughput
        ),
        "tpt_improvement": ar.macro.avg_tpt - eagle.macro.avg_tpt,
        "arithmetic_intensity_ratio": (
            eagle.hardware.arithmetic_intensity / ar.hardware.arithmetic_intensity
            if ar.hardware.arithmetic_intensity > 0 else 0
        ),
        "weight_reuse_improvement": (
            ar.hardware.weight_reuse_factor - eagle.hardware.weight_reuse_factor
        ),
        "eagle_acceptance_rate": eagle.algorithmic.acceptance_rate,
        "eagle_avg_acceptance_length": eagle.algorithmic.avg_acceptance_length,
        "eagle_efficiency": eagle.algorithmic.efficiency_score,
    }
