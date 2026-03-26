"""Utility modules for EAGLE benchmark analysis and visualization.

This module provides:
1. Results aggregation from multiple benchmark runs
2. Visualization utilities (plots, charts)
3. Statistical analysis helpers
4. Report generation
"""

import json
import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import statistics

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class AggregatedResults:
    """Aggregated results from multiple benchmark runs."""
    mode: str
    num_runs: int
    
    # Macro metrics (mean ± std)
    throughput_mean: float
    throughput_std: float
    ttft_mean: float
    ttft_std: float
    tpt_mean: float
    tpt_std: float
    
    # Hardware metrics
    arithmetic_intensity_mean: float
    weight_reuse_mean: float
    
    # Algorithmic metrics (EAGLE only)
    acceptance_length_mean: float = 0.0
    acceptance_rate_mean: float = 0.0
    efficiency_mean: float = 0.0
    
    # Raw data for plotting
    throughput_samples: list = None
    acceptance_distribution: dict = None
    
    def __post_init__(self):
        if self.throughput_samples is None:
            self.throughput_samples = []
    
    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            f"Mode: {self.mode} ({self.num_runs} runs)",
            f"Throughput: {self.throughput_mean:.2f} ± {self.throughput_std:.2f} tok/s",
            f"TTFT: {self.ttft_mean*1000:.2f} ± {self.ttft_std*1000:.2f} ms",
            f"TPT: {self.tpt_mean*1000:.4f} ± {self.tpt_std*1000:.4f} ms",
        ]
        
        if self.mode == "eagle":
            lines.extend([
                f"Acceptance Length: {self.acceptance_length_mean:.2f}",
                f"Acceptance Rate: {self.acceptance_rate_mean:.2%}",
                f"Efficiency: {self.efficiency_mean:.2%}",
            ])
        
        return "\n".join(lines)


class ResultsAggregator:
    """Aggregate and analyze benchmark results from multiple runs."""
    
    def __init__(self, results_dir: str = "./benchmark_results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def load_results(self, pattern: str = "*.json") -> list[dict]:
        """Load all benchmark results matching pattern."""
        results_files = list(self.results_dir.glob(pattern))
        
        results = []
        for filepath in results_files:
            try:
                with open(filepath) as f:
                    data = json.load(f)
                    results.append(data)
            except Exception as e:
                print(f"Warning: Could not load {filepath}: {e}")
        
        return results
    
    def aggregate_by_mode(self, results: list[dict], mode: str) -> AggregatedResults:
        """Aggregate results for a specific mode."""
        # Filter results by mode
        mode_results = [r for r in results if r.get("config", {}).get("mode") == mode]
        
        if not mode_results:
            raise ValueError(f"No results found for mode: {mode}")
        
        # Extract metrics
        throughputs = []
        ttfts = []
        tpts = []
        arith_intensities = []
        weight_reuses = []
        acceptance_lengths = []
        acceptance_rates = []
        efficiencies = []
        acceptance_distributions = []
        
        for result in mode_results:
            runs = result.get("runs", {})
            mode_data = runs.get(mode, {})
            
            macro = mode_data.get("macro", {})
            hardware = mode_data.get("hardware", {})
            algorithmic = mode_data.get("algorithmic", {})
            
            if macro.get("throughput", 0) > 0:
                throughputs.append(macro["throughput"])
                ttfts.append(macro.get("ttft", 0))
                tpts.append(macro.get("avg_tpt", 0))
            
            if hardware.get("arithmetic_intensity", 0) > 0:
                arith_intensities.append(hardware["arithmetic_intensity"])
            
            if hardware.get("weight_reuse_factor", 0) > 0:
                weight_reuses.append(hardware["weight_reuse_factor"])
            
            if mode == "eagle":
                if algorithmic.get("avg_acceptance_length", 0) > 0:
                    acceptance_lengths.append(algorithmic["avg_acceptance_length"])
                if algorithmic.get("acceptance_rate", 0) > 0:
                    acceptance_rates.append(algorithmic["acceptance_rate"])
                if algorithmic.get("efficiency_score", 0) > 0:
                    efficiencies.append(algorithmic["efficiency_score"])
                if algorithmic.get("acceptance_distribution"):
                    acceptance_distributions.append(algorithmic["acceptance_distribution"])
        
        # Compute statistics
        def safe_mean_std(values):
            if len(values) < 2:
                return (values[0] if values else 0), 0.0
            return statistics.mean(values), statistics.stdev(values)
        
        throughput_mean, throughput_std = safe_mean_std(throughputs)
        ttft_mean, ttft_std = safe_mean_std(ttfts)
        tpt_mean, tpt_std = safe_mean_std(tpts)
        ai_mean, _ = safe_mean_std(arith_intensities)
        wr_mean, _ = safe_mean_std(weight_reuses)
        
        acc_len_mean, _ = safe_mean_std(acceptance_lengths)
        acc_rate_mean, _ = safe_mean_std(acceptance_rates)
        eff_mean, _ = safe_mean_std(efficiencies)
        
        # Average acceptance distribution
        avg_acc_dist = {}
        if acceptance_distributions:
            all_keys = set()
            for dist in acceptance_distributions:
                all_keys.update(dist.keys())
            
            for key in all_keys:
                values = [dist.get(key, 0) for dist in acceptance_distributions]
                avg_acc_dist[key] = statistics.mean(values)
        
        return AggregatedResults(
            mode=mode,
            num_runs=len(mode_results),
            throughput_mean=throughput_mean,
            throughput_std=throughput_std,
            ttft_mean=ttft_mean,
            ttft_std=ttft_std,
            tpt_mean=tpt_mean,
            tpt_std=tpt_std,
            arithmetic_intensity_mean=ai_mean,
            weight_reuse_mean=wr_mean,
            acceptance_length_mean=acc_len_mean,
            acceptance_rate_mean=acc_rate_mean,
            efficiency_mean=eff_mean,
            throughput_samples=throughputs,
            acceptance_distribution=avg_acc_dist,
        )
    
    def compare_modes(self, eagle: AggregatedResults, ar: AggregatedResults) -> dict:
        """Compare EAGLE and AR aggregated results."""
        speedup = eagle.throughput_mean / ar.throughput_mean if ar.throughput_mean > 0 else 0
        
        return {
            "speedup": speedup,
            "speedup_std": eagle.throughput_std / ar.throughput_mean if ar.throughput_mean > 0 else 0,
            "throughput_improvement": eagle.throughput_mean - ar.throughput_mean,
            "ttft_delta": eagle.ttft_mean - ar.ttft_mean,
            "tpt_improvement": ar.tpt_mean - eagle.tpt_mean,
            "arithmetic_intensity_ratio": (
                eagle.arithmetic_intensity_mean / ar.arithmetic_intensity_mean
                if ar.arithmetic_intensity_mean > 0 else 0
            ),
            "weight_reuse_improvement": ar.weight_reuse_mean - eagle.weight_reuse_mean,
        }
    
    def generate_report(self, output_path: Optional[str] = None) -> str:
        """Generate a summary report from all results."""
        results = self.load_results()
        
        if not results:
            return "No results found."
        
        lines = [
            "=" * 60,
            "EAGLE BENCHMARK AGGREGATION REPORT",
            "=" * 60,
            f"Total runs analyzed: {len(results)}",
            "",
        ]
        
        # Aggregate by mode
        modes = set()
        for r in results:
            mode = r.get("config", {}).get("mode", "unknown")
            modes.add(mode)
        
        aggregated = {}
        for mode in modes:
            try:
                agg = self.aggregate_by_mode(results, mode)
                aggregated[mode] = agg
                lines.append(f"--- {mode.upper()} ({agg.num_runs} runs) ---")
                lines.append(agg.summary())
                lines.append("")
            except ValueError as e:
                lines.append(f"Could not aggregate {mode}: {e}")
                lines.append("")
        
        # Compare modes if both available
        if "eagle" in aggregated and "ar" in aggregated:
            comparison = self.compare_modes(aggregated["eagle"], aggregated["ar"])
            
            lines.extend([
                "=" * 60,
                "COMPARISON RESULTS",
                "=" * 60,
                f"Speedup (EAGLE/AR): {comparison['speedup']:.2f}x ± {comparison['speedup_std']:.2f}",
                f"Throughput Improvement: {comparison['throughput_improvement']:.2f} tok/s",
                f"TPT Improvement: {comparison['tpt_improvement']*1000:.4f} ms",
                f"Arithmetic Intensity Ratio: {comparison['arithmetic_intensity_ratio']:.2f}x",
                f"Weight Reuse Improvement: {comparison['weight_reuse_improvement']:.2f}",
                "",
            ])
        
        report = "\n".join(lines)
        
        if output_path:
            with open(output_path, "w") as f:
                f.write(report)
            print(f"Report saved to: {output_path}")
        
        return report


class BenchmarkVisualizer:
    """Visualization utilities for benchmark results."""
    
    def __init__(self):
        if not HAS_MATPLOTLIB:
            raise ImportError("matplotlib is required for visualization. "
                            "Install with: pip install matplotlib")
        
        if not HAS_NUMPY:
            raise ImportError("numpy is required for visualization. "
                            "Install with: pip install numpy")
        
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        self.plt = plt
        self.np = np
    
    def plot_throughput_comparison(
        self,
        eagle_results: AggregatedResults,
        ar_results: AggregatedResults,
        output_path: str,
    ):
        """Plot throughput comparison bar chart."""
        fig, ax = self.plt.subplots(figsize=(10, 6))
        
        modes = ['AR Baseline', 'EAGLE']
        throughputs = [ar_results.throughput_mean, eagle_results.throughput_mean]
        errors = [ar_results.throughput_std, eagle_results.throughput_std]
        
        colors = ['#3498db', '#2ecc71']
        bars = ax.bar(modes, throughputs, yerr=errors, capsize=5, color=colors, alpha=0.8)
        
        # Add speedup annotation
        speedup = eagle_results.throughput_mean / ar_results.throughput_mean
        ax.text(
            1.5, max(throughputs) * 0.9,
            f'Speedup: {speedup:.2f}x',
            ha='center', va='top',
            fontsize=14, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        ax.set_ylabel('Throughput (tokens/s)')
        ax.set_title('EAGLE vs AR Baseline: Throughput Comparison')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, throughput in zip(bars, throughputs):
            ax.text(
                bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{throughput:.1f}',
                ha='center', va='bottom', fontsize=12
            )
        
        self.plt.tight_layout()
        self.plt.savefig(output_path, dpi=150, bbox_inches='tight')
        self.plt.close()
        print(f"Plot saved to: {output_path}")
    
    def plot_acceptance_distribution(
        self,
        acceptance_dist: dict,
        output_path: str,
    ):
        """Plot acceptance rate distribution histogram."""
        fig, ax = self.plt.subplots(figsize=(10, 6))
        
        # Sort by number of tokens accepted
        sorted_items = sorted(acceptance_dist.items(), key=lambda x: int(x[0]))
        n_values = [int(k) for k, v in sorted_items]
        probabilities = [v for k, v in sorted_items]
        
        colors = self.plt.cm.Blues(self.np.linspace(0.3, 0.9, len(n_values)))
        bars = ax.bar(n_values, probabilities, color=colors, alpha=0.8)
        
        # Add value labels
        for bar, prob in zip(bars, probabilities):
            ax.text(
                bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{prob:.1%}',
                ha='center', va='bottom', fontsize=10
            )
        
        ax.set_xlabel('Number of Tokens Accepted')
        ax.set_ylabel('Probability')
        ax.set_title('EAGLE Acceptance Rate Distribution')
        ax.set_xticks(n_values)
        ax.grid(axis='y', alpha=0.3)
        
        self.plt.tight_layout()
        self.plt.savefig(output_path, dpi=150, bbox_inches='tight')
        self.plt.close()
        print(f"Plot saved to: {output_path}")
    
    def plot_arithmetic_intensity_comparison(
        self,
        eagle_ai: float,
        ar_ai: float,
        output_path: str,
    ):
        """Plot arithmetic intensity comparison."""
        fig, ax = self.plt.subplots(figsize=(10, 6))
        
        modes = ['AR Baseline', 'EAGLE']
        values = [ar_ai, eagle_ai]
        colors = ['#e74c3c', '#9b59b6']
        
        bars = ax.bar(modes, values, color=colors, alpha=0.8)
        
        # Add memory wall annotation
        ax.axhline(y=3.0, color='gray', linestyle='--', alpha=0.7, label='Memory-bound threshold')
        ax.text(
            1.5, 3.2, 'More compute-bound',
            ha='center', va='bottom', fontsize=10, alpha=0.7
        )
        
        ax.set_ylabel('Arithmetic Intensity (FLOPs/Byte)')
        ax.set_title('Arithmetic Intensity: Memory Wall Analysis')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                f'{value:.2f}',
                ha='center', va='bottom', fontsize=12
            )
        
        self.plt.tight_layout()
        self.plt.savefig(output_path, dpi=150, bbox_inches='tight')
        self.plt.close()
        print(f"Plot saved to: {output_path}")
    
    def plot_all_metrics(
        self,
        eagle_results: AggregatedResults,
        ar_results: AggregatedResults,
        output_dir: str,
    ):
        """Generate all visualization plots."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        self.plot_throughput_comparison(
            eagle_results, ar_results,
            str(output_dir / "throughput_comparison.png")
        )
        
        if eagle_results.acceptance_distribution:
            self.plot_acceptance_distribution(
                eagle_results.acceptance_distribution,
                str(output_dir / "acceptance_distribution.png")
            )
        
        self.plot_arithmetic_intensity_comparison(
            eagle_results.arithmetic_intensity_mean,
            ar_results.arithmetic_intensity_mean,
            str(output_dir / "arithmetic_intensity.png")
        )
        
        print(f"All plots saved to: {output_dir}")


def main():
    """Main entry point for analysis utilities."""
    import argparse
    
    parser = argparse.ArgumentParser(description="EAGLE Benchmark Analysis Tools")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="./benchmark_results",
        help="Directory containing benchmark results"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./analysis_output",
        help="Directory for analysis output"
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Generate aggregation report"
    )
    parser.add_argument(
        "--plots",
        action="store_true",
        help="Generate visualization plots"
    )
    
    args = parser.parse_args()
    
    # Initialize aggregator
    aggregator = ResultsAggregator(args.results_dir)
    
    if args.report:
        report_path = Path(args.output_dir) / "aggregation_report.txt"
        aggregator.generate_report(str(report_path))
    
    if args.plots:
        if not HAS_MATPLOTLIB or not HAS_NUMPY:
            print("Error: matplotlib and numpy are required for plots")
            print("Install with: pip install matplotlib numpy")
            return 1
        
        results = aggregator.load_results()
        
        # Find eagle and ar results
        eagle_agg = None
        ar_agg = None
        
        for result in results:
            mode = result.get("config", {}).get("mode", "")
            if mode == "eagle" and eagle_agg is None:
                eagle_agg = aggregator.aggregate_by_mode(results, "eagle")
            elif mode == "ar" and ar_agg is None:
                ar_agg = aggregator.aggregate_by_mode(results, "ar")
        
        if eagle_agg and ar_agg:
            visualizer = BenchmarkVisualizer()
            visualizer.plot_all_metrics(eagle_agg, ar_agg, args.output_dir)
        else:
            print("Need both eagle and ar results for comparison plots")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
