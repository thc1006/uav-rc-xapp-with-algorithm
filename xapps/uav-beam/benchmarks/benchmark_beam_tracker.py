"""
Beam Tracker Performance Benchmarks

Measures:
- Single decision latency (target < 1ms)
- Batch processing throughput
- Memory usage
- Scalability with different UE counts (1, 10, 100, 1000)
"""

import sys
import time
import gc
import statistics
import json
import tracemalloc
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uav_beam.beam_tracker import BeamTracker, BeamConfig, BeamMeasurement, BeamProcedure


@dataclass
class BenchmarkResult:
    """Benchmark result container"""
    name: str
    iterations: int
    latencies_ms: List[float]
    memory_peak_mb: float
    memory_current_mb: float

    @property
    def p50(self) -> float:
        sorted_latencies = sorted(self.latencies_ms)
        idx = int(len(sorted_latencies) * 0.50)
        return sorted_latencies[idx] if sorted_latencies else 0.0

    @property
    def p95(self) -> float:
        sorted_latencies = sorted(self.latencies_ms)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)] if sorted_latencies else 0.0

    @property
    def p99(self) -> float:
        sorted_latencies = sorted(self.latencies_ms)
        idx = int(len(sorted_latencies) * 0.99)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)] if sorted_latencies else 0.0

    @property
    def mean(self) -> float:
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def std(self) -> float:
        return statistics.stdev(self.latencies_ms) if len(self.latencies_ms) > 1 else 0.0

    @property
    def min_latency(self) -> float:
        return min(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def max_latency(self) -> float:
        return max(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def throughput_ops(self) -> float:
        """Operations per second"""
        total_time = sum(self.latencies_ms) / 1000.0
        return self.iterations / total_time if total_time > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "iterations": self.iterations,
            "latency_ms": {
                "p50": round(self.p50, 4),
                "p95": round(self.p95, 4),
                "p99": round(self.p99, 4),
                "mean": round(self.mean, 4),
                "std": round(self.std, 4),
                "min": round(self.min_latency, 4),
                "max": round(self.max_latency, 4),
            },
            "throughput_ops": round(self.throughput_ops, 2),
            "memory_mb": {
                "peak": round(self.memory_peak_mb, 2),
                "current": round(self.memory_current_mb, 2),
            },
            "target_met": self.p99 < 1.0,  # Target < 1ms
        }


def generate_measurement(
    ue_id: str,
    timestamp_ms: float,
    serving_beam: int,
    with_position: bool = False
) -> BeamMeasurement:
    """Generate a synthetic beam measurement"""
    # Random RSRP with realistic values
    serving_rsrp = np.random.uniform(-80, -60)

    # Generate neighbor beam measurements
    neighbor_beams = {}
    for i in range(5):
        beam_id = (serving_beam + i + 1) % 128
        neighbor_beams[beam_id] = serving_rsrp - np.random.uniform(3, 15)

    position = None
    velocity = None
    if with_position:
        position = (
            np.random.uniform(0, 500),
            np.random.uniform(0, 500),
            np.random.uniform(50, 200)
        )
        velocity = (
            np.random.uniform(-10, 10),
            np.random.uniform(-10, 10),
            np.random.uniform(-2, 2)
        )

    return BeamMeasurement(
        timestamp_ms=timestamp_ms,
        ue_id=ue_id,
        serving_beam_id=serving_beam,
        serving_rsrp_dbm=serving_rsrp,
        neighbor_beams=neighbor_beams,
        cqi=np.random.randint(1, 16),
        position=position,
        velocity=velocity
    )


class BeamTrackerBenchmark:
    """Benchmark suite for BeamTracker"""

    def __init__(self, warmup_iterations: int = 100, benchmark_iterations: int = 1000):
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self.results: List[BenchmarkResult] = []

    def benchmark_single_decision_latency(self) -> BenchmarkResult:
        """Benchmark single beam decision latency"""
        print("\n[Benchmark] Single Decision Latency")
        print("-" * 50)

        tracker = BeamTracker()
        latencies = []

        # Warmup
        for i in range(self.warmup_iterations):
            measurement = generate_measurement("warmup_ue", i * 10.0, i % 128)
            tracker.process_measurement(measurement)

        # Clear state
        tracker.reset()
        gc.collect()

        # Start memory tracking
        tracemalloc.start()

        # Benchmark
        for i in range(self.benchmark_iterations):
            measurement = generate_measurement("ue_0", i * 10.0, i % 128, with_position=True)

            start = time.perf_counter()
            decision = tracker.process_measurement(measurement)
            end = time.perf_counter()

            latencies.append((end - start) * 1000)  # Convert to ms

        # Get memory stats
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        result = BenchmarkResult(
            name="single_decision_latency",
            iterations=self.benchmark_iterations,
            latencies_ms=latencies,
            memory_peak_mb=peak / 1024 / 1024,
            memory_current_mb=current / 1024 / 1024
        )

        self._print_result(result)
        self.results.append(result)
        return result

    def benchmark_batch_throughput(self, batch_sizes: List[int] = [10, 50, 100, 500]) -> List[BenchmarkResult]:
        """Benchmark batch processing throughput"""
        print("\n[Benchmark] Batch Processing Throughput")
        print("-" * 50)

        results = []

        for batch_size in batch_sizes:
            tracker = BeamTracker()
            latencies = []

            # Generate batch of measurements
            def generate_batch():
                return [
                    generate_measurement(f"ue_{j}", j * 10.0, j % 128, with_position=True)
                    for j in range(batch_size)
                ]

            # Warmup
            for _ in range(10):
                batch = generate_batch()
                for m in batch:
                    tracker.process_measurement(m)

            tracker.reset()
            gc.collect()

            tracemalloc.start()

            # Benchmark
            iterations = max(100, self.benchmark_iterations // batch_size)
            for i in range(iterations):
                batch = generate_batch()

                start = time.perf_counter()
                for m in batch:
                    tracker.process_measurement(m)
                end = time.perf_counter()

                latencies.append((end - start) * 1000)  # Total batch time in ms

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            result = BenchmarkResult(
                name=f"batch_throughput_{batch_size}",
                iterations=iterations * batch_size,
                latencies_ms=latencies,
                memory_peak_mb=peak / 1024 / 1024,
                memory_current_mb=current / 1024 / 1024
            )

            print(f"\n  Batch Size: {batch_size}")
            self._print_result(result, indent=2)
            results.append(result)
            self.results.append(result)

        return results

    def benchmark_scalability(self, ue_counts: List[int] = [1, 10, 100, 1000]) -> List[BenchmarkResult]:
        """Benchmark scalability with different UE counts"""
        print("\n[Benchmark] Scalability (UE Count)")
        print("-" * 50)

        results = []

        for ue_count in ue_counts:
            tracker = BeamTracker()
            latencies = []

            # Initialize all UEs
            for ue_idx in range(ue_count):
                measurement = generate_measurement(
                    f"ue_{ue_idx}", 0, ue_idx % 128, with_position=True
                )
                tracker.process_measurement(measurement)

            gc.collect()
            tracemalloc.start()

            # Benchmark - process measurements for all UEs
            iterations = max(100, self.benchmark_iterations // ue_count)
            for i in range(iterations):
                ue_idx = i % ue_count
                measurement = generate_measurement(
                    f"ue_{ue_idx}", (i + 1) * 10.0, (ue_idx + i) % 128, with_position=True
                )

                start = time.perf_counter()
                tracker.process_measurement(measurement)
                end = time.perf_counter()

                latencies.append((end - start) * 1000)

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            result = BenchmarkResult(
                name=f"scalability_{ue_count}_ues",
                iterations=iterations,
                latencies_ms=latencies,
                memory_peak_mb=peak / 1024 / 1024,
                memory_current_mb=current / 1024 / 1024
            )

            print(f"\n  UE Count: {ue_count}")
            self._print_result(result, indent=2)
            results.append(result)
            self.results.append(result)

        return results

    def benchmark_memory_usage(self, ue_counts: List[int] = [1, 10, 100, 500, 1000]) -> Dict[int, float]:
        """Benchmark memory usage vs UE count"""
        print("\n[Benchmark] Memory Usage")
        print("-" * 50)

        memory_usage = {}

        for ue_count in ue_counts:
            gc.collect()
            tracemalloc.start()

            tracker = BeamTracker()

            # Initialize and process measurements for all UEs
            for ue_idx in range(ue_count):
                for t in range(10):  # 10 measurements per UE to build history
                    measurement = generate_measurement(
                        f"ue_{ue_idx}", t * 10.0, (ue_idx + t) % 128, with_position=True
                    )
                    tracker.process_measurement(measurement)

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            memory_mb = peak / 1024 / 1024
            memory_usage[ue_count] = memory_mb

            print(f"  {ue_count:4d} UEs: {memory_mb:6.2f} MB (peak), "
                  f"{memory_mb/ue_count:.4f} MB/UE")

        return memory_usage

    def benchmark_with_prediction(self) -> BenchmarkResult:
        """Benchmark decision latency with trajectory-based prediction"""
        print("\n[Benchmark] Decision with Trajectory Prediction")
        print("-" * 50)

        tracker = BeamTracker()
        latencies = []

        # Warmup with trajectory data
        for i in range(self.warmup_iterations):
            measurement = generate_measurement("warmup_ue", i * 10.0, i % 128, with_position=True)
            tracker.process_measurement(measurement)

        tracker.reset()
        gc.collect()
        tracemalloc.start()

        # Benchmark with trajectory prediction
        for i in range(self.benchmark_iterations):
            # Moving UAV simulation
            t = i * 10.0
            base_x = 100 + 10 * np.sin(t / 1000)
            base_y = 200 + 10 * np.cos(t / 1000)
            base_z = 100

            measurement = BeamMeasurement(
                timestamp_ms=t,
                ue_id="uav_0",
                serving_beam_id=i % 128,
                serving_rsrp_dbm=-70 + np.random.uniform(-5, 5),
                neighbor_beams={(i + j) % 128: -75 - j * 2 for j in range(1, 6)},
                cqi=12,
                position=(base_x, base_y, base_z),
                velocity=(10 * np.cos(t / 1000), -10 * np.sin(t / 1000), 0)
            )

            start = time.perf_counter()
            decision = tracker.process_measurement(measurement)
            end = time.perf_counter()

            latencies.append((end - start) * 1000)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        result = BenchmarkResult(
            name="decision_with_trajectory",
            iterations=self.benchmark_iterations,
            latencies_ms=latencies,
            memory_peak_mb=peak / 1024 / 1024,
            memory_current_mb=current / 1024 / 1024
        )

        self._print_result(result)
        self.results.append(result)
        return result

    def _print_result(self, result: BenchmarkResult, indent: int = 0):
        """Print benchmark result"""
        prefix = "  " * indent
        target_status = "PASS" if result.p99 < 1.0 else "FAIL"

        print(f"{prefix}  Iterations: {result.iterations}")
        print(f"{prefix}  Latency (ms): p50={result.p50:.4f}, p95={result.p95:.4f}, "
              f"p99={result.p99:.4f} [{target_status}]")
        print(f"{prefix}  Latency range: {result.min_latency:.4f} - {result.max_latency:.4f} ms")
        print(f"{prefix}  Throughput: {result.throughput_ops:.0f} ops/sec")
        print(f"{prefix}  Memory: {result.memory_peak_mb:.2f} MB (peak)")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all benchmark results"""
        return {
            "benchmark": "beam_tracker",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": [r.to_dict() for r in self.results],
            "summary": {
                "total_benchmarks": len(self.results),
                "targets_met": sum(1 for r in self.results if r.p99 < 1.0),
            }
        }


def run_benchmarks(output_path: Optional[str] = None) -> Dict[str, Any]:
    """Run all beam tracker benchmarks"""
    print("=" * 60)
    print("UAV Beam Tracker Performance Benchmarks")
    print("=" * 60)

    benchmark = BeamTrackerBenchmark(
        warmup_iterations=100,
        benchmark_iterations=1000
    )

    # Run all benchmarks
    benchmark.benchmark_single_decision_latency()
    benchmark.benchmark_with_prediction()
    benchmark.benchmark_batch_throughput()
    benchmark.benchmark_scalability()
    benchmark.benchmark_memory_usage()

    summary = benchmark.get_summary()

    # Save results
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 60)
    print(f"Summary: {summary['summary']['targets_met']}/{summary['summary']['total_benchmarks']} "
          f"benchmarks met target (< 1ms p99)")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    output_file = Path(__file__).parent / "results" / "beam_tracker_results.json"
    output_file.parent.mkdir(exist_ok=True)
    run_benchmarks(str(output_file))
