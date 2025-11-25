"""
Angle Estimation Performance Benchmarks

Measures:
- MUSIC computation time vs antenna count
- ESPRIT computation time
- Beamspace method comparison
- Accuracy vs computation time tradeoffs
"""

import sys
import time
import gc
import statistics
import json
import tracemalloc
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from uav_beam.angle_estimator import (
    AngleEstimator,
    AngleEstimatorConfig,
    EstimationMethod,
    AngleEstimate
)


@dataclass
class EstimationBenchmarkResult:
    """Benchmark result for angle estimation"""
    name: str
    method: str
    antenna_elements: int
    iterations: int
    latencies_ms: List[float]
    memory_peak_mb: float
    accuracy_errors_rad: List[float]  # Angular estimation errors

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
    def mean_latency(self) -> float:
        return statistics.mean(self.latencies_ms) if self.latencies_ms else 0.0

    @property
    def mean_error_deg(self) -> float:
        if not self.accuracy_errors_rad:
            return 0.0
        return np.degrees(statistics.mean(self.accuracy_errors_rad))

    @property
    def rmse_deg(self) -> float:
        if not self.accuracy_errors_rad:
            return 0.0
        return np.degrees(np.sqrt(np.mean(np.array(self.accuracy_errors_rad)**2)))

    @property
    def throughput_ops(self) -> float:
        total_time = sum(self.latencies_ms) / 1000.0
        return self.iterations / total_time if total_time > 0 else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "method": self.method,
            "antenna_elements": self.antenna_elements,
            "iterations": self.iterations,
            "latency_ms": {
                "p50": round(self.p50, 4),
                "p95": round(self.p95, 4),
                "p99": round(self.p99, 4),
                "mean": round(self.mean_latency, 4),
            },
            "throughput_ops": round(self.throughput_ops, 2),
            "accuracy": {
                "mean_error_deg": round(self.mean_error_deg, 4),
                "rmse_deg": round(self.rmse_deg, 4),
            },
            "memory_peak_mb": round(self.memory_peak_mb, 2),
        }


def generate_signal(
    num_h: int,
    num_v: int,
    num_snapshots: int,
    true_azimuth: float,
    true_elevation: float,
    snr_db: float = 20.0,
    element_spacing: float = 0.5
) -> Tuple[np.ndarray, float, float]:
    """
    Generate synthetic received signal with known angle

    Args:
        num_h: Number of horizontal antenna elements
        num_v: Number of vertical antenna elements
        num_snapshots: Number of signal snapshots
        true_azimuth: Ground truth azimuth angle in radians
        true_elevation: Ground truth elevation angle in radians
        snr_db: Signal to noise ratio in dB
        element_spacing: Element spacing in wavelengths

    Returns:
        signal: (num_elements, num_snapshots) complex signal
        true_azimuth: Ground truth azimuth
        true_elevation: Ground truth elevation
    """
    num_elements = num_h * num_v
    d = element_spacing

    # Steering vector for UPA
    sv = np.zeros(num_elements, dtype=complex)
    elem_idx = 0
    for m in range(num_h):
        for n in range(num_v):
            phase = 2 * np.pi * element_spacing * (
                m * np.sin(true_azimuth) * np.cos(true_elevation) +
                n * np.sin(true_elevation)
            )
            sv[elem_idx] = np.exp(1j * phase)
            elem_idx += 1

    sv = sv / np.linalg.norm(sv)

    # Signal source
    signal_power = 1.0
    source = np.sqrt(signal_power) * np.exp(1j * 2 * np.pi * np.random.rand(1, num_snapshots))

    # Received signal
    signal = np.outer(sv, source[0])

    # Add noise
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = np.sqrt(noise_power / 2) * (
        np.random.randn(num_elements, num_snapshots) +
        1j * np.random.randn(num_elements, num_snapshots)
    )

    return signal + noise, true_azimuth, true_elevation


class AngleEstimationBenchmark:
    """Benchmark suite for angle estimation algorithms"""

    def __init__(self, warmup_iterations: int = 20, benchmark_iterations: int = 100):
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self.results: List[EstimationBenchmarkResult] = []

    def benchmark_music_vs_antenna_count(
        self,
        antenna_counts: List[int] = [16, 32, 64, 128]
    ) -> List[EstimationBenchmarkResult]:
        """Benchmark MUSIC computation time vs antenna count"""
        print("\n[Benchmark] MUSIC vs Antenna Count")
        print("-" * 60)

        results = []

        for num_elements in antenna_counts:
            num_h = int(np.sqrt(num_elements))
            num_v = num_elements // num_h

            config = AngleEstimatorConfig(
                num_elements_h=num_h,
                num_elements_v=num_v,
                angular_resolution=0.05,  # Coarser for speed
                num_snapshots=64
            )

            estimator = AngleEstimator(config)
            latencies = []
            errors = []

            # Warmup
            for _ in range(self.warmup_iterations):
                true_az = np.random.uniform(-np.pi/4, np.pi/4)
                true_el = np.random.uniform(-np.pi/8, np.pi/8)
                signal, _, _ = generate_signal(num_h, num_v, 64, true_az, true_el, snr_db=20)
                estimator.estimate(signal, 0.0, EstimationMethod.MUSIC)

            gc.collect()
            tracemalloc.start()

            # Benchmark
            for i in range(self.benchmark_iterations):
                true_az = np.random.uniform(-np.pi/4, np.pi/4)
                true_el = np.random.uniform(-np.pi/8, np.pi/8)
                signal, _, _ = generate_signal(num_h, num_v, 64, true_az, true_el, snr_db=20)

                start = time.perf_counter()
                result = estimator.estimate(signal, i * 10.0, EstimationMethod.MUSIC)
                end = time.perf_counter()

                latencies.append((end - start) * 1000)

                # Calculate error
                az_error = abs(result.azimuth_rad - true_az)
                el_error = abs(result.elevation_rad - true_el)
                errors.append(np.sqrt(az_error**2 + el_error**2))

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            result = EstimationBenchmarkResult(
                name=f"music_{num_elements}_elements",
                method="MUSIC",
                antenna_elements=num_elements,
                iterations=self.benchmark_iterations,
                latencies_ms=latencies,
                memory_peak_mb=peak / 1024 / 1024,
                accuracy_errors_rad=errors
            )

            print(f"\n  {num_elements} elements ({num_h}x{num_v}):")
            self._print_result(result)
            results.append(result)
            self.results.append(result)

        return results

    def benchmark_esprit(
        self,
        antenna_counts: List[int] = [16, 32, 64, 128]
    ) -> List[EstimationBenchmarkResult]:
        """Benchmark ESPRIT computation time"""
        print("\n[Benchmark] ESPRIT vs Antenna Count")
        print("-" * 60)

        results = []

        for num_elements in antenna_counts:
            num_h = int(np.sqrt(num_elements))
            num_v = num_elements // num_h

            config = AngleEstimatorConfig(
                num_elements_h=num_h,
                num_elements_v=num_v,
                num_snapshots=64
            )

            estimator = AngleEstimator(config)
            latencies = []
            errors = []

            # Warmup
            for _ in range(self.warmup_iterations):
                true_az = np.random.uniform(-np.pi/4, np.pi/4)
                signal, _, _ = generate_signal(num_h, num_v, 64, true_az, 0.0, snr_db=20)
                try:
                    estimator.estimate(signal, 0.0, EstimationMethod.ESPRIT)
                except Exception:
                    pass

            gc.collect()
            tracemalloc.start()

            # Benchmark
            for i in range(self.benchmark_iterations):
                true_az = np.random.uniform(-np.pi/4, np.pi/4)
                signal, _, _ = generate_signal(num_h, num_v, 64, true_az, 0.0, snr_db=20)

                start = time.perf_counter()
                try:
                    result = estimator.estimate(signal, i * 10.0, EstimationMethod.ESPRIT)
                    error = abs(result.azimuth_rad - true_az)
                except Exception:
                    result = None
                    error = np.pi  # Max error on failure
                end = time.perf_counter()

                latencies.append((end - start) * 1000)
                errors.append(error)

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            result = EstimationBenchmarkResult(
                name=f"esprit_{num_elements}_elements",
                method="ESPRIT",
                antenna_elements=num_elements,
                iterations=self.benchmark_iterations,
                latencies_ms=latencies,
                memory_peak_mb=peak / 1024 / 1024,
                accuracy_errors_rad=errors
            )

            print(f"\n  {num_elements} elements ({num_h}x{num_v}):")
            self._print_result(result)
            results.append(result)
            self.results.append(result)

        return results

    def benchmark_beamspace(
        self,
        antenna_counts: List[int] = [16, 32, 64, 128]
    ) -> List[EstimationBenchmarkResult]:
        """Benchmark Beamspace method"""
        print("\n[Benchmark] Beamspace vs Antenna Count")
        print("-" * 60)

        results = []

        for num_elements in antenna_counts:
            num_h = int(np.sqrt(num_elements))
            num_v = num_elements // num_h

            config = AngleEstimatorConfig(
                num_elements_h=num_h,
                num_elements_v=num_v,
                num_snapshots=64
            )

            estimator = AngleEstimator(config)
            latencies = []
            errors = []

            # Warmup
            for _ in range(self.warmup_iterations):
                true_az = np.random.uniform(-np.pi/4, np.pi/4)
                true_el = np.random.uniform(-np.pi/8, np.pi/8)
                signal, _, _ = generate_signal(num_h, num_v, 64, true_az, true_el, snr_db=20)
                estimator.estimate(signal, 0.0, EstimationMethod.BEAMSPACE)

            gc.collect()
            tracemalloc.start()

            # Benchmark
            for i in range(self.benchmark_iterations):
                true_az = np.random.uniform(-np.pi/4, np.pi/4)
                true_el = np.random.uniform(-np.pi/8, np.pi/8)
                signal, _, _ = generate_signal(num_h, num_v, 64, true_az, true_el, snr_db=20)

                start = time.perf_counter()
                result = estimator.estimate(signal, i * 10.0, EstimationMethod.BEAMSPACE)
                end = time.perf_counter()

                latencies.append((end - start) * 1000)

                az_error = abs(result.azimuth_rad - true_az)
                el_error = abs(result.elevation_rad - true_el)
                errors.append(np.sqrt(az_error**2 + el_error**2))

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            result = EstimationBenchmarkResult(
                name=f"beamspace_{num_elements}_elements",
                method="Beamspace",
                antenna_elements=num_elements,
                iterations=self.benchmark_iterations,
                latencies_ms=latencies,
                memory_peak_mb=peak / 1024 / 1024,
                accuracy_errors_rad=errors
            )

            print(f"\n  {num_elements} elements ({num_h}x{num_v}):")
            self._print_result(result)
            results.append(result)
            self.results.append(result)

        return results

    def benchmark_method_comparison(self, num_elements: int = 64) -> Dict[str, Any]:
        """Compare all methods at fixed antenna count"""
        print(f"\n[Benchmark] Method Comparison ({num_elements} elements)")
        print("-" * 60)

        num_h = int(np.sqrt(num_elements))
        num_v = num_elements // num_h

        config = AngleEstimatorConfig(
            num_elements_h=num_h,
            num_elements_v=num_v,
            angular_resolution=0.02,
            num_snapshots=64
        )

        estimator = AngleEstimator(config)

        methods = [
            EstimationMethod.MUSIC,
            EstimationMethod.ESPRIT,
            EstimationMethod.BEAMSPACE
        ]

        comparison = {}

        for method in methods:
            latencies = []
            errors = []

            gc.collect()

            for i in range(self.benchmark_iterations):
                true_az = np.random.uniform(-np.pi/4, np.pi/4)
                true_el = np.random.uniform(-np.pi/8, np.pi/8)
                signal, _, _ = generate_signal(num_h, num_v, 64, true_az, true_el, snr_db=20)

                start = time.perf_counter()
                try:
                    result = estimator.estimate(signal, i * 10.0, method)
                    az_error = abs(result.azimuth_rad - true_az)
                    el_error = abs(result.elevation_rad - true_el)
                    error = np.sqrt(az_error**2 + el_error**2)
                except Exception:
                    error = np.pi
                end = time.perf_counter()

                latencies.append((end - start) * 1000)
                errors.append(error)

            comparison[method.value] = {
                "latency_ms": {
                    "mean": round(statistics.mean(latencies), 4),
                    "p95": round(sorted(latencies)[int(len(latencies) * 0.95)], 4),
                },
                "accuracy": {
                    "mean_error_deg": round(np.degrees(statistics.mean(errors)), 4),
                    "rmse_deg": round(np.degrees(np.sqrt(np.mean(np.array(errors)**2))), 4),
                },
                "throughput_ops": round(len(latencies) / (sum(latencies) / 1000), 2),
            }

            print(f"\n  {method.value.upper()}:")
            print(f"    Latency: {comparison[method.value]['latency_ms']['mean']:.4f} ms (mean)")
            print(f"    RMSE: {comparison[method.value]['accuracy']['rmse_deg']:.4f} deg")
            print(f"    Throughput: {comparison[method.value]['throughput_ops']:.0f} ops/sec")

        return comparison

    def benchmark_snr_impact(
        self,
        snr_values: List[float] = [-5, 0, 5, 10, 15, 20, 25, 30]
    ) -> Dict[str, Any]:
        """Benchmark accuracy vs SNR"""
        print("\n[Benchmark] Accuracy vs SNR")
        print("-" * 60)

        num_elements = 64
        num_h = int(np.sqrt(num_elements))
        num_v = num_elements // num_h

        config = AngleEstimatorConfig(
            num_elements_h=num_h,
            num_elements_v=num_v,
            angular_resolution=0.02,
            num_snapshots=64
        )

        estimator = AngleEstimator(config)

        snr_results = {}

        for snr_db in snr_values:
            errors_music = []
            errors_beamspace = []

            for i in range(self.benchmark_iterations):
                true_az = np.random.uniform(-np.pi/4, np.pi/4)
                true_el = np.random.uniform(-np.pi/8, np.pi/8)
                signal, _, _ = generate_signal(num_h, num_v, 64, true_az, true_el, snr_db=snr_db)

                # MUSIC
                result = estimator.estimate(signal, i * 10.0, EstimationMethod.MUSIC)
                az_error = abs(result.azimuth_rad - true_az)
                el_error = abs(result.elevation_rad - true_el)
                errors_music.append(np.sqrt(az_error**2 + el_error**2))

                # Beamspace
                result = estimator.estimate(signal, i * 10.0, EstimationMethod.BEAMSPACE)
                az_error = abs(result.azimuth_rad - true_az)
                el_error = abs(result.elevation_rad - true_el)
                errors_beamspace.append(np.sqrt(az_error**2 + el_error**2))

            snr_results[snr_db] = {
                "music_rmse_deg": round(np.degrees(np.sqrt(np.mean(np.array(errors_music)**2))), 4),
                "beamspace_rmse_deg": round(np.degrees(np.sqrt(np.mean(np.array(errors_beamspace)**2))), 4),
            }

            print(f"  SNR {snr_db:3d} dB: MUSIC={snr_results[snr_db]['music_rmse_deg']:.2f} deg, "
                  f"Beamspace={snr_results[snr_db]['beamspace_rmse_deg']:.2f} deg")

        return snr_results

    def _print_result(self, result: EstimationBenchmarkResult):
        """Print benchmark result"""
        print(f"    Latency: p50={result.p50:.4f}, p95={result.p95:.4f}, p99={result.p99:.4f} ms")
        print(f"    Throughput: {result.throughput_ops:.0f} ops/sec")
        print(f"    Accuracy: RMSE={result.rmse_deg:.4f} deg")
        print(f"    Memory: {result.memory_peak_mb:.2f} MB (peak)")

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all benchmark results"""
        return {
            "benchmark": "angle_estimation",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": [r.to_dict() for r in self.results],
        }


def run_benchmarks(output_path: Optional[str] = None) -> Dict[str, Any]:
    """Run all angle estimation benchmarks"""
    print("=" * 60)
    print("UAV Beam Angle Estimation Performance Benchmarks")
    print("=" * 60)

    benchmark = AngleEstimationBenchmark(
        warmup_iterations=10,
        benchmark_iterations=50  # Fewer iterations due to MUSIC complexity
    )

    # Run benchmarks
    benchmark.benchmark_music_vs_antenna_count()
    benchmark.benchmark_esprit()
    benchmark.benchmark_beamspace()
    comparison = benchmark.benchmark_method_comparison()
    snr_results = benchmark.benchmark_snr_impact()

    summary = benchmark.get_summary()
    summary["method_comparison"] = comparison
    summary["snr_analysis"] = snr_results

    # Save results
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"\nResults saved to: {output_path}")

    print("\n" + "=" * 60)
    print("Benchmark Complete")
    print("=" * 60)

    return summary


if __name__ == "__main__":
    output_file = Path(__file__).parent / "results" / "angle_estimation_results.json"
    output_file.parent.mkdir(exist_ok=True)
    run_benchmarks(str(output_file))
