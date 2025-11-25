"""
Trajectory Predictor Performance Benchmarks

Measures:
- Kalman Filter update latency
- Prediction accuracy vs prediction horizon
- Multi-UAV tracking performance
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

from uav_beam.trajectory_predictor import (
    TrajectoryPredictor,
    PredictorConfig,
    UAVState,
    KalmanFilter3D
)


@dataclass
class TrajectoryBenchmarkResult:
    """Benchmark result for trajectory prediction"""
    name: str
    iterations: int
    latencies_ms: List[float]
    memory_peak_mb: float
    position_errors_m: List[float]  # Position prediction errors

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
    def mean_error_m(self) -> float:
        return statistics.mean(self.position_errors_m) if self.position_errors_m else 0.0

    @property
    def rmse_m(self) -> float:
        if not self.position_errors_m:
            return 0.0
        return np.sqrt(np.mean(np.array(self.position_errors_m)**2))

    @property
    def throughput_ops(self) -> float:
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
                "mean": round(self.mean_latency, 4),
            },
            "throughput_ops": round(self.throughput_ops, 2),
            "accuracy": {
                "mean_error_m": round(self.mean_error_m, 4),
                "rmse_m": round(self.rmse_m, 4),
            },
            "memory_peak_mb": round(self.memory_peak_mb, 2),
        }


class UAVTrajectoryGenerator:
    """Generate realistic UAV trajectories for testing"""

    def __init__(self, seed: int = 42):
        np.random.seed(seed)

    def generate_linear_trajectory(
        self,
        start_pos: np.ndarray,
        velocity: np.ndarray,
        duration_ms: float,
        dt_ms: float = 10.0,
        noise_std: float = 0.5
    ) -> List[Tuple[float, np.ndarray]]:
        """Generate linear trajectory with noise"""
        trajectory = []
        t = 0.0
        while t <= duration_ms:
            true_pos = start_pos + velocity * (t / 1000.0)
            noisy_pos = true_pos + np.random.randn(3) * noise_std
            trajectory.append((t, noisy_pos))
            t += dt_ms
        return trajectory

    def generate_circular_trajectory(
        self,
        center: np.ndarray,
        radius: float,
        altitude: float,
        angular_velocity: float,  # rad/s
        duration_ms: float,
        dt_ms: float = 10.0,
        noise_std: float = 0.5
    ) -> List[Tuple[float, np.ndarray]]:
        """Generate circular trajectory"""
        trajectory = []
        t = 0.0
        while t <= duration_ms:
            theta = angular_velocity * (t / 1000.0)
            true_pos = np.array([
                center[0] + radius * np.cos(theta),
                center[1] + radius * np.sin(theta),
                altitude
            ])
            noisy_pos = true_pos + np.random.randn(3) * noise_std
            trajectory.append((t, noisy_pos))
            t += dt_ms
        return trajectory

    def generate_figure8_trajectory(
        self,
        center: np.ndarray,
        scale: float,
        altitude: float,
        period_ms: float,
        duration_ms: float,
        dt_ms: float = 10.0,
        noise_std: float = 0.5
    ) -> List[Tuple[float, np.ndarray]]:
        """Generate figure-8 trajectory"""
        trajectory = []
        t = 0.0
        while t <= duration_ms:
            phase = 2 * np.pi * t / period_ms
            true_pos = np.array([
                center[0] + scale * np.sin(phase),
                center[1] + scale * np.sin(2 * phase) / 2,
                altitude
            ])
            noisy_pos = true_pos + np.random.randn(3) * noise_std
            trajectory.append((t, noisy_pos))
            t += dt_ms
        return trajectory


class TrajectoryPredictorBenchmark:
    """Benchmark suite for trajectory predictor"""

    def __init__(self, warmup_iterations: int = 50, benchmark_iterations: int = 500):
        self.warmup_iterations = warmup_iterations
        self.benchmark_iterations = benchmark_iterations
        self.results: List[TrajectoryBenchmarkResult] = []
        self.trajectory_gen = UAVTrajectoryGenerator()

    def benchmark_kalman_update_latency(self) -> TrajectoryBenchmarkResult:
        """Benchmark Kalman filter update latency"""
        print("\n[Benchmark] Kalman Filter Update Latency")
        print("-" * 60)

        config = PredictorConfig()
        predictor = TrajectoryPredictor(config)
        latencies = []

        # Generate trajectory
        trajectory = self.trajectory_gen.generate_linear_trajectory(
            start_pos=np.array([0, 0, 100]),
            velocity=np.array([10, 5, 0]),
            duration_ms=self.benchmark_iterations * 10 + 1000,
            dt_ms=10.0
        )

        # Warmup
        for t, pos in trajectory[:self.warmup_iterations]:
            predictor.update("uav_warmup", pos, t)

        predictor.remove_uav("uav_warmup")
        gc.collect()
        tracemalloc.start()

        # Benchmark
        for i, (t, pos) in enumerate(trajectory[self.warmup_iterations:self.warmup_iterations + self.benchmark_iterations]):
            start = time.perf_counter()
            state = predictor.update("uav_0", pos, t)
            end = time.perf_counter()

            latencies.append((end - start) * 1000)

        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        result = TrajectoryBenchmarkResult(
            name="kalman_update",
            iterations=self.benchmark_iterations,
            latencies_ms=latencies,
            memory_peak_mb=peak / 1024 / 1024,
            position_errors_m=[]  # Not measuring accuracy here
        )

        print(f"  Latency: p50={result.p50:.4f}, p95={result.p95:.4f}, p99={result.p99:.4f} ms")
        print(f"  Throughput: {result.throughput_ops:.0f} updates/sec")
        print(f"  Memory: {result.memory_peak_mb:.2f} MB (peak)")

        self.results.append(result)
        return result

    def benchmark_prediction_accuracy(
        self,
        horizons_ms: List[float] = [10, 20, 50, 100, 200, 500]
    ) -> Dict[float, Dict[str, float]]:
        """Benchmark prediction accuracy vs horizon"""
        print("\n[Benchmark] Prediction Accuracy vs Horizon")
        print("-" * 60)

        accuracy_results = {}

        for horizon_ms in horizons_ms:
            config = PredictorConfig(max_prediction_horizon_ms=max(500, horizon_ms))
            predictor = TrajectoryPredictor(config)

            # Generate circular trajectory (challenging for prediction)
            trajectory = self.trajectory_gen.generate_circular_trajectory(
                center=np.array([200, 200]),
                radius=100,
                altitude=100,
                angular_velocity=0.5,
                duration_ms=10000,
                dt_ms=10.0
            )

            errors_kalman = []
            errors_polynomial = []
            errors_hybrid = []
            latencies = []

            # Feed trajectory and measure prediction accuracy
            for i in range(len(trajectory) - int(horizon_ms / 10) - 1):
                t, pos = trajectory[i]
                predictor.update("uav_0", pos, t)

                if i >= 10:  # Need some history
                    # Get ground truth at future time
                    future_idx = i + int(horizon_ms / 10)
                    _, true_future_pos = trajectory[future_idx]

                    start = time.perf_counter()

                    # Kalman prediction
                    pred_kalman = predictor.predict("uav_0", horizon_ms, method="kalman")
                    if pred_kalman:
                        errors_kalman.append(np.linalg.norm(pred_kalman.position - true_future_pos))

                    # Polynomial prediction
                    pred_poly = predictor.predict("uav_0", horizon_ms, method="polynomial")
                    if pred_poly:
                        errors_polynomial.append(np.linalg.norm(pred_poly.position - true_future_pos))

                    # Hybrid prediction
                    pred_hybrid = predictor.predict("uav_0", horizon_ms, method="hybrid")
                    if pred_hybrid:
                        errors_hybrid.append(np.linalg.norm(pred_hybrid.position - true_future_pos))

                    end = time.perf_counter()
                    latencies.append((end - start) * 1000)

            accuracy_results[horizon_ms] = {
                "kalman_rmse_m": round(np.sqrt(np.mean(np.array(errors_kalman)**2)), 4) if errors_kalman else 0,
                "polynomial_rmse_m": round(np.sqrt(np.mean(np.array(errors_polynomial)**2)), 4) if errors_polynomial else 0,
                "hybrid_rmse_m": round(np.sqrt(np.mean(np.array(errors_hybrid)**2)), 4) if errors_hybrid else 0,
                "latency_ms": round(statistics.mean(latencies), 4) if latencies else 0,
            }

            print(f"  Horizon {horizon_ms:4.0f} ms: "
                  f"Kalman={accuracy_results[horizon_ms]['kalman_rmse_m']:.2f}m, "
                  f"Poly={accuracy_results[horizon_ms]['polynomial_rmse_m']:.2f}m, "
                  f"Hybrid={accuracy_results[horizon_ms]['hybrid_rmse_m']:.2f}m")

            predictor.remove_uav("uav_0")

        return accuracy_results

    def benchmark_multi_uav_tracking(
        self,
        uav_counts: List[int] = [1, 5, 10, 50, 100]
    ) -> List[TrajectoryBenchmarkResult]:
        """Benchmark multi-UAV tracking performance"""
        print("\n[Benchmark] Multi-UAV Tracking Performance")
        print("-" * 60)

        results = []

        for num_uavs in uav_counts:
            config = PredictorConfig()
            predictor = TrajectoryPredictor(config)
            latencies = []

            # Initialize all UAVs with different trajectories
            trajectories = {}
            for uav_idx in range(num_uavs):
                start_pos = np.array([
                    np.random.uniform(0, 1000),
                    np.random.uniform(0, 1000),
                    np.random.uniform(50, 200)
                ])
                velocity = np.array([
                    np.random.uniform(-15, 15),
                    np.random.uniform(-15, 15),
                    np.random.uniform(-2, 2)
                ])
                trajectories[f"uav_{uav_idx}"] = self.trajectory_gen.generate_linear_trajectory(
                    start_pos=start_pos,
                    velocity=velocity,
                    duration_ms=5000,
                    dt_ms=10.0
                )

            gc.collect()
            tracemalloc.start()

            # Simulate real-time updates
            num_steps = min(500, len(list(trajectories.values())[0]))
            for step in range(num_steps):
                for uav_id, trajectory in trajectories.items():
                    if step < len(trajectory):
                        t, pos = trajectory[step]

                        start = time.perf_counter()
                        predictor.update(uav_id, pos, t)
                        end = time.perf_counter()

                        latencies.append((end - start) * 1000)

            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()

            result = TrajectoryBenchmarkResult(
                name=f"multi_uav_{num_uavs}",
                iterations=len(latencies),
                latencies_ms=latencies,
                memory_peak_mb=peak / 1024 / 1024,
                position_errors_m=[]
            )

            print(f"\n  {num_uavs} UAVs:")
            print(f"    Latency: p50={result.p50:.4f}, p95={result.p95:.4f}, p99={result.p99:.4f} ms")
            print(f"    Throughput: {result.throughput_ops:.0f} updates/sec")
            print(f"    Memory: {result.memory_peak_mb:.2f} MB, {result.memory_peak_mb/num_uavs:.4f} MB/UAV")

            results.append(result)
            self.results.append(result)

        return results

    def benchmark_trajectory_types(self) -> Dict[str, Dict[str, float]]:
        """Benchmark prediction accuracy for different trajectory types"""
        print("\n[Benchmark] Prediction by Trajectory Type")
        print("-" * 60)

        config = PredictorConfig()
        horizon_ms = 100.0

        trajectory_types = {
            "linear": self.trajectory_gen.generate_linear_trajectory(
                start_pos=np.array([0, 0, 100]),
                velocity=np.array([15, 10, 0]),
                duration_ms=5000,
                dt_ms=10.0
            ),
            "circular": self.trajectory_gen.generate_circular_trajectory(
                center=np.array([200, 200]),
                radius=100,
                altitude=100,
                angular_velocity=0.3,
                duration_ms=5000,
                dt_ms=10.0
            ),
            "figure8": self.trajectory_gen.generate_figure8_trajectory(
                center=np.array([200, 200]),
                scale=100,
                altitude=100,
                period_ms=5000,
                duration_ms=10000,
                dt_ms=10.0
            ),
        }

        results = {}

        for traj_name, trajectory in trajectory_types.items():
            predictor = TrajectoryPredictor(config)
            errors = []
            latencies = []

            horizon_steps = int(horizon_ms / 10)

            for i in range(len(trajectory) - horizon_steps - 1):
                t, pos = trajectory[i]
                predictor.update("uav_0", pos, t)

                if i >= 20:  # Need history
                    _, true_future_pos = trajectory[i + horizon_steps]

                    start = time.perf_counter()
                    pred = predictor.predict("uav_0", horizon_ms, method="hybrid")
                    end = time.perf_counter()

                    if pred:
                        errors.append(np.linalg.norm(pred.position - true_future_pos))
                        latencies.append((end - start) * 1000)

            results[traj_name] = {
                "rmse_m": round(np.sqrt(np.mean(np.array(errors)**2)), 4) if errors else 0,
                "mean_error_m": round(statistics.mean(errors), 4) if errors else 0,
                "max_error_m": round(max(errors), 4) if errors else 0,
                "latency_ms": round(statistics.mean(latencies), 4) if latencies else 0,
            }

            print(f"  {traj_name:10s}: RMSE={results[traj_name]['rmse_m']:.2f}m, "
                  f"Max={results[traj_name]['max_error_m']:.2f}m")

            predictor.remove_uav("uav_0")

        return results

    def get_summary(self) -> Dict[str, Any]:
        """Get summary of all benchmark results"""
        return {
            "benchmark": "trajectory_predictor",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "results": [r.to_dict() for r in self.results],
        }


def run_benchmarks(output_path: Optional[str] = None) -> Dict[str, Any]:
    """Run all trajectory predictor benchmarks"""
    print("=" * 60)
    print("UAV Trajectory Predictor Performance Benchmarks")
    print("=" * 60)

    benchmark = TrajectoryPredictorBenchmark(
        warmup_iterations=50,
        benchmark_iterations=500
    )

    # Run benchmarks
    benchmark.benchmark_kalman_update_latency()
    accuracy_results = benchmark.benchmark_prediction_accuracy()
    benchmark.benchmark_multi_uav_tracking()
    trajectory_results = benchmark.benchmark_trajectory_types()

    summary = benchmark.get_summary()
    summary["accuracy_vs_horizon"] = {str(k): v for k, v in accuracy_results.items()}
    summary["trajectory_types"] = trajectory_results

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
    output_file = Path(__file__).parent / "results" / "trajectory_results.json"
    output_file.parent.mkdir(exist_ok=True)
    run_benchmarks(str(output_file))
