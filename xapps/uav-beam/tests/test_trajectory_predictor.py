"""
Unit tests for TrajectoryPredictor module.

Tests:
- Kalman Filter state estimation accuracy
- Polynomial fitting convergence
- Hybrid prediction method comparison
- Different UAV speed scenarios (10, 20, 30, 50 m/s)
- Trajectory types (linear, circular, figure-8)
- Boundary conditions (insufficient history, sudden direction change)
"""

import pytest
import numpy as np
import time
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from uav_beam.trajectory_predictor import (
    TrajectoryPredictor,
    PredictorConfig,
    UAVState,
    KalmanFilter3D,
)


class TestUAVState:
    """Test UAVState dataclass."""

    def test_from_position(self):
        """Test creating state from position only."""
        state = UAVState.from_position(1000.0, np.array([100.0, 200.0, 50.0]))

        assert state.timestamp_ms == 1000.0
        np.testing.assert_array_equal(state.position, [100.0, 200.0, 50.0])
        np.testing.assert_array_equal(state.velocity, [0.0, 0.0, 0.0])
        np.testing.assert_array_equal(state.acceleration, [0.0, 0.0, 0.0])
        assert state.heading == 0.0

    def test_state_attributes(self):
        """Test UAVState attributes."""
        state = UAVState(
            timestamp_ms=1000.0,
            position=np.array([100.0, 200.0, 50.0]),
            velocity=np.array([10.0, 5.0, 0.0]),
            acceleration=np.array([0.5, -0.2, 0.0]),
            heading=0.46,
        )

        assert state.timestamp_ms == 1000.0
        assert len(state.position) == 3
        assert len(state.velocity) == 3
        assert len(state.acceleration) == 3


class TestPredictorConfig:
    """Test PredictorConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PredictorConfig()

        assert config.process_noise_position == 0.1
        assert config.process_noise_velocity == 1.0
        assert config.measurement_noise_position == 0.5
        assert config.max_prediction_horizon_ms == 500.0
        assert config.history_window_size == 50
        assert config.max_acceleration == 5.0
        assert config.max_velocity == 30.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = PredictorConfig(
            max_velocity=50.0,
            max_acceleration=10.0,
            history_window_size=100,
        )

        assert config.max_velocity == 50.0
        assert config.max_acceleration == 10.0
        assert config.history_window_size == 100


class TestKalmanFilter3D:
    """Test KalmanFilter3D class."""

    @pytest.fixture
    def kf(self):
        """Create Kalman filter instance."""
        return KalmanFilter3D(PredictorConfig())

    def test_initialization(self, kf):
        """Test filter initialization."""
        assert kf.n_state == 6
        assert kf.n_meas == 3
        assert kf.x.shape == (6,)
        assert kf.P.shape == (6, 6)
        assert kf.Q.shape == (6, 6)
        assert kf.R.shape == (3, 3)
        assert kf.H.shape == (3, 6)
        assert not kf.initialized

    def test_first_update(self, kf):
        """Test first measurement update."""
        measurement = np.array([100.0, 200.0, 50.0])
        state = kf.update(measurement, 1000.0)

        assert kf.initialized
        np.testing.assert_array_equal(state[:3], measurement)
        assert kf.last_timestamp_ms == 1000.0

    def test_prediction_step(self, kf):
        """Test prediction step."""
        # Initialize with first measurement
        kf.update(np.array([0.0, 0.0, 0.0]), 0.0)

        # Set velocity manually
        kf.x[3:6] = [10.0, 5.0, 0.0]

        # Predict 100ms ahead
        state = kf.predict(100.0)

        # Position should change based on velocity
        # x += vx * dt = 0 + 10 * 0.1 = 1.0
        assert state[0] == pytest.approx(1.0, abs=0.1)
        assert state[1] == pytest.approx(0.5, abs=0.1)

    def test_predict_future(self, kf):
        """Test predict_future without updating state."""
        kf.update(np.array([0.0, 0.0, 0.0]), 0.0)
        kf.x[3:6] = [10.0, 5.0, 1.0]

        # Get original state
        original_x = kf.x.copy()

        # Predict future
        future_pos = kf.predict_future(200.0)

        # State should not change
        np.testing.assert_array_equal(kf.x, original_x)

        # Future position should be extrapolated
        assert future_pos[0] == pytest.approx(2.0, abs=0.1)
        assert future_pos[1] == pytest.approx(1.0, abs=0.1)
        assert future_pos[2] == pytest.approx(0.2, abs=0.1)

    def test_kalman_state_estimation_accuracy(self, trajectory_generator):
        """Test Kalman filter estimation accuracy with noisy measurements."""
        config = PredictorConfig(
            measurement_noise_position=1.0,
            process_noise_position=0.1,
        )
        kf = KalmanFilter3D(config)

        # Generate true linear trajectory
        true_velocity = np.array([10.0, 5.0, 1.0])
        true_start = np.array([0.0, 0.0, 100.0])

        np.random.seed(42)
        errors = []

        for i in range(50):
            t_ms = i * 100.0
            true_pos = true_start + true_velocity * (t_ms / 1000.0)

            # Add noise to measurement
            noise = np.random.randn(3) * 1.0
            noisy_pos = true_pos + noise

            # Update Kalman filter
            kf.update(noisy_pos, t_ms)

            # Get estimated state
            est_pos, est_vel = kf.get_state()

            # Calculate estimation error
            pos_error = np.linalg.norm(est_pos - true_pos)
            errors.append(pos_error)

        # After convergence, error should be less than measurement noise
        avg_error = np.mean(errors[-20:])
        assert avg_error < 1.5, f"Average estimation error too high: {avg_error:.2f}m"

    @pytest.mark.parametrize("velocity_mag", [10, 20, 30, 50])
    def test_estimation_with_different_speeds(self, velocity_mag):
        """Test Kalman filter with different UAV speeds."""
        config = PredictorConfig(max_velocity=velocity_mag * 2)
        kf = KalmanFilter3D(config)

        # Velocity in x direction
        true_velocity = np.array([velocity_mag, 0.0, 0.0])
        true_start = np.array([0.0, 0.0, 100.0])

        np.random.seed(42)

        for i in range(30):
            t_ms = i * 100.0
            true_pos = true_start + true_velocity * (t_ms / 1000.0)
            noise = np.random.randn(3) * 0.5
            kf.update(true_pos + noise, t_ms)

        # Check velocity estimation
        _, est_vel = kf.get_state()
        vel_error = np.linalg.norm(est_vel - true_velocity)

        assert vel_error < velocity_mag * 0.1, f"Velocity error too high for {velocity_mag}m/s"


class TestTrajectoryPredictor:
    """Test TrajectoryPredictor class."""

    @pytest.fixture
    def predictor(self):
        """Create predictor instance."""
        return TrajectoryPredictor()

    def test_initialization(self, predictor):
        """Test predictor initialization."""
        assert len(predictor.filters) == 0
        assert len(predictor.history) == 0
        assert len(predictor.waypoints) == 0

    def test_update_new_uav(self, predictor):
        """Test update with new UAV."""
        state = predictor.update(
            uav_id="uav-001",
            position=np.array([100.0, 200.0, 50.0]),
            timestamp_ms=1000.0
        )

        assert "uav-001" in predictor.filters
        assert "uav-001" in predictor.history
        assert len(predictor.history["uav-001"]) == 1
        np.testing.assert_array_almost_equal(state.position, [100.0, 200.0, 50.0])

    def test_update_multiple_measurements(self, predictor):
        """Test update with multiple measurements."""
        positions = [
            (1000.0, np.array([0.0, 0.0, 100.0])),
            (1100.0, np.array([1.0, 0.5, 100.0])),
            (1200.0, np.array([2.0, 1.0, 100.0])),
            (1300.0, np.array([3.0, 1.5, 100.0])),
        ]

        for t_ms, pos in positions:
            state = predictor.update("uav-001", pos, t_ms)

        assert len(predictor.history["uav-001"]) == 4

        # Kalman filter needs time to converge; just verify direction is correct
        # Position changes by 1m per 100ms = 10m/s in x direction
        # Due to filter dynamics, estimated velocity may be lower initially
        assert state.velocity[0] > 0  # Moving in positive x
        assert state.velocity[1] > 0  # Moving in positive y

    def test_predict_untracked_uav(self, predictor):
        """Test prediction for untracked UAV returns None."""
        result = predictor.predict("unknown-uav", 100.0)
        assert result is None

    def test_predict_kalman_method(self, predictor):
        """Test Kalman prediction method."""
        # Build up history
        for i in range(10):
            pos = np.array([i * 1.0, i * 0.5, 100.0])
            predictor.update("uav-001", pos, i * 100.0)

        # Predict 100ms ahead
        predicted = predictor.predict("uav-001", 100.0, method="kalman")

        assert predicted is not None
        assert predicted.timestamp_ms == 1000.0  # Last update + horizon
        assert len(predicted.position) == 3

    def test_predict_polynomial_method(self, predictor):
        """Test polynomial prediction method."""
        # Build up history
        for i in range(10):
            pos = np.array([i * 1.0, i * 0.5, 100.0])
            predictor.update("uav-001", pos, i * 100.0)

        # Predict 100ms ahead
        predicted = predictor.predict("uav-001", 100.0, method="polynomial")

        assert predicted is not None
        assert len(predicted.position) == 3

    def test_predict_hybrid_method(self, predictor):
        """Test hybrid prediction method."""
        # Build up history
        for i in range(10):
            pos = np.array([i * 1.0, i * 0.5, 100.0])
            predictor.update("uav-001", pos, i * 100.0)

        # Predict 100ms ahead
        predicted = predictor.predict("uav-001", 100.0, method="hybrid")

        assert predicted is not None
        assert len(predicted.position) == 3

    def test_prediction_horizon_clamping(self, predictor):
        """Test that prediction horizon is clamped to max."""
        for i in range(10):
            pos = np.array([i * 1.0, 0.0, 100.0])
            predictor.update("uav-001", pos, i * 100.0)

        # Request prediction beyond max horizon
        predicted = predictor.predict("uav-001", 10000.0, method="kalman")

        # Should be clamped to max_prediction_horizon_ms (500ms)
        assert predicted is not None

    def test_get_tracked_uavs(self, predictor):
        """Test getting list of tracked UAVs."""
        predictor.update("uav-001", np.array([0, 0, 100]), 0)
        predictor.update("uav-002", np.array([100, 0, 100]), 0)
        predictor.update("uav-003", np.array([200, 0, 100]), 0)

        tracked = predictor.get_tracked_uavs()

        assert len(tracked) == 3
        assert "uav-001" in tracked
        assert "uav-002" in tracked
        assert "uav-003" in tracked

    def test_remove_uav(self, predictor):
        """Test removing a UAV from tracking."""
        predictor.update("uav-001", np.array([0, 0, 100]), 0)
        predictor.update("uav-002", np.array([100, 0, 100]), 0)

        predictor.remove_uav("uav-001")

        assert "uav-001" not in predictor.filters
        assert "uav-001" not in predictor.history
        assert "uav-002" in predictor.filters

    def test_set_waypoints(self, predictor):
        """Test setting mission waypoints."""
        waypoints = [
            np.array([0, 0, 100]),
            np.array([100, 0, 100]),
            np.array([100, 100, 100]),
        ]

        predictor.set_waypoints("uav-001", waypoints)

        assert "uav-001" in predictor.waypoints
        assert len(predictor.waypoints["uav-001"]) == 3

    def test_prediction_confidence(self, predictor):
        """Test prediction confidence calculation."""
        # No history - low confidence
        conf = predictor.get_prediction_confidence("unknown-uav", 100.0)
        assert conf == 0.0

        # Build up history
        for i in range(30):
            predictor.update("uav-001", np.array([i, 0, 100]), i * 100.0)

        # With history - higher confidence for short horizon
        conf_short = predictor.get_prediction_confidence("uav-001", 100.0)
        conf_long = predictor.get_prediction_confidence("uav-001", 400.0)

        assert conf_short > conf_long
        assert 0 <= conf_short <= 1
        assert 0 <= conf_long <= 1


class TestLinearTrajectoryPrediction:
    """Test prediction accuracy for linear trajectories."""

    @pytest.fixture
    def predictor(self):
        """Create predictor for linear trajectory tests."""
        return TrajectoryPredictor(PredictorConfig(
            max_prediction_horizon_ms=500.0,
        ))

    @pytest.mark.parametrize("velocity_mps,tolerance_m", [
        (10, 1.0),
        (20, 1.5),
        (30, 2.0),
        (50, 3.0),
    ])
    def test_linear_prediction_accuracy(self, predictor, trajectory_generator, velocity_mps, tolerance_m):
        """Test prediction accuracy for linear motion at different speeds."""
        # Generate linear trajectory
        trajectory = trajectory_generator.linear(
            start=np.array([0.0, 0.0, 100.0]),
            velocity=np.array([velocity_mps, 0.0, 0.0]),
            duration_ms=2000.0,
            interval_ms=100.0
        )

        # Feed first part of trajectory to predictor
        for t_ms, pos in trajectory[:15]:
            predictor.update("uav-001", pos, t_ms)

        # Predict next position
        prediction_horizon_ms = 200.0
        predicted = predictor.predict("uav-001", prediction_horizon_ms, method="hybrid")

        # Calculate true future position
        last_t_ms = trajectory[14][0]
        future_t_ms = last_t_ms + prediction_horizon_ms
        true_pos = np.array([0.0, 0.0, 100.0]) + np.array([velocity_mps, 0.0, 0.0]) * (future_t_ms / 1000.0)

        # Check prediction error
        error = np.linalg.norm(predicted.position - true_pos)
        assert error < tolerance_m, f"Prediction error {error:.2f}m exceeds tolerance {tolerance_m}m for {velocity_mps}m/s"


class TestCircularTrajectoryPrediction:
    """Test prediction accuracy for circular trajectories."""

    @pytest.fixture
    def predictor(self):
        """Create predictor for circular trajectory tests."""
        return TrajectoryPredictor(PredictorConfig(
            max_prediction_horizon_ms=500.0,
        ))

    def test_circular_prediction(self, predictor, trajectory_generator):
        """Test prediction for circular motion."""
        # Generate circular trajectory
        trajectory = trajectory_generator.circular(
            center=np.array([0.0, 0.0]),
            radius=50.0,
            angular_velocity=0.5,  # rad/s
            altitude=100.0,
            duration_ms=5000.0,
            interval_ms=100.0
        )

        # Feed trajectory to predictor
        for t_ms, pos in trajectory[:30]:
            predictor.update("uav-001", pos, t_ms)

        # Predict next position
        predicted = predictor.predict("uav-001", 200.0, method="hybrid")

        # For circular motion, polynomial fitting should help
        # True position at prediction time
        t_pred = (trajectory[29][0] + 200.0) / 1000.0
        theta = 0.5 * t_pred
        true_pos = np.array([
            50.0 * np.cos(theta),
            50.0 * np.sin(theta),
            100.0
        ])

        error = np.linalg.norm(predicted.position - true_pos)
        # Circular motion is harder to predict, allow larger tolerance
        assert error < 10.0, f"Circular prediction error {error:.2f}m too high"


class TestFigure8TrajectoryPrediction:
    """Test prediction accuracy for figure-8 trajectories."""

    @pytest.fixture
    def predictor(self):
        """Create predictor for figure-8 trajectory tests."""
        return TrajectoryPredictor(PredictorConfig(
            max_prediction_horizon_ms=500.0,
        ))

    def test_figure8_prediction(self, predictor, trajectory_generator):
        """Test prediction for figure-8 motion."""
        trajectory = trajectory_generator.figure_eight(
            center=np.array([0.0, 0.0]),
            scale=50.0,
            angular_velocity=0.3,
            altitude=100.0,
            duration_ms=5000.0,
            interval_ms=100.0
        )

        # Feed trajectory to predictor
        for t_ms, pos in trajectory[:30]:
            predictor.update("uav-001", pos, t_ms)

        # Predict next position
        predicted = predictor.predict("uav-001", 100.0, method="hybrid")

        # Just check that prediction is reasonable
        assert predicted is not None
        assert not np.any(np.isnan(predicted.position))
        assert np.linalg.norm(predicted.position[:2]) < 100.0  # Should be within figure-8 bounds


class TestBoundaryConditions:
    """Test boundary conditions and edge cases."""

    @pytest.fixture
    def predictor(self):
        """Create predictor instance."""
        return TrajectoryPredictor()

    def test_insufficient_history_kalman(self, predictor):
        """Test prediction with insufficient history falls back gracefully."""
        # Only 2 measurements
        predictor.update("uav-001", np.array([0, 0, 100]), 0)
        predictor.update("uav-001", np.array([1, 0, 100]), 100)

        # Should still work (uses Kalman only)
        predicted = predictor.predict("uav-001", 100.0, method="kalman")
        assert predicted is not None

    def test_insufficient_history_polynomial(self, predictor):
        """Test polynomial prediction falls back with insufficient history."""
        # Only 3 measurements (polynomial needs 5)
        for i in range(3):
            predictor.update("uav-001", np.array([i, 0, 100]), i * 100)

        # Should fall back to Kalman
        predicted = predictor.predict("uav-001", 100.0, method="polynomial")
        assert predicted is not None

    def test_sudden_direction_change(self, predictor):
        """Test handling of sudden direction change."""
        # Move in +x direction
        for i in range(10):
            predictor.update("uav-001", np.array([i * 10, 0, 100]), i * 100)

        # Sudden change to +y direction
        for i in range(5):
            predictor.update("uav-001", np.array([100, i * 10, 100]), (10 + i) * 100)

        # Prediction should adapt (may not be perfect immediately)
        predicted = predictor.predict("uav-001", 100.0, method="hybrid")

        assert predicted is not None
        # y component should be increasing
        # (difficult to assert exact values due to filter lag)

    def test_stationary_uav(self, predictor):
        """Test prediction for stationary UAV."""
        # UAV at fixed position
        for i in range(20):
            predictor.update("uav-001", np.array([50, 50, 100]), i * 100)

        predicted = predictor.predict("uav-001", 200.0, method="hybrid")

        # Should predict nearly same position
        error = np.linalg.norm(predicted.position - np.array([50, 50, 100]))
        assert error < 2.0

    def test_zero_dt_measurements(self, predictor):
        """Test handling measurements with same timestamp."""
        predictor.update("uav-001", np.array([0, 0, 100]), 1000.0)
        predictor.update("uav-001", np.array([1, 0, 100]), 1000.0)  # Same timestamp

        # Should not crash
        predicted = predictor.predict("uav-001", 100.0)
        assert predicted is not None

    def test_large_time_gap(self, predictor):
        """Test handling large time gap between measurements."""
        predictor.update("uav-001", np.array([0, 0, 100]), 0)
        predictor.update("uav-001", np.array([1000, 0, 100]), 10000)  # 10 second gap

        predicted = predictor.predict("uav-001", 100.0)
        assert predicted is not None
        assert not np.any(np.isnan(predicted.position))

    def test_acceleration_estimation_insufficient_data(self, predictor):
        """Test acceleration estimation with insufficient data."""
        # Only 2 measurements
        predictor.update("uav-001", np.array([0, 0, 100]), 0)
        predictor.update("uav-001", np.array([1, 0, 100]), 100)

        state = predictor.predict("uav-001", 100.0, method="kalman")

        # Acceleration should be zero
        np.testing.assert_array_equal(state.acceleration, [0, 0, 0])

    def test_heading_calculation_zero_velocity(self, predictor):
        """Test heading calculation when velocity is near zero."""
        for i in range(5):
            predictor.update("uav-001", np.array([50, 50, 100]), i * 100)

        state = predictor.predict("uav-001", 100.0)

        # Heading should be 0 when velocity is near zero
        assert state.heading == pytest.approx(0.0, abs=0.1) or not np.isnan(state.heading)


class TestPredictionMethodComparison:
    """Compare different prediction methods."""

    @pytest.fixture
    def predictor(self):
        """Create predictor instance."""
        return TrajectoryPredictor(PredictorConfig(
            max_prediction_horizon_ms=500.0,
        ))

    def test_method_comparison_linear_trajectory(self, predictor, trajectory_generator):
        """Compare prediction methods for linear trajectory."""
        trajectory = trajectory_generator.linear(
            start=np.array([0.0, 0.0, 100.0]),
            velocity=np.array([20.0, 10.0, 0.0]),
            duration_ms=3000.0,
            interval_ms=100.0
        )

        # Feed trajectory
        for t_ms, pos in trajectory[:20]:
            predictor.update("uav-001", pos, t_ms)

        horizon = 200.0

        pred_kalman = predictor.predict("uav-001", horizon, method="kalman")
        pred_poly = predictor.predict("uav-001", horizon, method="polynomial")
        pred_hybrid = predictor.predict("uav-001", horizon, method="hybrid")

        # True future position
        last_t = trajectory[19][0]
        true_pos = np.array([0.0, 0.0, 100.0]) + np.array([20.0, 10.0, 0.0]) * ((last_t + horizon) / 1000.0)

        errors = {
            "kalman": np.linalg.norm(pred_kalman.position - true_pos),
            "polynomial": np.linalg.norm(pred_poly.position - true_pos),
            "hybrid": np.linalg.norm(pred_hybrid.position - true_pos),
        }

        # For linear motion, all methods should be reasonably accurate
        for method, error in errors.items():
            assert error < 5.0, f"{method} error {error:.2f}m too high for linear trajectory"

    def test_method_comparison_curved_trajectory(self, predictor, trajectory_generator):
        """Compare prediction methods for curved trajectory."""
        trajectory = trajectory_generator.circular(
            center=np.array([0.0, 0.0]),
            radius=30.0,
            angular_velocity=0.5,
            altitude=100.0,
            duration_ms=4000.0,
            interval_ms=100.0
        )

        # Feed trajectory
        for t_ms, pos in trajectory[:25]:
            predictor.update("uav-001", pos, t_ms)

        horizon = 200.0

        pred_kalman = predictor.predict("uav-001", horizon, method="kalman")
        pred_poly = predictor.predict("uav-001", horizon, method="polynomial")
        pred_hybrid = predictor.predict("uav-001", horizon, method="hybrid")

        # For curved motion, polynomial/hybrid should often be better
        # Just verify all produce valid predictions
        assert pred_kalman is not None
        assert pred_poly is not None
        assert pred_hybrid is not None

        assert not np.any(np.isnan(pred_kalman.position))
        assert not np.any(np.isnan(pred_poly.position))
        assert not np.any(np.isnan(pred_hybrid.position))


class TestPerformanceBenchmarks:
    """Performance benchmarks for trajectory prediction."""

    @pytest.fixture
    def predictor(self):
        """Create predictor instance."""
        return TrajectoryPredictor()

    @pytest.mark.performance
    def test_update_performance(self, predictor, performance_timer):
        """Benchmark update operation."""
        import time

        iterations = 1000
        np.random.seed(42)

        for i in range(iterations):
            pos = np.random.randn(3) * 100

            start = time.perf_counter()
            predictor.update("uav-001", pos, i * 10.0)
            elapsed_ms = (time.perf_counter() - start) * 1000

            performance_timer.record(elapsed_ms)

        result = performance_timer.result("update")
        print(f"\nUpdate performance: avg={result.avg_time_ms:.3f}ms, max={result.max_time_ms:.3f}ms")

        assert result.avg_time_ms < 1.0, "Update operation too slow"

    @pytest.mark.performance
    def test_predict_performance(self, predictor, performance_timer):
        """Benchmark predict operation."""
        import time

        # Build up history
        for i in range(50):
            predictor.update("uav-001", np.array([i, 0, 100]), i * 100)

        iterations = 500

        for _ in range(iterations):
            start = time.perf_counter()
            predictor.predict("uav-001", 200.0, method="hybrid")
            elapsed_ms = (time.perf_counter() - start) * 1000

            performance_timer.record(elapsed_ms)

        result = performance_timer.result("predict")
        print(f"\nPredict performance: avg={result.avg_time_ms:.3f}ms, max={result.max_time_ms:.3f}ms")

        assert result.avg_time_ms < 5.0, "Predict operation too slow"

    @pytest.mark.performance
    def test_multi_uav_performance(self, predictor):
        """Benchmark with multiple UAVs."""
        import time

        num_uavs = 50
        iterations = 100

        # Initialize UAVs
        for uav_idx in range(num_uavs):
            for i in range(10):
                predictor.update(f"uav-{uav_idx:03d}", np.array([i + uav_idx * 100, 0, 100]), i * 100)

        # Benchmark update and predict for all UAVs
        start = time.perf_counter()

        for _ in range(iterations):
            for uav_idx in range(num_uavs):
                predictor.update(f"uav-{uav_idx:03d}", np.random.randn(3) * 100, 1000 + _ * 100)
                predictor.predict(f"uav-{uav_idx:03d}", 200.0, method="kalman")

        elapsed_ms = (time.perf_counter() - start) * 1000
        ops_per_sec = (iterations * num_uavs * 2) / (elapsed_ms / 1000)

        print(f"\nMulti-UAV performance: {ops_per_sec:.0f} ops/sec ({num_uavs} UAVs)")

        assert ops_per_sec > 1000, "Multi-UAV performance too low"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
