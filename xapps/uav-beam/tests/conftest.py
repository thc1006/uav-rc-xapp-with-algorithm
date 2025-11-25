"""
Pytest configuration and shared fixtures for UAV Beam xApp tests.

Provides:
- Test fixtures for beam tracking components
- Data generators for trajectory and signal simulation
- Mock objects for external dependencies
"""

import pytest
import numpy as np
from typing import List, Dict, Tuple, Generator
from dataclasses import dataclass
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from uav_beam.beam_tracker import (
    BeamTracker,
    BeamConfig,
    BeamMeasurement,
    BeamDecision,
    BeamState,
)
from uav_beam.trajectory_predictor import (
    TrajectoryPredictor,
    PredictorConfig,
    UAVState,
    KalmanFilter3D,
)
from uav_beam.angle_estimator import (
    AngleEstimator,
    AngleEstimatorConfig,
    AngleEstimate,
    EstimationMethod,
)


# ==============================================================================
# Configuration Fixtures
# ==============================================================================

@pytest.fixture
def default_beam_config() -> BeamConfig:
    """Default beam configuration for tests."""
    return BeamConfig()


@pytest.fixture
def small_beam_config() -> BeamConfig:
    """Small beam configuration for faster tests."""
    return BeamConfig(
        num_antenna_elements_h=4,
        num_antenna_elements_v=4,
        num_beams_h=8,
        num_beams_v=4,
    )


@pytest.fixture
def default_predictor_config() -> PredictorConfig:
    """Default predictor configuration."""
    return PredictorConfig()


@pytest.fixture
def fast_predictor_config() -> PredictorConfig:
    """Fast UAV predictor configuration."""
    return PredictorConfig(
        max_velocity=50.0,
        max_acceleration=10.0,
        max_prediction_horizon_ms=1000.0,
    )


@pytest.fixture
def small_estimator_config() -> AngleEstimatorConfig:
    """Small angle estimator configuration for faster tests."""
    return AngleEstimatorConfig(
        num_elements_h=4,
        num_elements_v=4,
        num_snapshots=32,
        angular_resolution=0.05,
    )


# ==============================================================================
# Component Fixtures
# ==============================================================================

@pytest.fixture
def beam_tracker(default_beam_config) -> BeamTracker:
    """Create a BeamTracker instance."""
    return BeamTracker(default_beam_config)


@pytest.fixture
def small_beam_tracker(small_beam_config) -> BeamTracker:
    """Create a BeamTracker with small configuration."""
    return BeamTracker(small_beam_config)


@pytest.fixture
def trajectory_predictor(default_predictor_config) -> TrajectoryPredictor:
    """Create a TrajectoryPredictor instance."""
    return TrajectoryPredictor(default_predictor_config)


@pytest.fixture
def kalman_filter(default_predictor_config) -> KalmanFilter3D:
    """Create a Kalman filter instance."""
    return KalmanFilter3D(default_predictor_config)


@pytest.fixture
def angle_estimator(small_estimator_config) -> AngleEstimator:
    """Create an AngleEstimator instance."""
    return AngleEstimator(small_estimator_config)


# ==============================================================================
# Trajectory Data Generators
# ==============================================================================

class TrajectoryGenerator:
    """Generate UAV trajectories for testing."""

    @staticmethod
    def linear(
        start: np.ndarray,
        velocity: np.ndarray,
        duration_ms: float,
        interval_ms: float = 100.0
    ) -> List[Tuple[float, np.ndarray]]:
        """
        Generate linear trajectory.

        Args:
            start: Starting position [x, y, z]
            velocity: Constant velocity [vx, vy, vz]
            duration_ms: Total duration in milliseconds
            interval_ms: Sample interval in milliseconds

        Returns:
            List of (timestamp_ms, position) tuples
        """
        trajectory = []
        num_samples = int(duration_ms / interval_ms) + 1

        for i in range(num_samples):
            t_ms = i * interval_ms
            t_s = t_ms / 1000.0
            pos = start + velocity * t_s
            trajectory.append((t_ms, pos.copy()))

        return trajectory

    @staticmethod
    def circular(
        center: np.ndarray,
        radius: float,
        angular_velocity: float,
        altitude: float,
        duration_ms: float,
        interval_ms: float = 100.0
    ) -> List[Tuple[float, np.ndarray]]:
        """
        Generate circular trajectory.

        Args:
            center: Center of circle [x, y]
            radius: Circle radius in meters
            angular_velocity: Angular velocity in rad/s
            altitude: Constant altitude in meters
            duration_ms: Total duration in milliseconds
            interval_ms: Sample interval in milliseconds

        Returns:
            List of (timestamp_ms, position) tuples
        """
        trajectory = []
        num_samples = int(duration_ms / interval_ms) + 1

        for i in range(num_samples):
            t_ms = i * interval_ms
            t_s = t_ms / 1000.0
            theta = angular_velocity * t_s
            pos = np.array([
                center[0] + radius * np.cos(theta),
                center[1] + radius * np.sin(theta),
                altitude
            ])
            trajectory.append((t_ms, pos.copy()))

        return trajectory

    @staticmethod
    def figure_eight(
        center: np.ndarray,
        scale: float,
        angular_velocity: float,
        altitude: float,
        duration_ms: float,
        interval_ms: float = 100.0
    ) -> List[Tuple[float, np.ndarray]]:
        """
        Generate figure-8 trajectory (lemniscate of Bernoulli).

        Args:
            center: Center of figure-8 [x, y]
            scale: Scale factor for size
            angular_velocity: Angular velocity in rad/s
            altitude: Constant altitude in meters
            duration_ms: Total duration in milliseconds
            interval_ms: Sample interval in milliseconds

        Returns:
            List of (timestamp_ms, position) tuples
        """
        trajectory = []
        num_samples = int(duration_ms / interval_ms) + 1

        for i in range(num_samples):
            t_ms = i * interval_ms
            t_s = t_ms / 1000.0
            theta = angular_velocity * t_s

            # Parametric equations for figure-8
            denom = 1 + np.sin(theta)**2
            x = scale * np.cos(theta) / denom
            y = scale * np.sin(theta) * np.cos(theta) / denom

            pos = np.array([
                center[0] + x,
                center[1] + y,
                altitude
            ])
            trajectory.append((t_ms, pos.copy()))

        return trajectory

    @staticmethod
    def random_walk(
        start: np.ndarray,
        step_size: float,
        duration_ms: float,
        interval_ms: float = 100.0,
        seed: int = 42
    ) -> List[Tuple[float, np.ndarray]]:
        """
        Generate random walk trajectory.

        Args:
            start: Starting position [x, y, z]
            step_size: Maximum step size in meters
            duration_ms: Total duration in milliseconds
            interval_ms: Sample interval in milliseconds
            seed: Random seed for reproducibility

        Returns:
            List of (timestamp_ms, position) tuples
        """
        np.random.seed(seed)
        trajectory = []
        num_samples = int(duration_ms / interval_ms) + 1

        pos = start.copy()
        for i in range(num_samples):
            t_ms = i * interval_ms
            trajectory.append((t_ms, pos.copy()))

            # Random step
            step = np.random.uniform(-step_size, step_size, 3)
            pos = pos + step

        return trajectory


@pytest.fixture
def trajectory_generator() -> TrajectoryGenerator:
    """Provide trajectory generator."""
    return TrajectoryGenerator()


# ==============================================================================
# Signal Data Generators
# ==============================================================================

class SignalGenerator:
    """Generate received signals for angle estimation tests."""

    @staticmethod
    def generate_signal(
        num_elements_h: int,
        num_elements_v: int,
        azimuth: float,
        elevation: float,
        snr_db: float,
        num_snapshots: int,
        element_spacing: float = 0.5,
        seed: int = 42
    ) -> np.ndarray:
        """
        Generate received signal from a single source.

        Args:
            num_elements_h: Number of horizontal antenna elements
            num_elements_v: Number of vertical antenna elements
            azimuth: Source azimuth angle in radians
            elevation: Source elevation angle in radians
            snr_db: Signal-to-noise ratio in dB
            num_snapshots: Number of signal snapshots
            element_spacing: Element spacing in wavelengths
            seed: Random seed

        Returns:
            Received signal matrix (num_elements, num_snapshots)
        """
        np.random.seed(seed)

        num_elements = num_elements_h * num_elements_v

        # Generate steering vector
        sv = np.zeros(num_elements, dtype=complex)
        elem_idx = 0
        for m in range(num_elements_h):
            for n in range(num_elements_v):
                phase = 2 * np.pi * element_spacing * (
                    m * np.sin(azimuth) * np.cos(elevation) +
                    n * np.sin(elevation)
                )
                sv[elem_idx] = np.exp(1j * phase)
                elem_idx += 1

        sv = sv / np.sqrt(num_elements)

        # Generate source signal
        signal_power = 1.0
        source_signal = np.sqrt(signal_power) * (
            np.random.randn(num_snapshots) + 1j * np.random.randn(num_snapshots)
        ) / np.sqrt(2)

        # Generate noise
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(num_elements, num_snapshots) +
            1j * np.random.randn(num_elements, num_snapshots)
        )

        # Received signal = steering_vector * source + noise
        received = np.outer(sv, source_signal) + noise

        return received

    @staticmethod
    def generate_multi_source_signal(
        num_elements_h: int,
        num_elements_v: int,
        sources: List[Tuple[float, float, float]],  # (azimuth, elevation, power_db)
        snr_db: float,
        num_snapshots: int,
        element_spacing: float = 0.5,
        seed: int = 42
    ) -> np.ndarray:
        """
        Generate received signal from multiple sources.

        Args:
            num_elements_h: Number of horizontal antenna elements
            num_elements_v: Number of vertical antenna elements
            sources: List of (azimuth, elevation, relative_power_db) tuples
            snr_db: Overall signal-to-noise ratio in dB
            num_snapshots: Number of signal snapshots
            element_spacing: Element spacing in wavelengths
            seed: Random seed

        Returns:
            Received signal matrix (num_elements, num_snapshots)
        """
        np.random.seed(seed)

        num_elements = num_elements_h * num_elements_v

        # Initialize received signal
        received = np.zeros((num_elements, num_snapshots), dtype=complex)

        for azimuth, elevation, power_db in sources:
            # Generate steering vector
            sv = np.zeros(num_elements, dtype=complex)
            elem_idx = 0
            for m in range(num_elements_h):
                for n in range(num_elements_v):
                    phase = 2 * np.pi * element_spacing * (
                        m * np.sin(azimuth) * np.cos(elevation) +
                        n * np.sin(elevation)
                    )
                    sv[elem_idx] = np.exp(1j * phase)
                    elem_idx += 1

            sv = sv / np.sqrt(num_elements)

            # Generate source signal
            signal_power = 10 ** (power_db / 10)
            source_signal = np.sqrt(signal_power) * (
                np.random.randn(num_snapshots) + 1j * np.random.randn(num_snapshots)
            ) / np.sqrt(2)

            received += np.outer(sv, source_signal)

        # Add noise
        total_signal_power = np.mean(np.abs(received)**2)
        noise_power = total_signal_power / (10 ** (snr_db / 10))
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(num_elements, num_snapshots) +
            1j * np.random.randn(num_elements, num_snapshots)
        )

        received += noise

        return received


@pytest.fixture
def signal_generator() -> SignalGenerator:
    """Provide signal generator."""
    return SignalGenerator()


# ==============================================================================
# Measurement Fixtures
# ==============================================================================

@pytest.fixture
def basic_measurement() -> BeamMeasurement:
    """Basic beam measurement for testing."""
    return BeamMeasurement(
        timestamp_ms=1000.0,
        ue_id="uav-001",
        serving_beam_id=42,
        serving_rsrp_dbm=-85.0,
        neighbor_beams={41: -88.0, 43: -87.5, 44: -90.0},
    )


@pytest.fixture
def measurement_with_position() -> BeamMeasurement:
    """Beam measurement with position and velocity."""
    return BeamMeasurement(
        timestamp_ms=1000.0,
        ue_id="uav-001",
        serving_beam_id=42,
        serving_rsrp_dbm=-85.0,
        neighbor_beams={41: -88.0, 43: -87.5},
        position=(100.0, 200.0, 50.0),
        velocity=(10.0, 5.0, 0.0),
    )


@pytest.fixture
def low_rsrp_measurement() -> BeamMeasurement:
    """Measurement with low RSRP (beam failure scenario)."""
    return BeamMeasurement(
        timestamp_ms=1000.0,
        ue_id="uav-002",
        serving_beam_id=42,
        serving_rsrp_dbm=-115.0,
        neighbor_beams={41: -100.0, 43: -105.0},
    )


def generate_measurement_sequence(
    ue_id: str,
    start_beam: int,
    start_rsrp: float,
    rsrp_trend: float,
    count: int,
    interval_ms: float = 100.0,
    start_time_ms: float = 1000.0,
) -> List[BeamMeasurement]:
    """
    Generate a sequence of beam measurements.

    Args:
        ue_id: UE identifier
        start_beam: Starting beam ID
        start_rsrp: Starting RSRP value
        rsrp_trend: RSRP change per measurement (negative = degrading)
        count: Number of measurements
        interval_ms: Time between measurements
        start_time_ms: Starting timestamp

    Returns:
        List of BeamMeasurement objects
    """
    measurements = []

    for i in range(count):
        meas = BeamMeasurement(
            timestamp_ms=start_time_ms + i * interval_ms,
            ue_id=ue_id,
            serving_beam_id=start_beam,
            serving_rsrp_dbm=start_rsrp + i * rsrp_trend,
            neighbor_beams={
                start_beam - 1: start_rsrp - 3.0 + i * rsrp_trend * 0.5,
                start_beam + 1: start_rsrp - 2.5 + i * rsrp_trend * 1.5,
            },
        )
        measurements.append(meas)

    return measurements


@pytest.fixture
def measurement_sequence_stable():
    """Generate stable measurement sequence (no RSRP change)."""
    return generate_measurement_sequence(
        ue_id="uav-stable",
        start_beam=42,
        start_rsrp=-85.0,
        rsrp_trend=0.0,
        count=20,
    )


@pytest.fixture
def measurement_sequence_degrading():
    """Generate degrading measurement sequence."""
    return generate_measurement_sequence(
        ue_id="uav-degrade",
        start_beam=42,
        start_rsrp=-85.0,
        rsrp_trend=-1.0,
        count=20,
    )


# ==============================================================================
# Flask Test Client Fixtures
# ==============================================================================

@pytest.fixture
def flask_app():
    """Create Flask test application."""
    from uav_beam.server import create_app

    app = create_app()
    app.config['TESTING'] = True
    return app


@pytest.fixture
def client(flask_app):
    """Create Flask test client."""
    return flask_app.test_client()


# ==============================================================================
# Performance Test Helpers
# ==============================================================================

@dataclass
class PerformanceResult:
    """Performance test result."""
    name: str
    iterations: int
    total_time_ms: float
    avg_time_ms: float
    min_time_ms: float
    max_time_ms: float
    std_time_ms: float


class PerformanceTimer:
    """Helper for performance measurements."""

    def __init__(self):
        self.times = []

    def record(self, time_ms: float):
        """Record a measurement."""
        self.times.append(time_ms)

    def result(self, name: str) -> PerformanceResult:
        """Get performance result."""
        times = np.array(self.times)
        return PerformanceResult(
            name=name,
            iterations=len(times),
            total_time_ms=float(np.sum(times)),
            avg_time_ms=float(np.mean(times)),
            min_time_ms=float(np.min(times)),
            max_time_ms=float(np.max(times)),
            std_time_ms=float(np.std(times)),
        )


@pytest.fixture
def performance_timer() -> PerformanceTimer:
    """Provide performance timer."""
    return PerformanceTimer()


# ==============================================================================
# Mock Objects
# ==============================================================================

class MockE2Interface:
    """Mock E2 interface for integration tests."""

    def __init__(self):
        self.indications_sent = []
        self.controls_received = []

    def send_indication(self, indication: Dict):
        """Record sent indication."""
        self.indications_sent.append(indication)

    def receive_control(self, control: Dict):
        """Record received control."""
        self.controls_received.append(control)

    def reset(self):
        """Reset recorded data."""
        self.indications_sent.clear()
        self.controls_received.clear()


@pytest.fixture
def mock_e2_interface() -> MockE2Interface:
    """Provide mock E2 interface."""
    return MockE2Interface()


class MockSDL:
    """Mock SDL (Shared Data Layer) for xApp tests."""

    def __init__(self):
        self.data: Dict[str, Dict] = {}

    def set(self, namespace: str, key: str, value: Dict):
        """Set data in SDL."""
        if namespace not in self.data:
            self.data[namespace] = {}
        self.data[namespace][key] = value

    def get(self, namespace: str, key: str) -> Dict:
        """Get data from SDL."""
        return self.data.get(namespace, {}).get(key, {})

    def delete(self, namespace: str, key: str):
        """Delete data from SDL."""
        if namespace in self.data and key in self.data[namespace]:
            del self.data[namespace][key]


@pytest.fixture
def mock_sdl() -> MockSDL:
    """Provide mock SDL."""
    return MockSDL()


# ==============================================================================
# Pytest Configuration
# ==============================================================================

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "performance: marks tests as performance benchmarks"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection based on markers."""
    if config.getoption("-m"):
        # If marker specified, use default behavior
        return

    # Add skip marker to slow tests by default
    skip_slow = pytest.mark.skip(reason="slow test - use -m slow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
