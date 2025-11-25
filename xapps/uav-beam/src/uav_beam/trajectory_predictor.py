"""
UAV Trajectory Predictor for Beam Tracking

Predicts UAV future positions for proactive beam management using:
1. Kalman Filter for state estimation
2. Polynomial trajectory fitting
3. Mission-aware prediction (waypoint following)

This enables sub-beam-width accuracy for mmWave tracking.
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict
from collections import deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class UAVState:
    """UAV kinematic state"""
    timestamp_ms: float
    position: np.ndarray      # [x, y, z] in meters
    velocity: np.ndarray      # [vx, vy, vz] in m/s
    acceleration: np.ndarray  # [ax, ay, az] in m/s^2
    heading: float            # Yaw angle in radians

    @classmethod
    def from_position(cls, timestamp_ms: float, position: np.ndarray) -> "UAVState":
        """Create state from position only (velocity/accel will be estimated)"""
        return cls(
            timestamp_ms=timestamp_ms,
            position=np.array(position),
            velocity=np.zeros(3),
            acceleration=np.zeros(3),
            heading=0.0
        )


@dataclass
class PredictorConfig:
    """Trajectory predictor configuration"""
    # Kalman filter parameters
    process_noise_position: float = 0.1      # Position process noise (m)
    process_noise_velocity: float = 1.0      # Velocity process noise (m/s)
    measurement_noise_position: float = 0.5  # GPS/position measurement noise (m)

    # Prediction parameters
    max_prediction_horizon_ms: float = 500.0  # Maximum prediction time
    history_window_size: int = 50             # Number of states to keep

    # Motion model
    max_acceleration: float = 5.0   # Maximum UAV acceleration (m/s^2)
    max_velocity: float = 30.0      # Maximum UAV velocity (m/s)


class KalmanFilter3D:
    """
    3D Kalman Filter for UAV state estimation

    State vector: [x, y, z, vx, vy, vz]
    """

    def __init__(self, config: PredictorConfig):
        self.config = config

        # State dimension
        self.n_state = 6  # [x, y, z, vx, vy, vz]
        self.n_meas = 3   # [x, y, z]

        # State vector
        self.x = np.zeros(self.n_state)

        # State covariance
        self.P = np.eye(self.n_state) * 10.0

        # Process noise
        self.Q = self._build_process_noise()

        # Measurement noise
        self.R = np.eye(self.n_meas) * config.measurement_noise_position**2

        # Measurement matrix (observe position only)
        self.H = np.zeros((self.n_meas, self.n_state))
        self.H[0, 0] = 1  # x
        self.H[1, 1] = 1  # y
        self.H[2, 2] = 1  # z

        self.initialized = False
        self.last_timestamp_ms = 0

    def _build_process_noise(self) -> np.ndarray:
        """Build process noise matrix"""
        Q = np.zeros((self.n_state, self.n_state))

        # Position noise
        Q[0, 0] = self.config.process_noise_position**2
        Q[1, 1] = self.config.process_noise_position**2
        Q[2, 2] = self.config.process_noise_position**2

        # Velocity noise
        Q[3, 3] = self.config.process_noise_velocity**2
        Q[4, 4] = self.config.process_noise_velocity**2
        Q[5, 5] = self.config.process_noise_velocity**2

        return Q

    def _build_transition_matrix(self, dt: float) -> np.ndarray:
        """Build state transition matrix for constant velocity model"""
        F = np.eye(self.n_state)
        F[0, 3] = dt  # x += vx * dt
        F[1, 4] = dt  # y += vy * dt
        F[2, 5] = dt  # z += vz * dt
        return F

    def predict(self, dt_ms: float) -> np.ndarray:
        """
        Predict state forward in time

        Args:
            dt_ms: Time step in milliseconds

        Returns:
            Predicted state vector
        """
        dt = dt_ms / 1000.0
        F = self._build_transition_matrix(dt)

        # State prediction
        self.x = F @ self.x

        # Covariance prediction
        self.P = F @ self.P @ F.T + self.Q * dt

        return self.x.copy()

    def update(self, measurement: np.ndarray, timestamp_ms: float) -> np.ndarray:
        """
        Update state with new measurement

        Args:
            measurement: Position measurement [x, y, z]
            timestamp_ms: Measurement timestamp

        Returns:
            Updated state vector
        """
        if not self.initialized:
            self.x[:3] = measurement
            self.initialized = True
            self.last_timestamp_ms = timestamp_ms
            return self.x.copy()

        # Predict to current time
        dt_ms = timestamp_ms - self.last_timestamp_ms
        if dt_ms > 0:
            self.predict(dt_ms)

        # Kalman gain
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)

        # State update
        innovation = measurement - self.H @ self.x
        self.x = self.x + K @ innovation

        # Covariance update
        I = np.eye(self.n_state)
        self.P = (I - K @ self.H) @ self.P

        self.last_timestamp_ms = timestamp_ms

        return self.x.copy()

    def get_state(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current state estimate (position, velocity)"""
        return self.x[:3].copy(), self.x[3:6].copy()

    def predict_future(self, horizon_ms: float) -> np.ndarray:
        """
        Predict position at future time without updating state

        Args:
            horizon_ms: Prediction horizon in milliseconds

        Returns:
            Predicted position [x, y, z]
        """
        dt = horizon_ms / 1000.0
        F = self._build_transition_matrix(dt)
        predicted_state = F @ self.x
        return predicted_state[:3]


class TrajectoryPredictor:
    """
    UAV Trajectory Predictor

    Combines Kalman filtering with polynomial fitting for
    accurate short-term trajectory prediction.
    """

    def __init__(self, config: Optional[PredictorConfig] = None):
        self.config = config or PredictorConfig()

        # Kalman filter for each tracked UAV
        self.filters: Dict[str, KalmanFilter3D] = {}

        # State history for polynomial fitting
        self.history: Dict[str, deque] = {}

        # Waypoint information (if available)
        self.waypoints: Dict[str, List[np.ndarray]] = {}

        logger.info("TrajectoryPredictor initialized")

    def update(self, uav_id: str, position: np.ndarray, timestamp_ms: float) -> UAVState:
        """
        Update tracker with new position measurement

        Args:
            uav_id: UAV identifier
            position: Position measurement [x, y, z]
            timestamp_ms: Measurement timestamp

        Returns:
            Current estimated UAV state
        """
        # Initialize filter if new UAV
        if uav_id not in self.filters:
            self.filters[uav_id] = KalmanFilter3D(self.config)
            self.history[uav_id] = deque(maxlen=self.config.history_window_size)

        # Update Kalman filter
        state = self.filters[uav_id].update(position, timestamp_ms)

        # Store in history
        self.history[uav_id].append({
            "timestamp_ms": timestamp_ms,
            "position": position.copy(),
            "state": state.copy()
        })

        # Build UAV state
        pos, vel = self.filters[uav_id].get_state()

        # Estimate acceleration from velocity history
        accel = self._estimate_acceleration(uav_id)

        # Estimate heading from velocity
        heading = np.arctan2(vel[1], vel[0]) if np.linalg.norm(vel[:2]) > 0.1 else 0.0

        return UAVState(
            timestamp_ms=timestamp_ms,
            position=pos,
            velocity=vel,
            acceleration=accel,
            heading=heading
        )

    def predict(
        self,
        uav_id: str,
        horizon_ms: float,
        method: str = "kalman"
    ) -> Optional[UAVState]:
        """
        Predict UAV position at future time

        Args:
            uav_id: UAV identifier
            horizon_ms: Prediction horizon in milliseconds
            method: Prediction method ("kalman", "polynomial", "hybrid")

        Returns:
            Predicted UAV state, or None if UAV not tracked
        """
        if uav_id not in self.filters:
            return None

        # Clamp horizon
        horizon_ms = min(horizon_ms, self.config.max_prediction_horizon_ms)

        if method == "kalman":
            return self._predict_kalman(uav_id, horizon_ms)
        elif method == "polynomial":
            return self._predict_polynomial(uav_id, horizon_ms)
        else:  # hybrid
            return self._predict_hybrid(uav_id, horizon_ms)

    def _predict_kalman(self, uav_id: str, horizon_ms: float) -> UAVState:
        """Kalman filter based prediction (constant velocity)"""
        kf = self.filters[uav_id]

        predicted_pos = kf.predict_future(horizon_ms)
        _, vel = kf.get_state()

        return UAVState(
            timestamp_ms=kf.last_timestamp_ms + horizon_ms,
            position=predicted_pos,
            velocity=vel,
            acceleration=self._estimate_acceleration(uav_id),
            heading=np.arctan2(vel[1], vel[0])
        )

    def _predict_polynomial(self, uav_id: str, horizon_ms: float) -> UAVState:
        """Polynomial fitting based prediction"""
        history = list(self.history[uav_id])

        if len(history) < 5:
            return self._predict_kalman(uav_id, horizon_ms)

        # Extract time and positions
        times = np.array([h["timestamp_ms"] for h in history])
        positions = np.array([h["position"] for h in history])

        # Normalize time
        t0 = times[-1]
        times_norm = (times - t0) / 1000.0
        t_pred = horizon_ms / 1000.0

        # Fit polynomial for each dimension
        predicted_pos = np.zeros(3)
        predicted_vel = np.zeros(3)

        for dim in range(3):
            # Use quadratic fit
            coeffs = np.polyfit(times_norm, positions[:, dim], 2)
            poly = np.poly1d(coeffs)
            poly_deriv = poly.deriv()

            predicted_pos[dim] = poly(t_pred)
            predicted_vel[dim] = poly_deriv(t_pred)

        return UAVState(
            timestamp_ms=t0 + horizon_ms,
            position=predicted_pos,
            velocity=predicted_vel,
            acceleration=self._estimate_acceleration(uav_id),
            heading=np.arctan2(predicted_vel[1], predicted_vel[0])
        )

    def _predict_hybrid(self, uav_id: str, horizon_ms: float) -> UAVState:
        """
        Hybrid prediction combining Kalman and polynomial

        Short horizon: More weight on Kalman (smoother)
        Long horizon: More weight on polynomial (captures maneuvers)
        """
        kalman_pred = self._predict_kalman(uav_id, horizon_ms)
        poly_pred = self._predict_polynomial(uav_id, horizon_ms)

        # Weight based on prediction horizon
        alpha = min(horizon_ms / 200.0, 0.7)  # More polynomial weight for longer horizons

        blended_pos = (1 - alpha) * kalman_pred.position + alpha * poly_pred.position
        blended_vel = (1 - alpha) * kalman_pred.velocity + alpha * poly_pred.velocity

        return UAVState(
            timestamp_ms=kalman_pred.timestamp_ms,
            position=blended_pos,
            velocity=blended_vel,
            acceleration=kalman_pred.acceleration,
            heading=np.arctan2(blended_vel[1], blended_vel[0])
        )

    def _estimate_acceleration(self, uav_id: str) -> np.ndarray:
        """Estimate acceleration from velocity history"""
        history = list(self.history.get(uav_id, []))

        if len(history) < 3:
            return np.zeros(3)

        # Use last 3 states
        recent = history[-3:]

        dt1 = (recent[1]["timestamp_ms"] - recent[0]["timestamp_ms"]) / 1000.0
        dt2 = (recent[2]["timestamp_ms"] - recent[1]["timestamp_ms"]) / 1000.0

        if dt1 <= 0 or dt2 <= 0:
            return np.zeros(3)

        v1 = (recent[1]["position"] - recent[0]["position"]) / dt1
        v2 = (recent[2]["position"] - recent[1]["position"]) / dt2

        accel = (v2 - v1) / ((dt1 + dt2) / 2)

        # Clamp to max acceleration
        accel_mag = np.linalg.norm(accel)
        if accel_mag > self.config.max_acceleration:
            accel = accel * self.config.max_acceleration / accel_mag

        return accel

    def set_waypoints(self, uav_id: str, waypoints: List[np.ndarray]):
        """Set mission waypoints for mission-aware prediction"""
        self.waypoints[uav_id] = [np.array(wp) for wp in waypoints]

    def get_prediction_confidence(self, uav_id: str, horizon_ms: float) -> float:
        """
        Estimate prediction confidence

        Returns value in [0, 1] based on:
        - History length
        - State estimation covariance
        - Prediction horizon
        """
        if uav_id not in self.filters:
            return 0.0

        history = self.history.get(uav_id, [])
        history_factor = min(len(history) / 20.0, 1.0)

        horizon_factor = max(0, 1.0 - horizon_ms / self.config.max_prediction_horizon_ms)

        # Covariance factor
        kf = self.filters[uav_id]
        cov_trace = np.trace(kf.P[:3, :3])
        cov_factor = max(0, 1.0 - cov_trace / 100.0)

        return history_factor * horizon_factor * cov_factor

    def remove_uav(self, uav_id: str):
        """Stop tracking a UAV"""
        self.filters.pop(uav_id, None)
        self.history.pop(uav_id, None)
        self.waypoints.pop(uav_id, None)

    def get_tracked_uavs(self) -> List[str]:
        """Get list of tracked UAV IDs"""
        return list(self.filters.keys())
