"""
UAV Beam Tracking xApp for O-RAN Near-RT RIC

This xApp provides mmWave beam management for UAV communication per 3GPP TS 38.214:
- P1: Initial beam acquisition (SSB beam sweeping)
- P2: Beam refinement (CSI-RS based)
- P3: Beam tracking and recovery with BFD/BFR
- AoA/AoD angle estimation (2D MUSIC, ESPRIT, Unitary-ESPRIT)
- Proactive beam switching with trajectory prediction
- E2 socket communication with ns-3 simulator

Target: 5G NR FR2 (mmWave) beamforming systems

References:
- 3GPP TS 38.214: Physical layer procedures for data
- 3GPP TS 38.321: MAC protocol specification (Beam Failure Recovery)
- 3GPP TS 38.213: Physical layer procedures for control
"""

__version__ = "0.2.0"
__author__ = "UAV O-RAN Research Team"

from .beam_tracker import (
    BeamTracker,
    BeamConfig,
    BeamMeasurement,
    BeamDecision,
    BeamState,
    BeamProcedure,
    UEBeamState,
)
from .trajectory_predictor import TrajectoryPredictor, PredictorConfig
from .angle_estimator import AngleEstimator, AngleEstimatorConfig, EstimationMethod, AngleEstimate
from .beam_codebook import BeamCodebook, CodebookConfig, BeamPattern, CodebookType
from .e2_messages import (
    BeamIndication,
    BeamControl,
    BeamAction,
    SubscriptionRequest,
    SubscriptionResponse,
    E2MessageType,
    E2MessageCodec,
)
from .e2_client import (
    E2Client,
    E2Connection,
    E2ClientConfig,
    E2Metrics,
    ConnectionState,
    create_e2_client,
)
from .server import UAVBeamXApp, create_app

__all__ = [
    # Core beam tracking (3GPP P1/P2/P3)
    "BeamTracker",
    "BeamConfig",
    "BeamMeasurement",
    "BeamDecision",
    "BeamState",
    "BeamProcedure",
    "UEBeamState",
    # Beam codebook
    "BeamCodebook",
    "CodebookConfig",
    "BeamPattern",
    "CodebookType",
    # Trajectory prediction
    "TrajectoryPredictor",
    "PredictorConfig",
    # Angle estimation (MUSIC/ESPRIT)
    "AngleEstimator",
    "AngleEstimatorConfig",
    "EstimationMethod",
    "AngleEstimate",
    # E2 messages
    "BeamIndication",
    "BeamControl",
    "BeamAction",
    "SubscriptionRequest",
    "SubscriptionResponse",
    "E2MessageType",
    "E2MessageCodec",
    # E2 client
    "E2Client",
    "E2Connection",
    "E2ClientConfig",
    "E2Metrics",
    "ConnectionState",
    "create_e2_client",
    # Server
    "UAVBeamXApp",
    "create_app",
]
