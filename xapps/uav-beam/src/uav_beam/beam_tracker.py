"""
Beam Tracker Module for UAV mmWave Communication

Implements complete beam management procedures per 3GPP TS 38.214:
- P1: Initial beam acquisition (SSB beam sweeping)
- P2: Beam refinement (CSI-RS based)
- P3: Beam tracking and recovery

Beam Failure Detection and Recovery per 3GPP TS 38.321 Section 5.17:
- Beam Failure Detection (BFD) using L1-RSRP monitoring
- Beam Failure Recovery (BFR) request procedure
- Candidate beam identification

References:
- 3GPP TS 38.214: Physical layer procedures for data
- 3GPP TS 38.321: MAC protocol specification
- 3GPP TS 38.213: Physical layer procedures for control
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, Callable
from enum import Enum
from collections import deque
import logging
import time

logger = logging.getLogger(__name__)


class BeamState(Enum):
    """Beam link state per 3GPP beam management procedures"""
    IDLE = "idle"                       # No active beam link
    P1_ACQUISITION = "p1_acquisition"   # P1: Initial beam acquisition
    P2_REFINEMENT = "p2_refinement"     # P2: Beam refinement
    P3_TRACKING = "p3_tracking"         # P3: Active tracking
    BFD_DETECTED = "bfd_detected"       # Beam failure detected
    BFR_RECOVERY = "bfr_recovery"       # Beam failure recovery in progress
    FAILED = "failed"                   # Recovery failed


class BeamProcedure(Enum):
    """3GPP Beam Management Procedures"""
    P1 = "p1"  # Initial acquisition using SSB
    P2 = "p2"  # Refinement using CSI-RS
    P3 = "p3"  # Tracking and maintenance


@dataclass
class BeamConfig:
    """mmWave beam configuration parameters per 3GPP TS 38.214"""
    # Antenna array configuration
    num_antenna_elements_h: int = 8      # Horizontal elements (N1)
    num_antenna_elements_v: int = 8      # Vertical elements (N2)

    # Codebook configuration per 3GPP TS 38.214 Table 5.2.2.2.1-2
    num_beams_h: int = 16                # O1: Horizontal oversampling
    num_beams_v: int = 8                 # O2: Vertical oversampling

    # SSB configuration (P1 procedure)
    num_ssb_beams: int = 8               # Number of SSB beams (max 64 for FR2)
    ssb_periodicity_ms: float = 20.0     # SSB burst periodicity (5, 10, 20, 40, 80, 160 ms)
    ssb_burst_duration_ms: float = 5.0   # Duration of SSB burst

    # CSI-RS configuration (P2 procedure)
    num_csirs_beams: int = 8             # CSI-RS beams for refinement
    csi_rs_periodicity_ms: float = 10.0  # CSI-RS periodicity
    csi_rs_slots_per_frame: int = 4      # CSI-RS resources per frame

    # Beam Failure Detection (BFD) per 3GPP TS 38.321
    bfd_rsrp_threshold_dbm: float = -110.0  # Q_out threshold
    bfd_timer_ms: float = 10.0              # beamFailureDetectionTimer
    max_bfd_count: int = 2                  # beamFailureInstanceMaxCount

    # Beam Failure Recovery (BFR) per 3GPP TS 38.321 Section 5.17
    bfr_rsrp_threshold_dbm: float = -100.0  # Q_in threshold for candidate beams
    bfr_timer_ms: float = 50.0              # beamFailureRecoveryTimer
    bfr_max_attempts: int = 4               # Max PRACH attempts for BFR

    # L1-RSRP monitoring
    rsrp_filter_coefficient: float = 0.5    # L1 filter coefficient (K0 per 38.214)

    # Tracking parameters (P3 procedure)
    tracking_update_interval_ms: float = 5.0
    prediction_horizon_ms: float = 20.0
    beam_switch_hysteresis_db: float = 3.0  # Hysteresis for beam switching

    # Handover margins
    a3_offset_db: float = 2.0              # A3 event offset
    time_to_trigger_ms: float = 40.0       # Time to trigger for A3

    @property
    def total_beams(self) -> int:
        return self.num_beams_h * self.num_beams_v

    @property
    def total_antenna_elements(self) -> int:
        return self.num_antenna_elements_h * self.num_antenna_elements_v


@dataclass
class BeamMeasurement:
    """
    Beam measurement report from UE per 3GPP TS 38.215

    Contains L1-RSRP measurements for serving and neighbor beams.
    """
    timestamp_ms: float
    ue_id: str
    cell_id: int = 0

    # Serving beam measurements
    serving_beam_id: int = 0
    serving_rsrp_dbm: float = -100.0        # SS-RSRP or CSI-RSRP
    serving_rsrq_db: float = -15.0          # Reference signal received quality
    serving_sinr_db: float = 0.0            # SINR

    # Neighbor beam measurements (beam_id -> L1-RSRP)
    neighbor_beams: Dict[int, float] = field(default_factory=dict)

    # CQI report
    cqi: int = 7                            # Wideband CQI (0-15)
    ri: int = 1                             # Rank indicator

    # Position/velocity info (if available from UAV)
    position: Optional[Tuple[float, float, float]] = None  # (x, y, z) meters
    velocity: Optional[Tuple[float, float, float]] = None  # (vx, vy, vz) m/s

    # SSB index for P1 measurements
    ssb_index: Optional[int] = None

    # CSI-RS resource index for P2 measurements
    csi_rs_index: Optional[int] = None


@dataclass
class BeamDecision:
    """Beam control decision output"""
    ue_id: str
    timestamp_ms: float
    procedure: BeamProcedure
    action: str                             # "maintain", "switch", "refine", "recover", "fail"
    current_beam_id: int
    target_beam_id: int
    confidence: float                       # Decision confidence [0, 1]
    predicted_rsrp_dbm: float
    reason: str
    candidate_beams: List[int] = field(default_factory=list)  # Alternative beams


@dataclass
class UEBeamState:
    """Per-UE beam state tracking"""
    ue_id: str
    state: BeamState = BeamState.IDLE
    current_beam_id: int = 0
    best_ssb_beam_id: int = 0              # Best beam from P1

    # L1-RSRP tracking (filtered)
    filtered_rsrp_dbm: float = -100.0
    rsrp_history: deque = field(default_factory=lambda: deque(maxlen=20))

    # BFD state per 3GPP TS 38.321
    bfd_count: int = 0                      # Beam failure instance counter
    bfd_timer_start_ms: float = 0.0
    last_good_beam_id: int = 0

    # BFR state
    bfr_attempts: int = 0
    bfr_timer_start_ms: float = 0.0
    candidate_beam_id: int = 0

    # P2 refinement state
    p2_beam_candidates: List[int] = field(default_factory=list)
    p2_measurements: Dict[int, float] = field(default_factory=dict)

    # Tracking state
    beam_switch_pending: bool = False
    ttt_timer_start_ms: float = 0.0         # Time to trigger start

    # Statistics
    created_at_ms: float = 0.0
    last_update_ms: float = 0.0
    num_beam_switches: int = 0
    num_beam_failures: int = 0


class BeamTracker:
    """
    mmWave Beam Tracker for UAV Communication

    Implements complete 3GPP beam management:

    P1 (Initial Beam Acquisition):
    - SSB beam sweeping simulation
    - Complete codebook search
    - Best beam selection based on L1-RSRP

    P2 (Beam Refinement):
    - CSI-RS based refinement within SSB beam coverage
    - Narrow beam search
    - Beam pair link establishment

    P3 (Beam Tracking/Recovery):
    - Continuous beam tracking with prediction
    - Beam Failure Detection (BFD)
    - Beam Failure Recovery (BFR)
    - L1-RSRP monitoring and filtering
    """

    def __init__(
        self,
        config: Optional[BeamConfig] = None,
        codebook: Optional['BeamCodebook'] = None
    ):
        self.config = config or BeamConfig()
        self.codebook = codebook

        # Per-UE state tracking
        self.ue_states: Dict[str, UEBeamState] = {}

        # Measurement history
        self.measurement_history: Dict[str, List[BeamMeasurement]] = {}

        # Beam codebook (steering vectors)
        if self.codebook is None:
            self._codebook_matrix = self._generate_dft_codebook()
        else:
            self._codebook_matrix = None

        # SSB beam configuration
        self.ssb_beam_set = self._configure_ssb_beams()

        # Callbacks for external integration
        self._on_beam_switch: Optional[Callable] = None
        self._on_beam_failure: Optional[Callable] = None

        # Statistics
        self.stats = {
            "p1_acquisitions": 0,
            "p2_refinements": 0,
            "p3_switches": 0,
            "beam_failures": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "prediction_accuracy": [],
        }

        logger.info(
            f"BeamTracker initialized: {self.config.total_beams} beams, "
            f"{self.config.num_ssb_beams} SSB beams"
        )

    # =========================================================================
    # Initialization and Configuration
    # =========================================================================

    def _generate_dft_codebook(self) -> np.ndarray:
        """
        Generate DFT-based beam codebook for UPA

        Per 3GPP TS 38.214, Type I single-panel codebook uses
        DFT-based beam directions.
        """
        num_beams = self.config.total_beams
        num_elements = self.config.total_antenna_elements

        codebook = np.zeros((num_beams, num_elements), dtype=complex)

        N1 = self.config.num_antenna_elements_h
        N2 = self.config.num_antenna_elements_v
        O1 = self.config.num_beams_h
        O2 = self.config.num_beams_v

        beam_idx = 0
        for i1 in range(O1):
            for i2 in range(O2):
                # DFT beam direction
                u = (2 * i1 / O1 - 1)  # Normalized spatial frequency [-1, 1]
                v = (2 * i2 / O2 - 1) * 0.5  # Narrower elevation range

                elem_idx = 0
                for m in range(N1):
                    for n in range(N2):
                        phase = np.pi * (m * u + n * v)
                        codebook[beam_idx, elem_idx] = np.exp(1j * phase)
                        elem_idx += 1

                # Normalize
                codebook[beam_idx, :] /= np.linalg.norm(codebook[beam_idx, :])
                beam_idx += 1

        return codebook

    def _configure_ssb_beams(self) -> List[int]:
        """
        Configure SSB beam set for P1 procedure

        SSB beams are wide beams that cover the entire sector.
        Typically 4-64 beams for FR2 per 3GPP.
        """
        total_beams = self.config.total_beams
        num_ssb = self.config.num_ssb_beams

        # Select evenly distributed beams
        if num_ssb >= total_beams:
            return list(range(total_beams))

        step = total_beams // num_ssb
        return [i * step for i in range(num_ssb)]

    def register_callbacks(
        self,
        on_beam_switch: Optional[Callable] = None,
        on_beam_failure: Optional[Callable] = None
    ):
        """Register callback functions for beam events"""
        self._on_beam_switch = on_beam_switch
        self._on_beam_failure = on_beam_failure

    # =========================================================================
    # P1: Initial Beam Acquisition (SSB Beam Sweeping)
    # =========================================================================

    def start_p1_acquisition(self, ue_id: str, timestamp_ms: float) -> List[int]:
        """
        Start P1 beam acquisition procedure

        Returns list of SSB beam IDs to sweep.

        Args:
            ue_id: UE identifier
            timestamp_ms: Current timestamp

        Returns:
            List of SSB beam IDs for sweeping
        """
        # Initialize UE state
        if ue_id not in self.ue_states:
            self.ue_states[ue_id] = UEBeamState(
                ue_id=ue_id,
                created_at_ms=timestamp_ms
            )

        state = self.ue_states[ue_id]
        state.state = BeamState.P1_ACQUISITION
        state.last_update_ms = timestamp_ms

        self.stats["p1_acquisitions"] += 1

        logger.info(f"P1 acquisition started for UE {ue_id}: sweeping {len(self.ssb_beam_set)} SSB beams")

        return self.ssb_beam_set.copy()

    def process_p1_measurement(
        self,
        ue_id: str,
        ssb_beam_id: int,
        rsrp_dbm: float,
        timestamp_ms: float
    ) -> Optional[BeamDecision]:
        """
        Process single SSB beam measurement during P1

        Args:
            ue_id: UE identifier
            ssb_beam_id: SSB beam index
            rsrp_dbm: Measured L1-RSRP
            timestamp_ms: Measurement timestamp

        Returns:
            BeamDecision if P1 is complete, None otherwise
        """
        if ue_id not in self.ue_states:
            self.start_p1_acquisition(ue_id, timestamp_ms)

        state = self.ue_states[ue_id]

        # Store measurement
        state.p2_measurements[ssb_beam_id] = rsrp_dbm

        # Check if all SSB beams have been measured
        measured_beams = set(state.p2_measurements.keys())
        if not measured_beams.issuperset(self.ssb_beam_set):
            return None  # More measurements needed

        # Find best SSB beam
        best_beam_id = max(state.p2_measurements.keys(),
                         key=lambda b: state.p2_measurements[b])
        best_rsrp = state.p2_measurements[best_beam_id]

        state.best_ssb_beam_id = best_beam_id
        state.current_beam_id = best_beam_id
        state.filtered_rsrp_dbm = best_rsrp
        state.state = BeamState.P2_REFINEMENT

        logger.info(
            f"P1 complete for UE {ue_id}: best SSB beam={best_beam_id}, "
            f"RSRP={best_rsrp:.1f} dBm"
        )

        return BeamDecision(
            ue_id=ue_id,
            timestamp_ms=timestamp_ms,
            procedure=BeamProcedure.P1,
            action="acquire",
            current_beam_id=0,
            target_beam_id=best_beam_id,
            confidence=0.8,
            predicted_rsrp_dbm=best_rsrp,
            reason=f"P1 acquisition complete: selected SSB beam {best_beam_id}",
            candidate_beams=sorted(
                state.p2_measurements.keys(),
                key=lambda b: state.p2_measurements[b],
                reverse=True
            )[:4]
        )

    def complete_p1_sweep(
        self,
        ue_id: str,
        ssb_measurements: Dict[int, float],
        timestamp_ms: float
    ) -> BeamDecision:
        """
        Complete P1 procedure with all SSB measurements

        Args:
            ue_id: UE identifier
            ssb_measurements: Dict of ssb_beam_id -> rsrp_dbm
            timestamp_ms: Measurement timestamp

        Returns:
            BeamDecision with selected SSB beam
        """
        if ue_id not in self.ue_states:
            self.start_p1_acquisition(ue_id, timestamp_ms)

        state = self.ue_states[ue_id]

        # Find best beam
        if not ssb_measurements:
            return BeamDecision(
                ue_id=ue_id,
                timestamp_ms=timestamp_ms,
                procedure=BeamProcedure.P1,
                action="fail",
                current_beam_id=0,
                target_beam_id=0,
                confidence=0.0,
                predicted_rsrp_dbm=-120.0,
                reason="P1 failed: no measurements received"
            )

        best_beam_id = max(ssb_measurements.keys(),
                          key=lambda b: ssb_measurements[b])
        best_rsrp = ssb_measurements[best_beam_id]

        # Update state
        state.best_ssb_beam_id = best_beam_id
        state.current_beam_id = best_beam_id
        state.filtered_rsrp_dbm = best_rsrp
        state.p2_measurements = ssb_measurements.copy()

        # Determine next state based on RSRP
        if best_rsrp >= self.config.bfd_rsrp_threshold_dbm:
            state.state = BeamState.P2_REFINEMENT
            action = "acquire"
        else:
            state.state = BeamState.FAILED
            action = "fail"

        self.stats["p1_acquisitions"] += 1

        logger.info(
            f"P1 sweep complete for UE {ue_id}: beam={best_beam_id}, "
            f"RSRP={best_rsrp:.1f} dBm"
        )

        return BeamDecision(
            ue_id=ue_id,
            timestamp_ms=timestamp_ms,
            procedure=BeamProcedure.P1,
            action=action,
            current_beam_id=0,
            target_beam_id=best_beam_id,
            confidence=min(1.0, (best_rsrp + 120) / 30),  # Normalize to [0,1]
            predicted_rsrp_dbm=best_rsrp,
            reason=f"P1 complete: selected beam {best_beam_id}",
            candidate_beams=sorted(
                ssb_measurements.keys(),
                key=lambda b: ssb_measurements[b],
                reverse=True
            )[:4]
        )

    # =========================================================================
    # P2: Beam Refinement (CSI-RS Based)
    # =========================================================================

    def start_p2_refinement(
        self,
        ue_id: str,
        timestamp_ms: float
    ) -> List[int]:
        """
        Start P2 beam refinement procedure

        Returns list of CSI-RS beam IDs to measure within SSB beam coverage.

        Args:
            ue_id: UE identifier
            timestamp_ms: Current timestamp

        Returns:
            List of narrow beam IDs for refinement
        """
        if ue_id not in self.ue_states:
            return []

        state = self.ue_states[ue_id]
        state.state = BeamState.P2_REFINEMENT

        # Get narrow beams within SSB beam coverage
        ssb_beam = state.best_ssb_beam_id
        csirs_beams = self._get_refinement_beams(ssb_beam)

        state.p2_beam_candidates = csirs_beams
        state.p2_measurements.clear()

        self.stats["p2_refinements"] += 1

        logger.info(
            f"P2 refinement started for UE {ue_id}: "
            f"{len(csirs_beams)} CSI-RS beams near SSB beam {ssb_beam}"
        )

        return csirs_beams

    def _get_refinement_beams(self, ssb_beam_id: int) -> List[int]:
        """
        Get narrow beams for P2 refinement within SSB beam coverage

        Maps SSB beam to set of narrow beams in fine codebook.
        """
        if self.codebook is not None:
            return self.codebook.get_csirs_beam_set(
                ssb_beam_id,
                self.config.num_csirs_beams
            )

        # Default: get neighboring beams in fine codebook
        i_h = ssb_beam_id // self.config.num_beams_v
        i_v = ssb_beam_id % self.config.num_beams_v

        candidates = []
        search_range = 2

        for di in range(-search_range, search_range + 1):
            for dj in range(-search_range, search_range + 1):
                ni = i_h + di
                nj = i_v + dj

                if (0 <= ni < self.config.num_beams_h and
                    0 <= nj < self.config.num_beams_v):
                    candidates.append(ni * self.config.num_beams_v + nj)

        return candidates[:self.config.num_csirs_beams]

    def process_p2_measurement(
        self,
        ue_id: str,
        beam_id: int,
        rsrp_dbm: float,
        timestamp_ms: float
    ) -> Optional[BeamDecision]:
        """
        Process CSI-RS beam measurement during P2

        Args:
            ue_id: UE identifier
            beam_id: Narrow beam index
            rsrp_dbm: Measured L1-RSRP
            timestamp_ms: Measurement timestamp

        Returns:
            BeamDecision if P2 is complete, None otherwise
        """
        if ue_id not in self.ue_states:
            return None

        state = self.ue_states[ue_id]
        state.p2_measurements[beam_id] = rsrp_dbm

        # Check if all candidate beams measured
        if len(state.p2_measurements) < len(state.p2_beam_candidates):
            return None

        # Find best refined beam
        best_beam_id = max(state.p2_measurements.keys(),
                          key=lambda b: state.p2_measurements[b])
        best_rsrp = state.p2_measurements[best_beam_id]

        # Update state
        state.current_beam_id = best_beam_id
        state.filtered_rsrp_dbm = best_rsrp
        state.state = BeamState.P3_TRACKING
        state.last_good_beam_id = best_beam_id

        logger.info(
            f"P2 refinement complete for UE {ue_id}: "
            f"beam={best_beam_id}, RSRP={best_rsrp:.1f} dBm"
        )

        return BeamDecision(
            ue_id=ue_id,
            timestamp_ms=timestamp_ms,
            procedure=BeamProcedure.P2,
            action="refine",
            current_beam_id=state.best_ssb_beam_id,
            target_beam_id=best_beam_id,
            confidence=0.9,
            predicted_rsrp_dbm=best_rsrp,
            reason=f"P2 refinement complete: refined to beam {best_beam_id}",
            candidate_beams=sorted(
                state.p2_measurements.keys(),
                key=lambda b: state.p2_measurements[b],
                reverse=True
            )[:4]
        )

    def complete_p2_refinement(
        self,
        ue_id: str,
        csirs_measurements: Dict[int, float],
        timestamp_ms: float
    ) -> BeamDecision:
        """
        Complete P2 procedure with all CSI-RS measurements

        Args:
            ue_id: UE identifier
            csirs_measurements: Dict of beam_id -> rsrp_dbm
            timestamp_ms: Measurement timestamp

        Returns:
            BeamDecision with refined beam
        """
        if ue_id not in self.ue_states:
            return BeamDecision(
                ue_id=ue_id,
                timestamp_ms=timestamp_ms,
                procedure=BeamProcedure.P2,
                action="fail",
                current_beam_id=0,
                target_beam_id=0,
                confidence=0.0,
                predicted_rsrp_dbm=-120.0,
                reason="P2 failed: UE not found"
            )

        state = self.ue_states[ue_id]

        if not csirs_measurements:
            # Fall back to SSB beam
            return BeamDecision(
                ue_id=ue_id,
                timestamp_ms=timestamp_ms,
                procedure=BeamProcedure.P2,
                action="maintain",
                current_beam_id=state.current_beam_id,
                target_beam_id=state.best_ssb_beam_id,
                confidence=0.5,
                predicted_rsrp_dbm=state.filtered_rsrp_dbm,
                reason="P2 skipped: no CSI-RS measurements, using SSB beam"
            )

        # Find best refined beam
        best_beam_id = max(csirs_measurements.keys(),
                          key=lambda b: csirs_measurements[b])
        best_rsrp = csirs_measurements[best_beam_id]

        # Update state
        old_beam = state.current_beam_id
        state.current_beam_id = best_beam_id
        state.filtered_rsrp_dbm = best_rsrp
        state.state = BeamState.P3_TRACKING
        state.last_good_beam_id = best_beam_id
        state.p2_measurements = csirs_measurements.copy()

        logger.info(
            f"P2 complete for UE {ue_id}: refined from beam {old_beam} to {best_beam_id}, "
            f"RSRP={best_rsrp:.1f} dBm"
        )

        return BeamDecision(
            ue_id=ue_id,
            timestamp_ms=timestamp_ms,
            procedure=BeamProcedure.P2,
            action="refine",
            current_beam_id=old_beam,
            target_beam_id=best_beam_id,
            confidence=0.9,
            predicted_rsrp_dbm=best_rsrp,
            reason=f"P2 refinement complete: {old_beam} -> {best_beam_id}",
            candidate_beams=sorted(
                csirs_measurements.keys(),
                key=lambda b: csirs_measurements[b],
                reverse=True
            )[:4]
        )

    # =========================================================================
    # P3: Beam Tracking and Maintenance
    # =========================================================================

    def process_measurement(self, measurement: BeamMeasurement) -> BeamDecision:
        """
        Process beam measurement during P3 tracking

        Main entry point for continuous beam tracking.

        Args:
            measurement: Beam measurement report from UE

        Returns:
            BeamDecision with recommended action
        """
        ue_id = measurement.ue_id
        timestamp_ms = measurement.timestamp_ms

        # Initialize UE state if new
        if ue_id not in self.ue_states:
            self._initialize_ue(ue_id, measurement)

        state = self.ue_states[ue_id]

        # Store measurement history
        if ue_id not in self.measurement_history:
            self.measurement_history[ue_id] = []
        self.measurement_history[ue_id].append(measurement)

        # Limit history size
        if len(self.measurement_history[ue_id]) > 100:
            self.measurement_history[ue_id] = self.measurement_history[ue_id][-100:]

        # Update filtered L1-RSRP per 3GPP TS 38.214
        self._update_filtered_rsrp(state, measurement.serving_rsrp_dbm)

        # Check beam state and execute appropriate procedure
        if state.state == BeamState.BFD_DETECTED:
            return self._execute_bfr(measurement)
        elif state.state == BeamState.BFR_RECOVERY:
            return self._continue_bfr(measurement)
        elif state.state in [BeamState.P1_ACQUISITION, BeamState.P2_REFINEMENT]:
            # Still in acquisition phase
            return BeamDecision(
                ue_id=ue_id,
                timestamp_ms=timestamp_ms,
                procedure=BeamProcedure.P1 if state.state == BeamState.P1_ACQUISITION else BeamProcedure.P2,
                action="continue",
                current_beam_id=state.current_beam_id,
                target_beam_id=state.current_beam_id,
                confidence=0.5,
                predicted_rsrp_dbm=measurement.serving_rsrp_dbm,
                reason=f"Acquisition in progress: {state.state.value}"
            )

        # P3: Active tracking
        # Check for Beam Failure Detection (BFD)
        if self._check_bfd(state, measurement):
            return self._handle_beam_failure(measurement)

        # Predict optimal beam
        predicted_beam, confidence = self._predict_optimal_beam(measurement)

        # Check if beam switch is beneficial
        if self._should_switch_beam(state, measurement, predicted_beam, confidence):
            return self._execute_beam_switch(state, measurement, predicted_beam, confidence)

        # Maintain current beam
        state.last_update_ms = timestamp_ms
        return BeamDecision(
            ue_id=ue_id,
            timestamp_ms=timestamp_ms,
            procedure=BeamProcedure.P3,
            action="maintain",
            current_beam_id=measurement.serving_beam_id,
            target_beam_id=measurement.serving_beam_id,
            confidence=1.0 - confidence,
            predicted_rsrp_dbm=measurement.serving_rsrp_dbm,
            reason="P3 tracking: current beam optimal"
        )

    def _initialize_ue(self, ue_id: str, measurement: BeamMeasurement):
        """Initialize tracking state for new UE"""
        self.ue_states[ue_id] = UEBeamState(
            ue_id=ue_id,
            state=BeamState.P3_TRACKING,
            current_beam_id=measurement.serving_beam_id,
            best_ssb_beam_id=measurement.serving_beam_id,
            filtered_rsrp_dbm=measurement.serving_rsrp_dbm,
            last_good_beam_id=measurement.serving_beam_id,
            created_at_ms=measurement.timestamp_ms,
            last_update_ms=measurement.timestamp_ms
        )

    def _update_filtered_rsrp(self, state: UEBeamState, rsrp_dbm: float):
        """
        Update L1 filtered RSRP per 3GPP TS 38.214

        F[n] = (1 - a) * F[n-1] + a * L1[n]

        where a = 1/2^K0 is the filter coefficient.
        """
        alpha = self.config.rsrp_filter_coefficient
        state.filtered_rsrp_dbm = (
            (1 - alpha) * state.filtered_rsrp_dbm +
            alpha * rsrp_dbm
        )
        state.rsrp_history.append(rsrp_dbm)

    # =========================================================================
    # Beam Failure Detection (BFD) per 3GPP TS 38.321
    # =========================================================================

    def _check_bfd(self, state: UEBeamState, measurement: BeamMeasurement) -> bool:
        """
        Check for Beam Failure Detection condition

        BFD is triggered when L1-RSRP falls below Q_out threshold
        for beamFailureInstanceMaxCount consecutive instances.
        """
        if state.filtered_rsrp_dbm < self.config.bfd_rsrp_threshold_dbm:
            state.bfd_count += 1

            if state.bfd_count >= self.config.max_bfd_count:
                logger.warning(
                    f"BFD triggered for UE {state.ue_id}: "
                    f"RSRP={state.filtered_rsrp_dbm:.1f} dBm < "
                    f"threshold={self.config.bfd_rsrp_threshold_dbm:.1f} dBm"
                )
                return True
        else:
            # Reset counter on good measurement
            state.bfd_count = max(0, state.bfd_count - 1)

        return False

    def _handle_beam_failure(self, measurement: BeamMeasurement) -> BeamDecision:
        """
        Handle beam failure detection - initiate BFR

        Per 3GPP TS 38.321 Section 5.17:
        1. Identify candidate beams (RSRP >= Q_in)
        2. Start BFR timer
        3. Send PRACH with BFR indication
        """
        ue_id = measurement.ue_id
        state = self.ue_states[ue_id]

        state.state = BeamState.BFD_DETECTED
        state.num_beam_failures += 1
        state.bfr_timer_start_ms = measurement.timestamp_ms
        state.bfr_attempts = 0

        self.stats["beam_failures"] += 1

        if self._on_beam_failure:
            self._on_beam_failure(ue_id, measurement)

        logger.warning(
            f"Beam failure for UE {ue_id}: initiating BFR, "
            f"RSRP={state.filtered_rsrp_dbm:.1f} dBm"
        )

        return self._execute_bfr(measurement)

    def _execute_bfr(self, measurement: BeamMeasurement) -> BeamDecision:
        """
        Execute Beam Failure Recovery procedure

        Per 3GPP TS 38.321 Section 5.17:
        1. Select candidate beam with RSRP >= Q_in
        2. Transmit PRACH for BFR
        """
        ue_id = measurement.ue_id
        state = self.ue_states[ue_id]

        # Find candidate beams (RSRP >= Q_in threshold)
        candidates = self._find_candidate_beams(measurement)

        if candidates:
            # Select best candidate
            best_candidate = candidates[0]
            state.candidate_beam_id = best_candidate
            state.state = BeamState.BFR_RECOVERY
            state.bfr_attempts += 1

            logger.info(
                f"BFR for UE {ue_id}: selected candidate beam {best_candidate}, "
                f"attempt {state.bfr_attempts}"
            )

            return BeamDecision(
                ue_id=ue_id,
                timestamp_ms=measurement.timestamp_ms,
                procedure=BeamProcedure.P3,
                action="recover",
                current_beam_id=measurement.serving_beam_id,
                target_beam_id=best_candidate,
                confidence=0.6,
                predicted_rsrp_dbm=measurement.neighbor_beams.get(
                    best_candidate, self.config.bfr_rsrp_threshold_dbm
                ),
                reason=f"BFR: switching to candidate beam {best_candidate}",
                candidate_beams=candidates
            )
        else:
            # No candidate found - trigger re-acquisition (P1)
            state.state = BeamState.P1_ACQUISITION

            logger.warning(f"BFR for UE {ue_id}: no candidates, triggering P1")

            return BeamDecision(
                ue_id=ue_id,
                timestamp_ms=measurement.timestamp_ms,
                procedure=BeamProcedure.P1,
                action="reacquire",
                current_beam_id=measurement.serving_beam_id,
                target_beam_id=0,
                confidence=0.3,
                predicted_rsrp_dbm=-100.0,
                reason="BFR failed: no candidate beams, initiating P1"
            )

    def _continue_bfr(self, measurement: BeamMeasurement) -> BeamDecision:
        """Continue BFR procedure after initial candidate selection"""
        ue_id = measurement.ue_id
        state = self.ue_states[ue_id]

        # Check if recovery was successful
        if measurement.serving_rsrp_dbm >= self.config.bfr_rsrp_threshold_dbm:
            # Recovery successful
            state.state = BeamState.P3_TRACKING
            state.current_beam_id = measurement.serving_beam_id
            state.last_good_beam_id = measurement.serving_beam_id
            state.bfd_count = 0

            self.stats["successful_recoveries"] += 1

            logger.info(
                f"BFR successful for UE {ue_id}: "
                f"beam={measurement.serving_beam_id}, RSRP={measurement.serving_rsrp_dbm:.1f} dBm"
            )

            return BeamDecision(
                ue_id=ue_id,
                timestamp_ms=measurement.timestamp_ms,
                procedure=BeamProcedure.P3,
                action="recovered",
                current_beam_id=measurement.serving_beam_id,
                target_beam_id=measurement.serving_beam_id,
                confidence=0.9,
                predicted_rsrp_dbm=measurement.serving_rsrp_dbm,
                reason="BFR completed successfully"
            )

        # Check BFR timer expiry
        elapsed_ms = measurement.timestamp_ms - state.bfr_timer_start_ms
        if elapsed_ms > self.config.bfr_timer_ms:
            if state.bfr_attempts >= self.config.bfr_max_attempts:
                # BFR failed
                state.state = BeamState.FAILED
                self.stats["failed_recoveries"] += 1

                logger.error(
                    f"BFR failed for UE {ue_id} after {state.bfr_attempts} attempts"
                )

                return BeamDecision(
                    ue_id=ue_id,
                    timestamp_ms=measurement.timestamp_ms,
                    procedure=BeamProcedure.P3,
                    action="fail",
                    current_beam_id=measurement.serving_beam_id,
                    target_beam_id=0,
                    confidence=0.0,
                    predicted_rsrp_dbm=measurement.serving_rsrp_dbm,
                    reason=f"BFR failed after {state.bfr_attempts} attempts"
                )
            else:
                # Try another candidate
                return self._execute_bfr(measurement)

        # Continue waiting
        return BeamDecision(
            ue_id=ue_id,
            timestamp_ms=measurement.timestamp_ms,
            procedure=BeamProcedure.P3,
            action="recovering",
            current_beam_id=measurement.serving_beam_id,
            target_beam_id=state.candidate_beam_id,
            confidence=0.5,
            predicted_rsrp_dbm=measurement.serving_rsrp_dbm,
            reason=f"BFR in progress: attempt {state.bfr_attempts}"
        )

    def _find_candidate_beams(self, measurement: BeamMeasurement) -> List[int]:
        """
        Find candidate beams for BFR per 3GPP TS 38.321

        Candidates must have L1-RSRP >= Q_in threshold.
        """
        candidates = []

        for beam_id, rsrp in measurement.neighbor_beams.items():
            if rsrp >= self.config.bfr_rsrp_threshold_dbm:
                candidates.append((beam_id, rsrp))

        # Sort by RSRP (descending)
        candidates.sort(key=lambda x: x[1], reverse=True)

        return [c[0] for c in candidates]

    # =========================================================================
    # Beam Prediction and Switching
    # =========================================================================

    def _predict_optimal_beam(
        self,
        measurement: BeamMeasurement
    ) -> Tuple[int, float]:
        """
        Predict optimal beam based on trajectory and measurements

        Uses multiple prediction methods:
        1. Trajectory-based (if position/velocity available)
        2. RSRP trend analysis
        3. Neighbor beam comparison

        Returns:
            (predicted_beam_id, confidence)
        """
        ue_id = measurement.ue_id
        history = self.measurement_history.get(ue_id, [])

        if len(history) < 3:
            return self._simple_beam_selection(measurement)

        # If position/velocity available, use trajectory prediction
        if measurement.position and measurement.velocity:
            return self._trajectory_based_prediction(measurement)

        # Otherwise use RSRP trend analysis
        return self._rsrp_trend_prediction(history)

    def _simple_beam_selection(
        self,
        measurement: BeamMeasurement
    ) -> Tuple[int, float]:
        """Simple beam selection based on current measurements"""
        if measurement.neighbor_beams:
            best_neighbor = max(
                measurement.neighbor_beams.items(),
                key=lambda x: x[1]
            )
            if (best_neighbor[1] >
                measurement.serving_rsrp_dbm + self.config.beam_switch_hysteresis_db):
                return best_neighbor[0], 0.6

        return measurement.serving_beam_id, 0.5

    def _trajectory_based_prediction(
        self,
        measurement: BeamMeasurement
    ) -> Tuple[int, float]:
        """Predict beam based on UAV trajectory"""
        pos = np.array(measurement.position)
        vel = np.array(measurement.velocity)

        # Predict position after prediction_horizon
        dt = self.config.prediction_horizon_ms / 1000.0
        predicted_pos = pos + vel * dt

        # Calculate AoA/AoD to predicted position (assuming gNB at origin)
        azimuth = np.arctan2(predicted_pos[1], predicted_pos[0])
        elevation = np.arctan2(
            predicted_pos[2],
            np.sqrt(predicted_pos[0]**2 + predicted_pos[1]**2)
        )

        # Map angles to beam index
        predicted_beam = self._angle_to_beam(azimuth, elevation)

        # Confidence based on velocity magnitude
        speed = np.linalg.norm(vel)
        confidence = max(0.5, 1.0 - speed / 50.0)

        return predicted_beam, confidence

    def _rsrp_trend_prediction(
        self,
        history: List[BeamMeasurement]
    ) -> Tuple[int, float]:
        """Predict beam based on RSRP trends"""
        recent = history[-10:]

        # Track RSRP trend for each beam
        beam_trends: Dict[int, List[Tuple[float, float]]] = {}

        for m in recent:
            t = m.timestamp_ms
            if m.serving_beam_id not in beam_trends:
                beam_trends[m.serving_beam_id] = []
            beam_trends[m.serving_beam_id].append((t, m.serving_rsrp_dbm))

            for beam_id, rsrp in m.neighbor_beams.items():
                if beam_id not in beam_trends:
                    beam_trends[beam_id] = []
                beam_trends[beam_id].append((t, rsrp))

        # Find beam with best projected RSRP
        best_beam = recent[-1].serving_beam_id
        best_score = -float('inf')

        for beam_id, measurements in beam_trends.items():
            if len(measurements) >= 3:
                times = np.array([m[0] for m in measurements])
                rsrps = np.array([m[1] for m in measurements])

                # Linear fit for trend
                times_norm = (times - times[0]) / 1000.0
                coeffs = np.polyfit(times_norm, rsrps, 1)
                trend = coeffs[0]

                # Project 5 samples ahead
                score = rsrps[-1] + trend * 0.025  # 25ms ahead

                if score > best_score:
                    best_score = score
                    best_beam = beam_id

        confidence = 0.7 if best_beam != recent[-1].serving_beam_id else 0.4
        return best_beam, confidence

    def _should_switch_beam(
        self,
        state: UEBeamState,
        measurement: BeamMeasurement,
        predicted_beam: int,
        confidence: float
    ) -> bool:
        """
        Determine if beam switch should be executed

        Implements A3 event-like condition with time-to-trigger
        per 3GPP TS 38.331.
        """
        if predicted_beam == measurement.serving_beam_id:
            state.beam_switch_pending = False
            return False

        if confidence < 0.7:
            return False

        # Get predicted beam RSRP
        predicted_rsrp = measurement.neighbor_beams.get(
            predicted_beam,
            self._estimate_rsrp(predicted_beam, measurement)
        )

        # A3 condition: neighbor > serving + offset + hysteresis
        threshold = (measurement.serving_rsrp_dbm +
                    self.config.a3_offset_db +
                    self.config.beam_switch_hysteresis_db)

        if predicted_rsrp > threshold:
            # Check time-to-trigger
            if not state.beam_switch_pending:
                state.beam_switch_pending = True
                state.ttt_timer_start_ms = measurement.timestamp_ms
                return False

            elapsed = measurement.timestamp_ms - state.ttt_timer_start_ms
            if elapsed >= self.config.time_to_trigger_ms:
                return True

        else:
            state.beam_switch_pending = False

        return False

    def _execute_beam_switch(
        self,
        state: UEBeamState,
        measurement: BeamMeasurement,
        target_beam: int,
        confidence: float
    ) -> BeamDecision:
        """Execute beam switch"""
        ue_id = measurement.ue_id

        old_beam = state.current_beam_id
        state.current_beam_id = target_beam
        state.last_good_beam_id = target_beam
        state.num_beam_switches += 1
        state.beam_switch_pending = False
        state.last_update_ms = measurement.timestamp_ms

        self.stats["p3_switches"] += 1

        if self._on_beam_switch:
            self._on_beam_switch(ue_id, old_beam, target_beam)

        predicted_rsrp = measurement.neighbor_beams.get(
            target_beam,
            self._estimate_rsrp(target_beam, measurement)
        )

        logger.info(
            f"Beam switch for UE {ue_id}: {old_beam} -> {target_beam}, "
            f"predicted RSRP={predicted_rsrp:.1f} dBm"
        )

        return BeamDecision(
            ue_id=ue_id,
            timestamp_ms=measurement.timestamp_ms,
            procedure=BeamProcedure.P3,
            action="switch",
            current_beam_id=old_beam,
            target_beam_id=target_beam,
            confidence=confidence,
            predicted_rsrp_dbm=predicted_rsrp,
            reason=f"P3 proactive switch: {old_beam} -> {target_beam}"
        )

    def _angle_to_beam(self, azimuth: float, elevation: float) -> int:
        """Map angles to beam index"""
        if self.codebook is not None:
            return self.codebook.angle_to_beam_id(azimuth, elevation)

        # Default mapping
        az_norm = (azimuth + np.pi/2) / np.pi
        el_norm = (elevation + np.pi/4) / (np.pi/2)

        h_idx = int(az_norm * self.config.num_beams_h) % self.config.num_beams_h
        v_idx = int(el_norm * self.config.num_beams_v) % self.config.num_beams_v

        return h_idx * self.config.num_beams_v + v_idx

    def _estimate_rsrp(self, beam_id: int, measurement: BeamMeasurement) -> float:
        """Estimate RSRP for a given beam"""
        if beam_id in measurement.neighbor_beams:
            return measurement.neighbor_beams[beam_id]

        # Estimate based on angular distance
        current_beam = measurement.serving_beam_id
        beam_distance = abs(beam_id - current_beam) / self.config.total_beams

        # Approximately -3dB per beam width
        return measurement.serving_rsrp_dbm - 3.0 * beam_distance * self.config.num_beams_h

    # =========================================================================
    # Statistics and Monitoring
    # =========================================================================

    def get_ue_state(self, ue_id: str) -> Optional[UEBeamState]:
        """Get current state for UE"""
        return self.ue_states.get(ue_id)

    def get_statistics(self) -> Dict:
        """Get beam tracking statistics"""
        return {
            **self.stats,
            "active_ues": len(self.ue_states),
            "avg_prediction_accuracy": (
                np.mean(self.stats["prediction_accuracy"])
                if self.stats["prediction_accuracy"] else 0.0
            ),
        }

    def get_ue_statistics(self, ue_id: str) -> Optional[Dict]:
        """Get statistics for specific UE"""
        if ue_id not in self.ue_states:
            return None

        state = self.ue_states[ue_id]
        return {
            "ue_id": ue_id,
            "state": state.state.value,
            "current_beam_id": state.current_beam_id,
            "filtered_rsrp_dbm": state.filtered_rsrp_dbm,
            "num_beam_switches": state.num_beam_switches,
            "num_beam_failures": state.num_beam_failures,
            "bfd_count": state.bfd_count,
        }

    def reset(self):
        """Reset tracker state"""
        self.ue_states.clear()
        self.measurement_history.clear()
        self.stats = {
            "p1_acquisitions": 0,
            "p2_refinements": 0,
            "p3_switches": 0,
            "beam_failures": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "prediction_accuracy": [],
        }

    def remove_ue(self, ue_id: str):
        """Remove UE from tracking"""
        self.ue_states.pop(ue_id, None)
        self.measurement_history.pop(ue_id, None)
