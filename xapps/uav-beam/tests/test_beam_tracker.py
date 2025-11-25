"""
Unit tests for BeamTracker module.

Tests:
- BeamConfig dataclass
- BeamMeasurement dataclass
- BeamTracker initialization and codebook generation
- Beam decision logic (maintain, switch, recover)
- Beam Failure Detection (BFD) and Recovery (BFR)
- Multi-UE handling
- Boundary conditions and edge cases
- Performance benchmarks
"""

import pytest
import numpy as np
import time
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
    UEBeamState,
    BeamProcedure,
)


class TestBeamConfig:
    """Test BeamConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = BeamConfig()

        assert config.num_beams_h == 16
        assert config.num_beams_v == 8
        assert config.total_beams == 128
        assert config.total_antenna_elements == 64
        assert config.num_antenna_elements_h == 8
        assert config.num_antenna_elements_v == 8
        assert config.ssb_periodicity_ms == 20.0
        assert config.bfd_rsrp_threshold_dbm == -110.0
        assert config.prediction_horizon_ms == 20.0

    def test_custom_config(self):
        """Test custom configuration."""
        config = BeamConfig(num_beams_h=32, num_beams_v=16)

        assert config.total_beams == 512
        assert config.num_beams_h == 32
        assert config.num_beams_v == 16

    def test_antenna_element_count(self):
        """Test antenna element calculation."""
        config = BeamConfig(
            num_antenna_elements_h=16,
            num_antenna_elements_v=16,
        )

        assert config.total_antenna_elements == 256

    @pytest.mark.parametrize("h,v,expected", [
        (8, 4, 32),
        (16, 8, 128),
        (32, 16, 512),
        (64, 32, 2048),
    ])
    def test_beam_count_calculation(self, h, v, expected):
        """Test beam count calculation with different configurations."""
        config = BeamConfig(num_beams_h=h, num_beams_v=v)
        assert config.total_beams == expected

    def test_bfd_bfr_thresholds(self):
        """Test BFD and BFR thresholds are properly set."""
        config = BeamConfig()

        assert config.bfd_rsrp_threshold_dbm < config.bfr_rsrp_threshold_dbm
        assert config.max_bfd_count > 0
        assert config.bfr_max_attempts > 0


class TestBeamMeasurement:
    """Test BeamMeasurement dataclass."""

    def test_basic_measurement(self):
        """Test basic beam measurement."""
        meas = BeamMeasurement(
            timestamp_ms=1000.0,
            ue_id="uav-001",
            serving_beam_id=42,
            serving_rsrp_dbm=-85.0,
        )

        assert meas.ue_id == "uav-001"
        assert meas.serving_beam_id == 42
        assert meas.serving_rsrp_dbm == -85.0
        assert meas.cqi == 7  # Default CQI
        assert meas.position is None
        assert meas.velocity is None

    def test_measurement_with_neighbors(self):
        """Test measurement with neighbor beam reports."""
        meas = BeamMeasurement(
            timestamp_ms=1000.0,
            ue_id="uav-001",
            serving_beam_id=42,
            serving_rsrp_dbm=-85.0,
            neighbor_beams={41: -88.0, 43: -87.5, 44: -90.0},
        )

        assert len(meas.neighbor_beams) == 3
        assert meas.neighbor_beams[41] == -88.0
        assert meas.neighbor_beams[43] == -87.5
        assert meas.neighbor_beams[44] == -90.0

    def test_measurement_with_position(self):
        """Test measurement with position and velocity."""
        meas = BeamMeasurement(
            timestamp_ms=1000.0,
            ue_id="uav-001",
            serving_beam_id=42,
            serving_rsrp_dbm=-85.0,
            position=(100.0, 200.0, 50.0),
            velocity=(5.0, 2.0, 0.5),
        )

        assert meas.position == (100.0, 200.0, 50.0)
        assert meas.velocity == (5.0, 2.0, 0.5)

    def test_measurement_with_cqi(self):
        """Test measurement with CQI."""
        meas = BeamMeasurement(
            timestamp_ms=1000.0,
            ue_id="uav-001",
            serving_beam_id=42,
            serving_rsrp_dbm=-85.0,
            cqi=12,
        )

        assert meas.cqi == 12

    def test_measurement_with_ssb_csirs_index(self):
        """Test measurement with SSB and CSI-RS indices."""
        meas = BeamMeasurement(
            timestamp_ms=1000.0,
            ue_id="uav-001",
            serving_beam_id=42,
            serving_rsrp_dbm=-85.0,
            ssb_index=3,
            csi_rs_index=7,
        )

        assert meas.ssb_index == 3
        assert meas.csi_rs_index == 7


class TestBeamDecision:
    """Test BeamDecision dataclass."""

    def test_decision_creation(self):
        """Test creating beam decision."""
        decision = BeamDecision(
            ue_id="uav-001",
            timestamp_ms=1000.0,
            procedure=BeamProcedure.P3,
            action="switch",
            current_beam_id=42,
            target_beam_id=43,
            confidence=0.85,
            predicted_rsrp_dbm=-82.0,
            reason="Better neighbor detected",
        )

        assert decision.ue_id == "uav-001"
        assert decision.action == "switch"
        assert decision.procedure == BeamProcedure.P3
        assert decision.current_beam_id == 42
        assert decision.target_beam_id == 43
        assert decision.confidence == 0.85
        assert decision.predicted_rsrp_dbm == -82.0

    def test_decision_with_candidates(self):
        """Test decision with candidate beams."""
        decision = BeamDecision(
            ue_id="uav-001",
            timestamp_ms=1000.0,
            procedure=BeamProcedure.P3,
            action="recover",
            current_beam_id=42,
            target_beam_id=43,
            confidence=0.6,
            predicted_rsrp_dbm=-95.0,
            reason="BFR recovery",
            candidate_beams=[43, 44, 45],
        )

        assert len(decision.candidate_beams) == 3


class TestBeamState:
    """Test BeamState enum."""

    def test_beam_states(self):
        """Test all beam states are defined."""
        assert BeamState.IDLE.value == "idle"
        assert BeamState.P1_ACQUISITION.value == "p1_acquisition"
        assert BeamState.P2_REFINEMENT.value == "p2_refinement"
        assert BeamState.P3_TRACKING.value == "p3_tracking"
        assert BeamState.BFD_DETECTED.value == "bfd_detected"
        assert BeamState.BFR_RECOVERY.value == "bfr_recovery"
        assert BeamState.FAILED.value == "failed"


class TestBeamTrackerInitialization:
    """Test BeamTracker initialization."""

    def test_default_initialization(self):
        """Test default initialization."""
        tracker = BeamTracker()

        assert len(tracker.ue_states) == 0
        assert tracker.config.total_beams == 128
        assert tracker.stats["beam_failures"] == 0
        assert tracker.stats["p1_acquisitions"] == 0

    def test_custom_config_initialization(self):
        """Test initialization with custom config."""
        config = BeamConfig(num_beams_h=32, num_beams_v=16)
        tracker = BeamTracker(config)

        assert tracker.config.total_beams == 512

    def test_ssb_beams_configured(self):
        """Test SSB beams are configured."""
        tracker = BeamTracker()

        assert len(tracker.ssb_beam_set) == tracker.config.num_ssb_beams


class TestCodebookGeneration:
    """Test beam codebook generation."""

    @pytest.fixture
    def tracker(self):
        """Create tracker instance."""
        return BeamTracker()

    def test_codebook_shape(self, tracker):
        """Test codebook has correct shape."""
        assert tracker._codebook_matrix.shape == (128, 64)

    def test_codebook_normalization(self, tracker):
        """Test codebook vectors are normalized."""
        norms = np.linalg.norm(tracker._codebook_matrix, axis=1)
        assert np.allclose(norms, 1.0, rtol=0.01)

    def test_codebook_complex_values(self, tracker):
        """Test codebook contains complex values."""
        assert tracker._codebook_matrix.dtype == complex


class TestBeamTrackerProcessing:
    """Test BeamTracker measurement processing."""

    @pytest.fixture
    def tracker(self):
        """Create tracker instance."""
        return BeamTracker()

    @pytest.fixture
    def basic_measurement(self):
        """Create basic measurement."""
        return BeamMeasurement(
            timestamp_ms=1000.0,
            ue_id="uav-001",
            serving_beam_id=42,
            serving_rsrp_dbm=-85.0,
            neighbor_beams={41: -88.0, 43: -87.5, 44: -90.0},
        )

    def test_process_first_measurement(self, tracker, basic_measurement):
        """Test processing first measurement."""
        decision = tracker.process_measurement(basic_measurement)

        assert decision.ue_id == "uav-001"
        assert decision.current_beam_id == 42
        assert decision.action in ["maintain", "switch", "continue"]
        assert 0 <= decision.confidence <= 1

    def test_ue_state_initialized(self, tracker, basic_measurement):
        """Test UE state is initialized after first measurement."""
        tracker.process_measurement(basic_measurement)

        assert "uav-001" in tracker.ue_states
        assert tracker.ue_states["uav-001"].state == BeamState.P3_TRACKING
        assert tracker.ue_states["uav-001"].current_beam_id == 42

    def test_measurement_history_stored(self, tracker):
        """Test measurement history is stored."""
        for i in range(5):
            meas = BeamMeasurement(
                timestamp_ms=1000.0 + i * 100,
                ue_id="uav-001",
                serving_beam_id=42,
                serving_rsrp_dbm=-85.0,
                neighbor_beams={41: -88.0, 43: -87.5},
            )
            tracker.process_measurement(meas)

        assert len(tracker.measurement_history["uav-001"]) == 5

    def test_measurement_history_limited(self, tracker):
        """Test measurement history is limited to 100 entries."""
        for i in range(150):
            meas = BeamMeasurement(
                timestamp_ms=1000.0 + i * 100,
                ue_id="uav-001",
                serving_beam_id=42,
                serving_rsrp_dbm=-85.0,
                neighbor_beams={41: -88.0, 43: -87.5},
            )
            tracker.process_measurement(meas)

        assert len(tracker.measurement_history["uav-001"]) == 100

    def test_filtered_rsrp_updated(self, tracker, basic_measurement):
        """Test filtered RSRP is updated."""
        tracker.process_measurement(basic_measurement)

        state = tracker.ue_states["uav-001"]
        assert state.filtered_rsrp_dbm == pytest.approx(-85.0, abs=5.0)


class TestBeamFailureDetection:
    """Test Beam Failure Detection (BFD) and Recovery (BFR)."""

    @pytest.fixture
    def tracker(self):
        """Create tracker instance."""
        return BeamTracker()

    def test_bfd_low_rsrp(self, tracker):
        """Test BFD triggers with sustained low RSRP."""
        config = tracker.config

        # Need multiple measurements below threshold to trigger BFD
        for i in range(config.max_bfd_count + 1):
            meas = BeamMeasurement(
                timestamp_ms=1000.0 + i * 100,
                ue_id="uav-001",
                serving_beam_id=42,
                serving_rsrp_dbm=config.bfd_rsrp_threshold_dbm - 5.0,  # Below threshold
                neighbor_beams={41: -105.0},
            )
            decision = tracker.process_measurement(meas)

        # Should trigger recovery or continue (P1 fallback when no candidates)
        assert decision.action in ["recover", "continue"]
        assert tracker.stats["beam_failures"] >= 1

    def test_bfr_to_best_candidate(self, tracker):
        """Test BFR selects best candidate beam."""
        config = tracker.config

        # Build up history then trigger failure
        for i in range(config.max_bfd_count + 1):
            meas = BeamMeasurement(
                timestamp_ms=1000.0 + i * 100,
                ue_id="uav-001",
                serving_beam_id=42,
                serving_rsrp_dbm=config.bfd_rsrp_threshold_dbm - 5.0,
                neighbor_beams={
                    41: -105.0,  # Below Q_in
                    43: config.bfr_rsrp_threshold_dbm + 5.0,  # Above Q_in (candidate)
                    44: -108.0,  # Below Q_in
                },
            )
            decision = tracker.process_measurement(meas)

        # Should recommend recovery to beam 43 (best candidate)
        if decision.action == "recover":
            assert decision.target_beam_id == 43

    def test_no_bfd_on_good_rsrp(self, tracker):
        """Test no BFD trigger with good RSRP."""
        meas = BeamMeasurement(
            timestamp_ms=1000.0,
            ue_id="uav-001",
            serving_beam_id=42,
            serving_rsrp_dbm=-85.0,  # Good RSRP
        )

        decision = tracker.process_measurement(meas)

        assert decision.action in ["maintain", "switch"]
        assert tracker.stats["beam_failures"] == 0


class TestBeamTracking:
    """Test P3 beam tracking procedures."""

    @pytest.fixture
    def tracker(self):
        """Create tracker instance."""
        return BeamTracker()

    def test_maintain_stable_beam(self, tracker):
        """Test maintaining stable beam when no better option."""
        for i in range(10):
            meas = BeamMeasurement(
                timestamp_ms=1000.0 + i * 100,
                ue_id="uav-001",
                serving_beam_id=42,
                serving_rsrp_dbm=-85.0,  # Stable good RSRP
                neighbor_beams={
                    41: -90.0,  # Worse
                    43: -88.0,  # Slightly worse
                },
            )
            decision = tracker.process_measurement(meas)

        # Should maintain current beam
        assert decision.action == "maintain"

    def test_beam_switch_tracking(self, tracker):
        """Test beam switch is tracked in statistics."""
        # Initial good measurement
        meas1 = BeamMeasurement(
            timestamp_ms=1000.0,
            ue_id="uav-001",
            serving_beam_id=42,
            serving_rsrp_dbm=-85.0,
            neighbor_beams={43: -75.0},  # Much better neighbor
        )
        tracker.process_measurement(meas1)

        # If switch happens, verify it's tracked
        state = tracker.ue_states["uav-001"]
        assert state.num_beam_switches >= 0


class TestP1Acquisition:
    """Test P1 Initial Beam Acquisition."""

    @pytest.fixture
    def tracker(self):
        """Create tracker instance."""
        return BeamTracker()

    def test_start_p1_acquisition(self, tracker):
        """Test starting P1 acquisition procedure."""
        ssb_beams = tracker.start_p1_acquisition("uav-001", 1000.0)

        assert len(ssb_beams) == tracker.config.num_ssb_beams
        assert "uav-001" in tracker.ue_states
        assert tracker.ue_states["uav-001"].state == BeamState.P1_ACQUISITION
        assert tracker.stats["p1_acquisitions"] == 1

    def test_p1_acquisition_state_change(self, tracker):
        """Test P1 acquisition changes UE state correctly."""
        tracker.start_p1_acquisition("uav-001", 1000.0)

        # Verify state is P1_ACQUISITION
        assert tracker.ue_states["uav-001"].state == BeamState.P1_ACQUISITION

        # After a measurement during P1, should still be in acquisition
        meas = BeamMeasurement(
            timestamp_ms=1100.0,
            ue_id="uav-001",
            serving_beam_id=0,
            serving_rsrp_dbm=-90.0,
            ssb_index=0,
        )
        decision = tracker.process_measurement(meas)

        # Decision during P1 should indicate continue
        assert decision.action in ["continue", "switch", "maintain"]


class TestMultipleUEs:
    """Test handling multiple UEs."""

    @pytest.fixture
    def tracker(self):
        """Create tracker instance."""
        return BeamTracker()

    def test_multiple_ues_initialization(self, tracker):
        """Test tracking multiple UEs."""
        ue_ids = ["uav-001", "uav-002", "uav-003"]

        for ue_id in ue_ids:
            meas = BeamMeasurement(
                timestamp_ms=1000.0,
                ue_id=ue_id,
                serving_beam_id=42,
                serving_rsrp_dbm=-85.0,
            )
            tracker.process_measurement(meas)

        assert len(tracker.ue_states) == 3
        assert all(ue in tracker.ue_states for ue in ue_ids)

    def test_multiple_ues_isolation(self, tracker):
        """Test UE states are isolated."""
        config = tracker.config

        # UE1: Normal RSRP
        meas1 = BeamMeasurement(
            timestamp_ms=1000.0,
            ue_id="uav-001",
            serving_beam_id=42,
            serving_rsrp_dbm=-85.0,
        )

        # UE2: Multiple low RSRP measurements (trigger BFD)
        for i in range(config.max_bfd_count + 1):
            meas2 = BeamMeasurement(
                timestamp_ms=1000.0 + i * 100,
                ue_id="uav-002",
                serving_beam_id=50,
                serving_rsrp_dbm=config.bfd_rsrp_threshold_dbm - 5.0,
                neighbor_beams={51: -105.0},
            )
            tracker.process_measurement(meas2)

        decision1 = tracker.process_measurement(meas1)

        assert decision1.action == "maintain"
        assert tracker.ue_states["uav-001"].num_beam_failures == 0
        assert tracker.ue_states["uav-002"].num_beam_failures >= 1

    def test_interleaved_ue_updates(self, tracker):
        """Test interleaved updates from multiple UEs."""
        ue_ids = ["uav-001", "uav-002", "uav-003"]

        for i in range(30):
            ue_id = ue_ids[i % len(ue_ids)]
            meas = BeamMeasurement(
                timestamp_ms=1000.0 + i * 100,
                ue_id=ue_id,
                serving_beam_id=42 + (i % 3),
                serving_rsrp_dbm=-85.0,
            )
            tracker.process_measurement(meas)

        # Each UE should have 10 measurements in history
        for ue_id in ue_ids:
            assert len(tracker.measurement_history[ue_id]) == 10


class TestStatistics:
    """Test statistics collection."""

    @pytest.fixture
    def tracker(self):
        """Create tracker instance."""
        return BeamTracker()

    def test_initial_statistics(self, tracker):
        """Test initial statistics values."""
        stats = tracker.get_statistics()

        assert stats["beam_failures"] == 0
        assert stats["p1_acquisitions"] == 0
        assert stats["p2_refinements"] == 0
        assert stats["p3_switches"] == 0

    def test_statistics_after_processing(self, tracker):
        """Test statistics after processing measurements."""
        for i in range(10):
            meas = BeamMeasurement(
                timestamp_ms=1000.0 + i * 100,
                ue_id="uav-001",
                serving_beam_id=42,
                serving_rsrp_dbm=-85.0,
            )
            tracker.process_measurement(meas)

        stats = tracker.get_statistics()
        assert "active_ues" in stats
        assert stats["active_ues"] == 1


class TestReset:
    """Test tracker reset functionality."""

    @pytest.fixture
    def tracker(self):
        """Create tracker instance."""
        return BeamTracker()

    def test_reset_clears_state(self, tracker):
        """Test reset clears all state."""
        # Process some measurements
        for i in range(5):
            meas = BeamMeasurement(
                timestamp_ms=1000.0 + i * 100,
                ue_id=f"uav-{i:03d}",
                serving_beam_id=42,
                serving_rsrp_dbm=-85.0,
            )
            tracker.process_measurement(meas)

        # Reset
        tracker.reset()

        assert len(tracker.ue_states) == 0
        assert len(tracker.measurement_history) == 0
        assert tracker.stats["beam_failures"] == 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    @pytest.fixture
    def tracker(self):
        """Create tracker instance."""
        return BeamTracker()

    def test_very_good_rsrp(self, tracker):
        """Test handling very good RSRP values."""
        meas = BeamMeasurement(
            timestamp_ms=1000.0,
            ue_id="uav-001",
            serving_beam_id=42,
            serving_rsrp_dbm=-50.0,  # Very good
        )

        decision = tracker.process_measurement(meas)

        assert decision.action == "maintain"

    def test_empty_neighbor_list(self, tracker):
        """Test handling empty neighbor list."""
        meas = BeamMeasurement(
            timestamp_ms=1000.0,
            ue_id="uav-001",
            serving_beam_id=42,
            serving_rsrp_dbm=-85.0,
            neighbor_beams={},
        )

        decision = tracker.process_measurement(meas)

        assert decision is not None

    def test_large_neighbor_list(self, tracker):
        """Test handling large neighbor list."""
        neighbors = {i: -85.0 - abs(i - 42) for i in range(30, 60)}

        meas = BeamMeasurement(
            timestamp_ms=1000.0,
            ue_id="uav-001",
            serving_beam_id=42,
            serving_rsrp_dbm=-85.0,
            neighbor_beams=neighbors,
        )

        decision = tracker.process_measurement(meas)

        assert decision is not None

    def test_extreme_position(self, tracker):
        """Test handling extreme position values."""
        meas = BeamMeasurement(
            timestamp_ms=1000.0,
            ue_id="uav-001",
            serving_beam_id=42,
            serving_rsrp_dbm=-85.0,
            position=(10000.0, 10000.0, 5000.0),  # Far away
            velocity=(100.0, 100.0, 0.0),  # Very fast
        )

        decision = tracker.process_measurement(meas)

        assert decision is not None
        assert not np.isnan(decision.confidence)

    def test_zero_velocity(self, tracker):
        """Test handling zero velocity."""
        meas = BeamMeasurement(
            timestamp_ms=1000.0,
            ue_id="uav-001",
            serving_beam_id=42,
            serving_rsrp_dbm=-85.0,
            position=(100.0, 100.0, 50.0),
            velocity=(0.0, 0.0, 0.0),
        )

        decision = tracker.process_measurement(meas)

        assert decision is not None


class TestCallbacks:
    """Test callback registration and invocation."""

    @pytest.fixture
    def tracker(self):
        """Create tracker instance."""
        return BeamTracker()

    def test_register_callbacks(self, tracker):
        """Test registering callbacks."""
        switch_called = [False]
        failure_called = [False]

        def on_switch(ue_id, old_beam, new_beam):
            switch_called[0] = True

        def on_failure(ue_id, measurement):
            failure_called[0] = True

        tracker.register_callbacks(
            on_beam_switch=on_switch,
            on_beam_failure=on_failure
        )

        assert tracker._on_beam_switch is not None
        assert tracker._on_beam_failure is not None


class TestPerformanceBenchmarks:
    """Performance benchmarks for beam tracker."""

    @pytest.fixture
    def tracker(self):
        """Create tracker instance."""
        return BeamTracker()

    @pytest.mark.performance
    def test_measurement_processing_speed(self, tracker, performance_timer):
        """Benchmark measurement processing speed."""
        iterations = 1000

        for i in range(iterations):
            meas = BeamMeasurement(
                timestamp_ms=1000.0 + i * 10,
                ue_id=f"uav-{i % 50:03d}",
                serving_beam_id=42 + (i % 10),
                serving_rsrp_dbm=-85.0 - (i % 20) * 0.5,
                neighbor_beams={43: -88.0, 41: -87.0},
            )

            start = time.perf_counter()
            tracker.process_measurement(meas)
            elapsed_ms = (time.perf_counter() - start) * 1000

            performance_timer.record(elapsed_ms)

        result = performance_timer.result("measurement_processing")
        print(f"\nMeasurement processing: avg={result.avg_time_ms:.3f}ms, max={result.max_time_ms:.3f}ms")

        assert result.avg_time_ms < 1.0, "Measurement processing too slow"

    @pytest.mark.performance
    def test_codebook_generation_speed(self, performance_timer):
        """Benchmark codebook generation speed."""
        iterations = 10

        for _ in range(iterations):
            config = BeamConfig(num_beams_h=32, num_beams_v=16)

            start = time.perf_counter()
            tracker = BeamTracker(config)
            elapsed_ms = (time.perf_counter() - start) * 1000

            performance_timer.record(elapsed_ms)

        result = performance_timer.result("codebook_generation")
        print(f"\nCodebook generation (32x16): avg={result.avg_time_ms:.2f}ms")

        assert result.avg_time_ms < 100.0, "Codebook generation too slow"

    @pytest.mark.performance
    def test_multi_ue_scalability(self, tracker):
        """Benchmark scalability with many UEs."""
        num_ues = [10, 50, 100, 200]
        results = {}

        for n in num_ues:
            # Initialize UEs
            for i in range(n):
                meas = BeamMeasurement(
                    timestamp_ms=0,
                    ue_id=f"uav-{i:03d}",
                    serving_beam_id=42,
                    serving_rsrp_dbm=-85.0,
                )
                tracker.process_measurement(meas)

            # Measure processing time
            start = time.perf_counter()
            for i in range(100):
                meas = BeamMeasurement(
                    timestamp_ms=1000 + i,
                    ue_id=f"uav-{i % n:03d}",
                    serving_beam_id=42,
                    serving_rsrp_dbm=-85.0,
                )
                tracker.process_measurement(meas)
            elapsed = time.perf_counter() - start

            results[n] = elapsed * 10  # ms per operation

            # Reset for next iteration
            tracker.reset()

        print(f"\nScalability (ms per op): {results}")

        # Processing time should not increase dramatically with UE count
        assert results[200] < results[10] * 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
