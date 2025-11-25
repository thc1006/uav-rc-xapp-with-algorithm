from __future__ import annotations

from dataclasses import dataclass
from math import ceil, log2
from typing import List, Optional


@dataclass
class UavState:
    """Minimal UAV state used by the policy engine."""

    uav_id: str
    x: float
    y: float
    z: float
    slice_id: Optional[str] = None
    path_position: Optional[float] = None


@dataclass
class RadioSnapshot:
    """Per-UAV view of the radio environment at a given time step."""

    serving_cell_id: str
    neighbor_cell_ids: List[str]
    rsrp_serving: float
    rsrp_best_neighbor: float
    prb_utilization_serving: float
    prb_utilization_slice: Optional[float] = None


@dataclass
class PathSegmentPlan:
    """Planned serving cell and resource profile for a path segment."""

    start_pos: float
    end_pos: float
    planned_cell_id: str
    slice_id: str
    base_prb_quota: int


@dataclass
class FlightPlanPolicy:
    """Offline-derived flight-plan policy for a single UAV."""

    uav_id: str
    segments: List[PathSegmentPlan]


@dataclass
class ServiceProfile:
    """QoS profile for a UAV service (e.g., HD video uplink)."""

    name: str
    target_bitrate_mbps: float
    min_sinr_db: float = 0.0


@dataclass
class ResourceDecision:
    """High-level decision about how to treat this UAV."""

    uav_id: str
    target_cell_id: str
    slice_id: Optional[str]
    prb_quota: Optional[int]
    reason: str


def find_active_segment(plan: FlightPlanPolicy, path_position: float) -> Optional[PathSegmentPlan]:
    """Return the active path segment for a given path position."""
    for seg in plan.segments:
        if seg.start_pos <= path_position < seg.end_pos:
            return seg
    if plan.segments:
        return plan.segments[-1]
    return None


def estimate_required_prb(
    target_bitrate_mbps: float,
    sinr_db: float,
    prb_bandwidth_hz: float = 180e3,
) -> int:
    """Estimate the PRB count needed for a target bitrate using a rough model."""
    if sinr_db < -10.0:
        sinr_db = -10.0

    sinr_linear = 10 ** (sinr_db / 10.0)
    se_bps_per_hz = log2(1.0 + sinr_linear)
    if se_bps_per_hz <= 0:
        return 1

    throughput_per_prb_mbps = se_bps_per_hz * prb_bandwidth_hz / 1e6
    if throughput_per_prb_mbps <= 0:
        return 1

    return max(1, int(ceil(target_bitrate_mbps / throughput_per_prb_mbps)))


def simple_path_aware_policy(
    uav: UavState,
    radio: RadioSnapshot,
    overloaded_threshold: float = 0.8,
    hysteresis_db: float = 3.0,
) -> ResourceDecision:
    """Baseline policy: neighbor if serving is hot + clearly weaker."""

    target_cell = radio.serving_cell_id
    reason: str

    if (
        radio.prb_utilization_serving > overloaded_threshold
        and radio.rsrp_best_neighbor > radio.rsrp_serving + hysteresis_db
        and radio.neighbor_cell_ids
    ):
        target_cell = radio.neighbor_cell_ids[0]
        reason = "Serving cell overloaded, neighbor clearly stronger."
    else:
        reason = "Stay on serving cell; load acceptable or neighbors not clearly better."

    prb_quota = 20

    return ResourceDecision(
        uav_id=uav.uav_id,
        target_cell_id=target_cell,
        slice_id=uav.slice_id,
        prb_quota=prb_quota,
        reason=reason,
    )


def path_aware_rc_policy(
    uav: UavState,
    radio: RadioSnapshot,
    plan: Optional[FlightPlanPolicy] = None,
    service: Optional[ServiceProfile] = None,
    overloaded_threshold: float = 0.8,
    hysteresis_db: float = 3.0,
    min_prb_quota: int = 5,
    max_prb_quota: int = 100,
) -> ResourceDecision:
    """Path-aware RC policy implementing docs/algorithms.md."""

    active_seg: Optional[PathSegmentPlan] = None
    if plan is not None and uav.path_position is not None:
        active_seg = find_active_segment(plan, uav.path_position)

    target_cell = radio.serving_cell_id
    reason_parts: List[str] = []

    if active_seg is not None:
        planned_cell = active_seg.planned_cell_id
        if planned_cell != radio.serving_cell_id:
            if radio.prb_utilization_serving > overloaded_threshold:
                if radio.rsrp_best_neighbor > radio.rsrp_serving + hysteresis_db:
                    target_cell = planned_cell
                    reason_parts.append(
                        "Follow flight-plan cell; serving overloaded and neighbor stronger."
                    )
                else:
                    reason_parts.append(
                        "Flight-plan suggests different cell but neighbor not clearly better; stay on serving."
                    )
            else:
                reason_parts.append(
                    "Flight-plan suggests different cell but serving not overloaded; stay on serving for stability."
                )
        else:
            reason_parts.append("Serving cell matches flight-plan segment.")
    else:
        reason_parts.append("No active flight-plan segment; using reactive policy only.")
        # Apply reactive handover logic when no flight plan
        if (
            radio.prb_utilization_serving > overloaded_threshold
            and radio.rsrp_best_neighbor > radio.rsrp_serving + hysteresis_db
            and radio.neighbor_cell_ids
        ):
            target_cell = radio.neighbor_cell_ids[0]
            reason_parts.append(
                f"Reactive handover: serving overloaded (util={radio.prb_utilization_serving:.1%}), "
                f"neighbor stronger by {radio.rsrp_best_neighbor - radio.rsrp_serving:.1f} dB."
            )
        else:
            reason_parts.append("Staying on serving cell (not overloaded or neighbors not clearly better).")

    target_slice: Optional[str]
    if uav.slice_id is not None:
        target_slice = uav.slice_id
        reason_parts.append(f"Using UAV slice_id={uav.slice_id}.")
    elif active_seg is not None:
        target_slice = active_seg.slice_id
        reason_parts.append(f"Using slice from flight-plan segment: {active_seg.slice_id}.")
    else:
        target_slice = None
        reason_parts.append("No slice info; leaving slice_id unset.")

    base_quota = active_seg.base_prb_quota if active_seg is not None else min_prb_quota
    required_quota = base_quota

    if service is not None:
        if target_cell == radio.serving_cell_id:
            sinr_for_estimation = radio.rsrp_serving
            reason_parts.append("Estimating PRB from serving-cell RSRP as SINR proxy.")
        else:
            sinr_for_estimation = radio.rsrp_best_neighbor
            reason_parts.append("Estimating PRB from best-neighbor RSRP as SINR proxy.")

        if sinr_for_estimation < service.min_sinr_db:
            reason_parts.append(
                f"Effective SINR ({sinr_for_estimation:.1f} dB) below service minimum ({service.min_sinr_db:.1f} dB)."
            )

        required_quota = estimate_required_prb(
            target_bitrate_mbps=service.target_bitrate_mbps,
            sinr_db=sinr_for_estimation,
        )
        reason_parts.append(
            f"Service '{service.name}' targets {service.target_bitrate_mbps:.2f} Mbps; "
            f"estimated required PRB quota ≈ {required_quota}."
        )

        required_quota = max(required_quota, base_quota)
    else:
        reason_parts.append("No ServiceProfile provided; using base quota from flight-plan or default.")

    prb_quota = max(min_prb_quota, min(required_quota, max_prb_quota))
    reason = " ".join(reason_parts)

    return ResourceDecision(
        uav_id=uav.uav_id,
        target_cell_id=target_cell,
        slice_id=target_slice,
        prb_quota=prb_quota,
        reason=reason,
    )
