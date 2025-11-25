from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from uav_policy.policy_engine import (
    PathSegmentPlan,
    FlightPlanPolicy,
    ServiceProfile,
)


@dataclass
class Waypoint:
    """Discretized waypoint along a UAV path."""

    index: int
    x: float
    y: float
    z: float


@dataclass
class CellMetric:
    """Radio metric for a cell at a given waypoint."""

    sinr_db: float
    load: float  # Expected PRB utilization [0, 1]


@dataclass
class RadioMap:
    """Radio map indexed by waypoint index and cell id."""

    metrics: Dict[int, Dict[str, CellMetric]]

    def get_cells_for_step(self, step: int) -> Dict[str, CellMetric]:
        return self.metrics.get(step, {})


@dataclass
class PlannerConfig:
    """Configuration and utility parameters for the DP planner."""

    sinr_min_db: float = -5.0
    w_sinr: float = 1.0
    w_load: float = 0.5
    ho_penalty: float = 0.5

    def utility(self, sinr_db: float, load: float) -> float:
        """Higher is better. Encourages SINR above threshold, discourages load."""
        sinr_margin = sinr_db - self.sinr_min_db
        return self.w_sinr * sinr_margin - self.w_load * load


def _choose_cells_dp(
    waypoints: List[Waypoint],
    radio_map: RadioMap,
    config: PlannerConfig,
) -> List[str]:
    """Dynamic programming over waypoints to choose a serving cell per step."""
    if not waypoints:
        return []

    # Ensure waypoints are sorted by index
    waypoints = sorted(waypoints, key=lambda w: w.index)
    steps = [w.index for w in waypoints]
    step_to_pos = {w.index: i for i, w in enumerate(waypoints)}
    n_steps = len(steps)

    # dp_states[step_idx][cell_id] = (score, prev_cell_id)
    dp_states: List[Dict[str, Tuple[float, Optional[str]]]] = []

    # Initialize at first step
    first_step = steps[0]
    first_cells = radio_map.get_cells_for_step(first_step)
    if not first_cells:
        raise ValueError("RadioMap has no metrics for first waypoint.")

    # Candidate cells for first step
    candidates_first: Dict[str, CellMetric] = {}
    for cell_id, met in first_cells.items():
        if met.sinr_db >= config.sinr_min_db:
            candidates_first[cell_id] = met
    if not candidates_first:
        # Fallback: choose cell with highest SINR
        best_cell = max(first_cells.items(), key=lambda kv: kv[1].sinr_db)[0]
        candidates_first[best_cell] = first_cells[best_cell]

    dp0: Dict[str, Tuple[float, Optional[str]]] = {}
    for cell_id, met in candidates_first.items():
        score = config.utility(met.sinr_db, met.load)
        dp0[cell_id] = (score, None)
    dp_states.append(dp0)

    # Iterate over remaining steps
    for i in range(1, n_steps):
        step = steps[i]
        cell_metrics = radio_map.get_cells_for_step(step)
        if not cell_metrics:
            raise ValueError(f"RadioMap has no metrics for waypoint index={step}.")

        candidates: Dict[str, CellMetric] = {}
        for cell_id, met in cell_metrics.items():
            if met.sinr_db >= config.sinr_min_db:
                candidates[cell_id] = met
        if not candidates:
            # Fallback: pick cell with highest SINR
            best_cell = max(cell_metrics.items(), key=lambda kv: kv[1].sinr_db)[0]
            candidates[best_cell] = cell_metrics[best_cell]

        prev_dp = dp_states[-1]
        curr_dp: Dict[str, Tuple[float, Optional[str]]] = {}

        for cell_id, met in candidates.items():
            best_score: Optional[float] = None
            best_prev_cell: Optional[str] = None
            for prev_cell, (prev_score, _) in prev_dp.items():
                score = prev_score + config.utility(met.sinr_db, met.load)
                if prev_cell != cell_id:
                    score -= config.ho_penalty
                if best_score is None or score > best_score:
                    best_score = score
                    best_prev_cell = prev_cell
            assert best_score is not None
            curr_dp[cell_id] = (best_score, best_prev_cell)

        dp_states.append(curr_dp)

    # Backtrack
    last_dp = dp_states[-1]
    last_cell = max(last_dp.items(), key=lambda kv: kv[1][0])[0]

    chosen_cells: List[str] = ["" for _ in range(n_steps)]
    chosen_cells[-1] = last_cell
    for i in range(n_steps - 1, 0, -1):
        _, prev_cell = dp_states[i][chosen_cells[i]]
        assert prev_cell is not None
        chosen_cells[i - 1] = prev_cell

    return chosen_cells


def _compress_to_segments(
    uav_id: str,
    waypoints: List[Waypoint],
    cells: List[str],
    service: ServiceProfile,
) -> FlightPlanPolicy:
    if not waypoints or not cells:
        return FlightPlanPolicy(uav_id=uav_id, segments=[])

    waypoints = sorted(waypoints, key=lambda w: w.index)
    n = len(waypoints)
    norm = max(1, n - 1)

    segments: List[PathSegmentPlan] = []

    current_cell = cells[0]
    seg_start_idx = 0

    def add_segment(end_idx: int) -> None:
        nonlocal seg_start_idx, current_cell, segments
        start_wp = waypoints[seg_start_idx]
        end_wp = waypoints[end_idx]

        start_pos = start_wp.index / norm
        end_pos = end_wp.index / norm
        if end_idx == n - 1:
            end_pos = min(1.0, end_pos + 1e-9)

        base_quota = max(5, int(service.target_bitrate_mbps))
        seg = PathSegmentPlan(
            start_pos=start_pos,
            end_pos=end_pos,
            planned_cell_id=current_cell,
            slice_id=service.name,
            base_prb_quota=base_quota,
        )
        segments.append(seg)

    for i in range(1, n):
        if cells[i] != current_cell:
            add_segment(i - 1)
            current_cell = cells[i]
            seg_start_idx = i

    add_segment(n - 1)
    return FlightPlanPolicy(uav_id=uav_id, segments=segments)


def plan_flight_path(
    uav_id: str,
    waypoints: List[Waypoint],
    radio_map: RadioMap,
    service: ServiceProfile,
    config: Optional[PlannerConfig] = None,
) -> FlightPlanPolicy:
    """High-level entry point: waypoints + radio map -> FlightPlanPolicy."""
    if config is None:
        config = PlannerConfig(sinr_min_db=service.min_sinr_db)

    cells = _choose_cells_dp(waypoints, radio_map, config)
    return _compress_to_segments(uav_id, waypoints, cells, service)


def policy_to_dict(policy: FlightPlanPolicy) -> dict:
    """Serialize FlightPlanPolicy to a JSON-serializable dict."""
    return {
        "uav_id": policy.uav_id,
        "segments": [
            {
                "start_pos": s.start_pos,
                "end_pos": s.end_pos,
                "planned_cell_id": s.planned_cell_id,
                "slice_id": s.slice_id,
                "base_prb_quota": s.base_prb_quota,
            }
            for s in policy.segments
        ],
    }


def policy_from_dict(data: dict) -> FlightPlanPolicy:
    """Deserialize FlightPlanPolicy from a dict."""
    segments = [
        PathSegmentPlan(
            start_pos=s["start_pos"],
            end_pos=s["end_pos"],
            planned_cell_id=s["planned_cell_id"],
            slice_id=s["slice_id"],
            base_prb_quota=s["base_prb_quota"],
        )
        for s in data.get("segments", [])
    ]
    return FlightPlanPolicy(uav_id=data["uav_id"], segments=segments)
