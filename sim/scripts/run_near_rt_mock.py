"""Mock Near-RT loop applying the UAV policy along a path.

This script:

1. Loads a FlightPlanPolicy JSON artifact produced by run_nonrt_planner.py.
2. Replays the UAV path from `sim/nsoran/uav_paths.yaml`.
3. For each step, builds a synthetic RadioSnapshot and applies
   `path_aware_rc_policy`.
4. Writes decisions as JSONL to `sim/artifacts/decisions/uav-001-demo.jsonl`.

In a real setup, you would replace the synthetic RadioSnapshot with live
KPM reports from ns-O-RAN or another testbed.
"""

import json
import sys
from pathlib import Path

import yaml  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT / "nonrt" / "uav-path-planner" / "src"))
sys.path.append(str(REPO_ROOT / "xapps" / "uav-policy" / "src"))

from uav_policy.policy_engine import (  # noqa: E402
    UavState,
    RadioSnapshot,
    ServiceProfile,
    path_aware_rc_policy,
)
from uav_path_planner.planner import (  # noqa: E402
    Waypoint,
    CellMetric,
    RadioMap,
    policy_from_dict,
)


def load_uav_path(path: Path) -> tuple[str, list[Waypoint]]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    uav = data["uavs"][0]
    uav_id = uav["uav_id"]
    wps = [
        Waypoint(
            index=wp["index"],
            x=float(wp["x"]),
            y=float(wp["y"]),
            z=float(wp["z"]),
        )
        for wp in uav["waypoints"]
    ]
    return uav_id, wps


def build_synthetic_radiomap(waypoints: list[Waypoint]) -> RadioMap:
    metrics = {}
    for wp in waypoints:
        if wp.x <= 75.0:
            metrics[wp.index] = {
                "cell-A": CellMetric(sinr_db=0.0 + 0.02 * wp.x, load=0.4),
                "cell-B": CellMetric(sinr_db=-2.0 + 0.01 * wp.x, load=0.2),
            }
        else:
            metrics[wp.index] = {
                "cell-A": CellMetric(sinr_db=-2.0 + 0.01 * (150.0 - wp.x), load=0.6),
                "cell-B": CellMetric(sinr_db=0.0 + 0.02 * (wp.x - 75.0), load=0.3),
            }
    return RadioMap(metrics=metrics)


def main() -> None:
    uav_id, waypoints = load_uav_path(REPO_ROOT / "sim" / "nsoran" / "uav_paths.yaml")

    policy_path = REPO_ROOT / "sim" / "artifacts" / "policies" / f"{uav_id}-demo.json"
    policy_data = json.loads(policy_path.read_text(encoding="utf-8"))
    policy = policy_from_dict(policy_data)

    radio_map = build_synthetic_radiomap(waypoints)
    service = ServiceProfile(name="uav-hd-video", target_bitrate_mbps=10.0, min_sinr_db=-3.0)

    decisions_dir = REPO_ROOT / "sim" / "artifacts" / "decisions"
    decisions_dir.mkdir(parents=True, exist_ok=True)
    out_path = decisions_dir / f"{uav_id}-demo.jsonl"

    with out_path.open("w", encoding="utf-8") as f:
        for wp in sorted(waypoints, key=lambda w: w.index):
            cell_metrics = radio_map.metrics[wp.index]
            # For simplicity, assume cell-A is serving if its SINR >= cell-B; else B.
            if cell_metrics["cell-A"].sinr_db >= cell_metrics["cell-B"].sinr_db:
                serving = "cell-A"
                neighbor = "cell-B"
            else:
                serving = "cell-B"
                neighbor = "cell-A"

            serving_met = cell_metrics[serving]
            neighbor_met = cell_metrics[neighbor]

            uav_state = UavState(
                uav_id=uav_id,
                x=wp.x,
                y=wp.y,
                z=wp.z,
                path_position=wp.index / max(1, len(waypoints) - 1),
            )

            radio = RadioSnapshot(
                serving_cell_id=serving,
                neighbor_cell_ids=[neighbor],
                rsrp_serving=serving_met.sinr_db,
                rsrp_best_neighbor=neighbor_met.sinr_db,
                prb_utilization_serving=serving_met.load,
                prb_utilization_slice=None,
            )

            decision = path_aware_rc_policy(uav_state, radio, plan=policy, service=service)
            record = {
                "step_index": wp.index,
                "uav_id": uav_id,
                "decision": {
                    "uav_id": decision.uav_id,
                    "target_cell_id": decision.target_cell_id,
                    "slice_id": decision.slice_id,
                    "prb_quota": decision.prb_quota,
                    "reason": decision.reason,
                },
            }
            f.write(json.dumps(record) + "\n")

    print(f"[near-rt-mock] Wrote decisions to {out_path}")


if __name__ == "__main__":
    main()
