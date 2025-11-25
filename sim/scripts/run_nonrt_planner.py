"""Run the Non-RT DP planner and write a FlightPlanPolicy JSON artifact.

This script is a glue layer between:

- `sim/nsoran/uav_paths.yaml`          -> Waypoints
- a simple synthetic RadioMap          -> RadioMap
- `nonrt/uav-path-planner`             -> FlightPlanPolicy
- `sim/artifacts/policies/uav-001-demo.json`

In a real setup, you would replace the synthetic RadioMap with one
derived from propagation maps, ns-3 traces, or an external planning
tool.
"""

import json
import sys
from pathlib import Path

import yaml  # type: ignore

# Adjust sys.path so we can import local packages without installation.
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO_ROOT / "nonrt" / "uav-path-planner" / "src"))
sys.path.append(str(REPO_ROOT / "xapps" / "uav-policy" / "src"))

from uav_policy.policy_engine import ServiceProfile  # noqa: E402
from uav_path_planner.planner import (  # noqa: E402
    Waypoint,
    CellMetric,
    RadioMap,
    PlannerConfig,
    plan_flight_path,
    policy_to_dict,
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
        # Simple synthetic pattern: cell-A is better at low x, cell-B at high x.
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
    uav_paths_yaml = REPO_ROOT / "sim" / "nsoran" / "uav_paths.yaml"
    uav_id, waypoints = load_uav_path(uav_paths_yaml)
    radio_map = build_synthetic_radiomap(waypoints)

    service = ServiceProfile(name="uav-hd-video", target_bitrate_mbps=10.0, min_sinr_db=-3.0)
    config = PlannerConfig(sinr_min_db=service.min_sinr_db)

    policy = plan_flight_path(uav_id, waypoints, radio_map, service, config)
    policy_dict = policy_to_dict(policy)

    out_dir = REPO_ROOT / "sim" / "artifacts" / "policies"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{uav_id}-demo.json"
    out_path.write_text(json.dumps(policy_dict, indent=2), encoding="utf-8")

    print(f"[nonrt] Wrote FlightPlanPolicy for {uav_id} to {out_path}")


if __name__ == "__main__":
    main()
