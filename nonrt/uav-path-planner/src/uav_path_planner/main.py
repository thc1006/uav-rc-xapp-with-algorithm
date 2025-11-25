"""CLI demo for the Non-RT path planner."""

import json
from pathlib import Path

from uav_policy.policy_engine import ServiceProfile
from uav_path_planner.planner import (
    Waypoint,
    CellMetric,
    RadioMap,
    PlannerConfig,
    plan_flight_path,
    policy_to_dict,
)


def build_demo_inputs():
    waypoints = [
        Waypoint(index=0, x=0.0, y=0.0, z=50.0),
        Waypoint(index=1, x=50.0, y=0.0, z=60.0),
        Waypoint(index=2, x=100.0, y=20.0, z=80.0),
        Waypoint(index=3, x=150.0, y=40.0, z=90.0),
    ]

    metrics = {
        0: {
            "cell-A": CellMetric(sinr_db=-2.0, load=0.3),
            "cell-B": CellMetric(sinr_db=-4.0, load=0.1),
        },
        1: {
            "cell-A": CellMetric(sinr_db=0.0, load=0.5),
            "cell-B": CellMetric(sinr_db=2.0, load=0.2),
        },
        2: {
            "cell-A": CellMetric(sinr_db=-1.0, load=0.7),
            "cell-B": CellMetric(sinr_db=3.0, load=0.4),
        },
        3: {
            "cell-A": CellMetric(sinr_db=-3.0, load=0.6),
            "cell-B": CellMetric(sinr_db=1.0, load=0.3),
        },
    }

    radio_map = RadioMap(metrics=metrics)
    service = ServiceProfile(name="uav-hd-video", target_bitrate_mbps=10.0, min_sinr_db=-3.0)
    config = PlannerConfig(sinr_min_db=service.min_sinr_db)

    return waypoints, radio_map, service, config


def main() -> None:
    uav_id = "uav-001"
    waypoints, radio_map, service, config = build_demo_inputs()
    policy = plan_flight_path(uav_id, waypoints, radio_map, service, config)

    out = policy_to_dict(policy)
    print(json.dumps(out, indent=2))

    # Optionally write to sim artifacts location if present
    repo_root = Path(__file__).resolve().parents[3]
    artifacts_dir = repo_root / "sim" / "artifacts" / "policies"
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    (artifacts_dir / f"{uav_id}-demo.json").write_text(
        json.dumps(out, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
