from uav_policy.policy_engine import ServiceProfile
from uav_path_planner.planner import (
    Waypoint,
    CellMetric,
    RadioMap,
    PlannerConfig,
    plan_flight_path,
)


def test_plan_flight_path_produces_segments():
    waypoints = [
        Waypoint(index=0, x=0.0, y=0.0, z=50.0),
        Waypoint(index=1, x=50.0, y=0.0, z=60.0),
        Waypoint(index=2, x=100.0, y=20.0, z=80.0),
    ]

    metrics = {
        0: {"cell-A": CellMetric(sinr_db=-1.0, load=0.3)},
        1: {"cell-A": CellMetric(sinr_db=0.0, load=0.4)},
        2: {"cell-A": CellMetric(sinr_db=1.0, load=0.2)},
    }

    radio_map = RadioMap(metrics=metrics)
    service = ServiceProfile(name="uav-hd-video", target_bitrate_mbps=8.0, min_sinr_db=-3.0)
    config = PlannerConfig(sinr_min_db=service.min_sinr_db)

    policy = plan_flight_path("uav-001", waypoints, radio_map, service, config)
    assert policy.segments
    seg = policy.segments[0]
    assert 0.0 <= seg.start_pos <= seg.end_pos <= 1.0
    assert seg.base_prb_quota >= 5
