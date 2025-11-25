from uav_policy.policy_engine import (
    UavState,
    RadioSnapshot,
    PathSegmentPlan,
    FlightPlanPolicy,
    ServiceProfile,
    simple_path_aware_policy,
    path_aware_rc_policy,
)


def test_simple_policy_prefers_neighbor_when_serving_is_hot():
    uav = UavState(uav_id="uav-001", x=0.0, y=0.0, z=100.0)
    radio = RadioSnapshot(
        serving_cell_id="cell-A",
        neighbor_cell_ids=["cell-B"],
        rsrp_serving=-90.0,
        rsrp_best_neighbor=-84.0,
        prb_utilization_serving=0.95,
        prb_utilization_slice=0.8,
    )

    decision = simple_path_aware_policy(uav, radio)
    assert decision.target_cell_id == "cell-B"


def test_path_aware_policy_follows_flight_plan_when_serving_hot():
    plan = FlightPlanPolicy(
        uav_id="uav-001",
        segments=[
            PathSegmentPlan(
                start_pos=0.0,
                end_pos=0.5,
                planned_cell_id="cell-A",
                slice_id="uav-hd-video",
                base_prb_quota=20,
            ),
            PathSegmentPlan(
                start_pos=0.5,
                end_pos=1.0,
                planned_cell_id="cell-B",
                slice_id="uav-hd-video",
                base_prb_quota=30,
            ),
        ],
    )

    uav = UavState(
        uav_id="uav-001",
        x=100.0,
        y=50.0,
        z=120.0,
        slice_id=None,
        path_position=0.8,
    )
    radio = RadioSnapshot(
        serving_cell_id="cell-A",
        neighbor_cell_ids=["cell-B"],
        rsrp_serving=-90.0,
        rsrp_best_neighbor=-84.0,
        prb_utilization_serving=0.95,
        prb_utilization_slice=0.7,
    )

    service = ServiceProfile(name="uav-hd-video", target_bitrate_mbps=10.0, min_sinr_db=-5.0)

    decision = path_aware_rc_policy(uav, radio, plan=plan, service=service)

    assert decision.target_cell_id == "cell-B"
    assert decision.slice_id == "uav-hd-video"
    assert decision.prb_quota is not None
    assert decision.prb_quota >= 30


def test_path_aware_policy_stays_on_serving_when_load_ok():
    plan = FlightPlanPolicy(
        uav_id="uav-001",
        segments=[
            PathSegmentPlan(
                start_pos=0.0,
                end_pos=1.0,
                planned_cell_id="cell-B",
                slice_id="uav-hd-video",
                base_prb_quota=20,
            ),
        ],
    )

    uav = UavState(
        uav_id="uav-001",
        x=0.0,
        y=0.0,
        z=100.0,
        path_position=0.2,
    )
    radio = RadioSnapshot(
        serving_cell_id="cell-A",
        neighbor_cell_ids=["cell-B"],
        rsrp_serving=-88.0,
        rsrp_best_neighbor=-86.0,
        prb_utilization_serving=0.3,
        prb_utilization_slice=0.2,
    )

    decision = path_aware_rc_policy(uav, radio, plan=plan, service=None)

    assert decision.target_cell_id == "cell-A"
    assert decision.prb_quota >= 5
