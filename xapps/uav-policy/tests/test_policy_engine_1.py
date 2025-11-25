from uav_policy.policy_engine import UavState, RadioSnapshot, simple_path_aware_policy


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
