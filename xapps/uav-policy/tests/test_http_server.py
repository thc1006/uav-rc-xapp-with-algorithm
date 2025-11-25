"""Tests for the HTTP server that handles E2 indications."""

import pytest
from uav_policy.server import create_app, PolicyEngineHandler


@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app = create_app()
    app.config["TESTING"] = True
    with app.test_client() as client:
        yield client


def test_health_check_endpoint(client):
    """Test: GET /health returns 200 OK."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.get_json()
    assert data["status"] == "healthy"
    assert "timestamp" in data


def test_e2_indication_basic(client):
    """Test: POST /e2/indication with valid UAV data returns ResourceDecision."""
    indication_data = {
        "uav_id": "uav-001",
        "position": {"x": 100.0, "y": 50.0, "z": 120.0},
        "path_position": 0.5,
        "slice_id": "uav-hd-video",
        "radio_snapshot": {
            "serving_cell_id": "cell-A",
            "neighbor_cell_ids": ["cell-B", "cell-C"],
            "rsrp_serving": -88.0,
            "rsrp_best_neighbor": -82.0,
            "prb_utilization_serving": 0.75,
            "prb_utilization_slice": 0.6,
        },
    }

    response = client.post(
        "/e2/indication",
        json=indication_data,
        headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 200
    data = response.get_json()

    # Verify ResourceDecision fields
    assert data["uav_id"] == "uav-001"
    assert "target_cell_id" in data
    assert "slice_id" in data
    assert "prb_quota" in data
    assert "reason" in data
    assert isinstance(data["reason"], str)


def test_e2_indication_with_flight_plan(client):
    """Test: POST /e2/indication with flight plan follows planned cell."""
    indication_data = {
        "uav_id": "uav-001",
        "position": {"x": 200.0, "y": 100.0, "z": 150.0},
        "path_position": 0.8,
        "slice_id": None,
        "flight_plan": {
            "segments": [
                {
                    "start_pos": 0.0,
                    "end_pos": 0.5,
                    "planned_cell_id": "cell-A",
                    "slice_id": "uav-hd-video",
                    "base_prb_quota": 20,
                },
                {
                    "start_pos": 0.5,
                    "end_pos": 1.0,
                    "planned_cell_id": "cell-B",
                    "slice_id": "uav-hd-video",
                    "base_prb_quota": 30,
                },
            ]
        },
        "radio_snapshot": {
            "serving_cell_id": "cell-A",
            "neighbor_cell_ids": ["cell-B"],
            "rsrp_serving": -90.0,
            "rsrp_best_neighbor": -84.0,
            "prb_utilization_serving": 0.95,
            "prb_utilization_slice": 0.7,
        },
        "service_profile": {
            "name": "uav-hd-video",
            "target_bitrate_mbps": 10.0,
            "min_sinr_db": -5.0,
        },
    }

    response = client.post(
        "/e2/indication",
        json=indication_data,
        headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 200
    data = response.get_json()
    assert data["uav_id"] == "uav-001"
    # When serving is hot and neighbor is stronger, should follow flight plan
    assert data["target_cell_id"] == "cell-B"
    assert data["slice_id"] == "uav-hd-video"


def test_e2_indication_missing_fields(client):
    """Test: POST /e2/indication with missing required fields returns 400."""
    indication_data = {
        "uav_id": "uav-001",
        # Missing position and radio_snapshot
    }

    response = client.post(
        "/e2/indication",
        json=indication_data,
        headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data


def test_e2_indication_invalid_json(client):
    """Test: POST /e2/indication with invalid JSON returns 400."""
    response = client.post(
        "/e2/indication",
        data="not json",
        headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 400


def test_decisions_endpoint(client):
    """Test: GET /decisions returns list of recent decisions."""
    # First, submit an indication
    indication_data = {
        "uav_id": "uav-001",
        "position": {"x": 100.0, "y": 50.0, "z": 120.0},
        "path_position": 0.5,
        "slice_id": None,
        "radio_snapshot": {
            "serving_cell_id": "cell-A",
            "neighbor_cell_ids": ["cell-B"],
            "rsrp_serving": -88.0,
            "rsrp_best_neighbor": -82.0,
            "prb_utilization_serving": 0.75,
            "prb_utilization_slice": 0.6,
        },
    }

    client.post("/e2/indication", json=indication_data)

    # Then retrieve decisions
    response = client.get("/decisions")
    assert response.status_code == 200
    data = response.get_json()
    assert "decisions" in data
    assert isinstance(data["decisions"], list)
    assert len(data["decisions"]) >= 1


def test_e2_indication_with_service_profile(client):
    """Test: POST /e2/indication respects service profile QoS."""
    indication_data = {
        "uav_id": "uav-002",
        "position": {"x": 150.0, "y": 75.0, "z": 100.0},
        "path_position": 0.3,
        "slice_id": None,
        "radio_snapshot": {
            "serving_cell_id": "cell-A",
            "neighbor_cell_ids": ["cell-B"],
            "rsrp_serving": -85.0,
            "rsrp_best_neighbor": -80.0,
            "prb_utilization_serving": 0.5,
            "prb_utilization_slice": None,
        },
        "service_profile": {
            "name": "video-streaming",
            "target_bitrate_mbps": 5.0,
            "min_sinr_db": -10.0,
        },
    }

    response = client.post(
        "/e2/indication",
        json=indication_data,
        headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 200
    data = response.get_json()
    # PRB quota should be computed to meet service requirements
    assert data["prb_quota"] is not None
    assert data["prb_quota"] >= 5


def test_policy_engine_handler_parse_indication():
    """Test: PolicyEngineHandler.parse_indication correctly parses E2 data."""
    indication_json = {
        "uav_id": "uav-001",
        "position": {"x": 100.0, "y": 50.0, "z": 120.0},
        "path_position": 0.5,
        "slice_id": "uav-hd-video",
        "radio_snapshot": {
            "serving_cell_id": "cell-A",
            "neighbor_cell_ids": ["cell-B"],
            "rsrp_serving": -88.0,
            "rsrp_best_neighbor": -82.0,
            "prb_utilization_serving": 0.75,
            "prb_utilization_slice": 0.6,
        },
    }

    handler = PolicyEngineHandler()
    uav_state, radio_snapshot = handler.parse_indication(indication_json)

    assert uav_state.uav_id == "uav-001"
    assert uav_state.x == 100.0
    assert uav_state.y == 50.0
    assert uav_state.z == 120.0
    assert uav_state.path_position == 0.5
    assert uav_state.slice_id == "uav-hd-video"

    assert radio_snapshot.serving_cell_id == "cell-A"
    assert radio_snapshot.neighbor_cell_ids == ["cell-B"]
    assert radio_snapshot.rsrp_serving == -88.0
    assert radio_snapshot.rsrp_best_neighbor == -82.0


def test_multiple_uavs_independent_decisions(client):
    """Test: Multiple UAVs get independent decisions."""
    for uav_id in ["uav-001", "uav-002", "uav-003"]:
        indication_data = {
            "uav_id": uav_id,
            "position": {"x": 100.0, "y": 50.0, "z": 120.0},
            "path_position": 0.5,
            "slice_id": None,
            "radio_snapshot": {
                "serving_cell_id": "cell-A",
                "neighbor_cell_ids": ["cell-B"],
                "rsrp_serving": -88.0,
                "rsrp_best_neighbor": -82.0,
                "prb_utilization_serving": 0.75,
                "prb_utilization_slice": 0.6,
            },
        }

        response = client.post("/e2/indication", json=indication_data)
        assert response.status_code == 200
        data = response.get_json()
        assert data["uav_id"] == uav_id

    # Verify all decisions are recorded
    response = client.get("/decisions")
    assert response.status_code == 200
    data = response.get_json()
    assert len(data["decisions"]) >= 3
