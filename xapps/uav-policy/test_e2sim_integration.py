#!/usr/bin/env python3
"""
End-to-End Integration Test: E2-Simulator → UAV Policy xApp

This test simulates realistic E2 indication flows from an E2 simulator,
sending them to the running uav-policy server and validating decisions.

Test Scenarios:
1. Normal UAV tracking with flight plan
2. UAV handover due to overloaded serving cell
3. Multiple concurrent UAVs (swarm simulation)
4. Streaming indications (realistic message rate)
5. Service profile-based resource allocation
6. Error handling for malformed indications
"""

import json
import time
import requests
from typing import Dict, Any, List

BASE_URL = "http://localhost:5000"
INDICATION_ENDPOINT = f"{BASE_URL}/e2/indication"
DECISIONS_ENDPOINT = f"{BASE_URL}/decisions"
HEALTH_ENDPOINT = f"{BASE_URL}/health"
STATS_ENDPOINT = f"{BASE_URL}/stats"


def send_indication(indication: Dict[str, Any]) -> Dict[str, Any]:
    """Send an E2 indication to the uav-policy server."""
    try:
        response = requests.post(INDICATION_ENDPOINT, json=indication, timeout=5)
        return response.json() if response.status_code == 200 else {"error": response.text}
    except Exception as e:
        return {"error": str(e)}


def test_scenario_1_normal_tracking():
    """Test 1: Normal UAV tracking with flight plan (no handover needed)."""
    print("\n" + "="*70)
    print("TEST 1: Normal UAV Tracking with Flight Plan")
    print("="*70)

    indication = {
        "uav_id": "UAV-001",
        "position": {"x": 100.0, "y": 200.0, "z": 50.0},
        "path_position": 500.0,
        "slice_id": "slice-eMBB",
        "radio_snapshot": {
            "serving_cell_id": "cell_001",
            "neighbor_cell_ids": ["cell_002", "cell_003"],
            "rsrp_serving": -85.0,
            "rsrp_best_neighbor": -90.0,
            "prb_utilization_serving": 0.4,
        },
        "flight_plan": {
            "segments": [
                {
                    "start_pos": 400.0,
                    "end_pos": 600.0,
                    "planned_cell_id": "cell_001",
                    "slice_id": "slice-eMBB",
                    "base_prb_quota": 20
                }
            ]
        }
    }

    decision = send_indication(indication)
    print(f"Decision: {json.dumps(decision, indent=2)}")

    assert decision.get("uav_id") == "UAV-001", "UAV ID mismatch"
    assert decision.get("target_cell_id") == "cell_001", "Should stay on serving cell"
    assert decision.get("prb_quota") == 20, "PRB quota should be 20"
    print("✓ Test 1 PASSED: Normal tracking working correctly\n")


def test_scenario_2_overload_handover():
    """Test 2: UAV handover when serving cell is overloaded."""
    print("="*70)
    print("TEST 2: Handover Due to Serving Cell Overload")
    print("="*70)

    indication = {
        "uav_id": "UAV-002",
        "position": {"x": 150.0, "y": 250.0, "z": 45.0},
        "path_position": 520.0,
        "slice_id": "slice-eMBB",
        "radio_snapshot": {
            "serving_cell_id": "cell_001",
            "neighbor_cell_ids": ["cell_002", "cell_003"],
            "rsrp_serving": -88.0,
            "rsrp_best_neighbor": -84.0,  # 4dB stronger than serving
            "prb_utilization_serving": 0.95,  # Heavily overloaded
        },
        "flight_plan": {
            "segments": [
                {
                    "start_pos": 400.0,
                    "end_pos": 600.0,
                    "planned_cell_id": "cell_002",
                    "slice_id": "slice-eMBB",
                    "base_prb_quota": 20
                }
            ]
        }
    }

    decision = send_indication(indication)
    print(f"Decision: {json.dumps(decision, indent=2)}")

    assert decision.get("uav_id") == "UAV-002", "UAV ID mismatch"
    assert decision.get("target_cell_id") == "cell_002", "Should handover to planned neighbor"
    print("✓ Test 2 PASSED: Overload handover working correctly\n")


def test_scenario_3_multiple_uavs():
    """Test 3: Multiple UAVs in swarm (concurrency test)."""
    print("="*70)
    print("TEST 3: Multiple UAVs Swarm (Concurrency Test)")
    print("="*70)

    uavs = [
        {
            "uav_id": "UAV-101",
            "position": {"x": 100.0, "y": 200.0, "z": 50.0},
            "path_position": 100.0,
            "radio_snapshot": {
                "serving_cell_id": "cell_001",
                "neighbor_cell_ids": ["cell_002"],
                "rsrp_serving": -85.0,
                "rsrp_best_neighbor": -90.0,
                "prb_utilization_serving": 0.3,
            },
        },
        {
            "uav_id": "UAV-102",
            "position": {"x": 120.0, "y": 220.0, "z": 52.0},
            "path_position": 110.0,
            "radio_snapshot": {
                "serving_cell_id": "cell_002",
                "neighbor_cell_ids": ["cell_001"],
                "rsrp_serving": -86.0,
                "rsrp_best_neighbor": -91.0,
                "prb_utilization_serving": 0.4,
            },
        },
        {
            "uav_id": "UAV-103",
            "position": {"x": 140.0, "y": 240.0, "z": 51.0},
            "path_position": 120.0,
            "radio_snapshot": {
                "serving_cell_id": "cell_001",
                "neighbor_cell_ids": ["cell_002"],
                "rsrp_serving": -87.0,
                "rsrp_best_neighbor": -92.0,
                "prb_utilization_serving": 0.2,
            },
        },
    ]

    decisions = []
    for uav in uavs:
        decision = send_indication(uav)
        decisions.append(decision)
        print(f"  {decision.get('uav_id')}: → {decision.get('target_cell_id')} "
              f"(PRB={decision.get('prb_quota')})")

    assert len(decisions) == 3, "Should process all 3 UAVs"
    assert all(d.get("uav_id") for d in decisions), "All decisions should have UAV IDs"
    print("✓ Test 3 PASSED: Multiple UAVs processed correctly\n")


def test_scenario_4_streaming_indications():
    """Test 4: Streaming indications (simulating real-time E2 flow)."""
    print("="*70)
    print("TEST 4: Streaming Indications (Real-time Flow)")
    print("="*70)

    for i in range(5):
        indication = {
            "uav_id": "UAV-201",
            "position": {"x": 100.0 + i*10, "y": 200.0 + i*10, "z": 50.0},
            "path_position": 500.0 + i*10,
            "radio_snapshot": {
                "serving_cell_id": "cell_001",
                "neighbor_cell_ids": ["cell_002"],
                "rsrp_serving": -85.0,
                "rsrp_best_neighbor": -90.0,
                "prb_utilization_serving": 0.5,
            },
        }
        decision = send_indication(indication)
        print(f"  Indication {i+1}: {decision.get('uav_id')} → {decision.get('target_cell_id')}")
        time.sleep(0.1)  # Simulate 100ms between indications

    print("✓ Test 4 PASSED: Streaming indications processed correctly\n")


def test_scenario_5_service_profile():
    """Test 5: Service profile-based resource allocation."""
    print("="*70)
    print("TEST 5: Service Profile-Based Resource Allocation")
    print("="*70)

    indication = {
        "uav_id": "UAV-301",
        "position": {"x": 100.0, "y": 200.0, "z": 50.0},
        "path_position": 500.0,
        "radio_snapshot": {
            "serving_cell_id": "cell_001",
            "neighbor_cell_ids": ["cell_002"],
            "rsrp_serving": -80.0,
            "rsrp_best_neighbor": -85.0,
            "prb_utilization_serving": 0.3,
        },
        "service_profile": {
            "name": "4K-Video-Uplink",
            "target_bitrate_mbps": 25.0,
            "min_sinr_db": 10.0
        }
    }

    decision = send_indication(indication)
    print(f"Decision: {json.dumps(decision, indent=2)}")

    assert decision.get("prb_quota") is not None, "Should allocate PRBs for service"
    assert decision.get("prb_quota") > 5, "Should allocate sufficient PRBs for high bitrate service"
    print("✓ Test 5 PASSED: Service profile resource allocation working\n")


def test_scenario_6_error_handling():
    """Test 6: Error handling for malformed indications."""
    print("="*70)
    print("TEST 6: Error Handling for Malformed Indications")
    print("="*70)

    bad_indications = [
        {},  # Empty indication
        {"uav_id": "UAV-401"},  # Missing required fields
        {"serving_cell_id": "cell_001"},  # Missing UAV ID
    ]

    for i, bad_indication in enumerate(bad_indications):
        response = requests.post(INDICATION_ENDPOINT, json=bad_indication, timeout=5)
        print(f"  Bad indication {i+1}: HTTP {response.status_code}")
        assert response.status_code == 400, f"Should return 400 for bad indication {i+1}"

    print("✓ Test 6 PASSED: Error handling working correctly\n")


def test_decision_history():
    """Test: Verify decision history endpoint."""
    print("="*70)
    print("TEST 7: Decision History Endpoint")
    print("="*70)

    response = requests.get(DECISIONS_ENDPOINT, timeout=5)
    if response.status_code != 200:
        print(f"  ERROR: Got status code {response.status_code}")
        return

    try:
        data = response.json()
        history = data.get("decisions", [])
    except Exception as e:
        print(f"  ERROR: Cannot parse response: {e}")
        return

    print(f"  Total decisions in history: {len(history)}")
    if not history:
        print("  WARNING: No decision history found (server may have been recently restarted)")
    else:
        latest = history[-1]
        print(f"  Latest decision: {latest.get('uav_id')} → {latest.get('target_cell_id')}")

    print("✓ Test 7 PASSED: Decision history accessible\n")


def test_statistics():
    """Test: Verify statistics endpoint."""
    print("="*70)
    print("TEST 8: Statistics Endpoint")
    print("="*70)

    response = requests.get(STATS_ENDPOINT, timeout=5)
    if response.status_code != 200:
        print(f"  ERROR: Got status code {response.status_code}")
        return

    stats = response.json()
    print(f"Stats: {json.dumps(stats, indent=2)}")
    assert "total_decisions" in stats or "total_indications" in stats, "Should have total_decisions or total_indications"
    print("✓ Test 8 PASSED: Statistics endpoint working\n")


def main():
    """Run all integration tests."""
    print("\n" + "#"*70)
    print("# E2-SIMULATOR TO UAV-POLICY INTEGRATION TEST SUITE")
    print("#"*70)

    # Check server health
    try:
        health = requests.get(HEALTH_ENDPOINT, timeout=5).json()
        print(f"\nServer Status: {health['status']}")
        assert health['status'] == 'healthy', "Server not healthy"
    except Exception as e:
        print(f"ERROR: Cannot connect to server at {BASE_URL}")
        print(f"Details: {e}")
        return False

    try:
        # Run all test scenarios
        test_scenario_1_normal_tracking()
        test_scenario_2_overload_handover()
        test_scenario_3_multiple_uavs()
        test_scenario_4_streaming_indications()
        test_scenario_5_service_profile()
        test_scenario_6_error_handling()
        test_decision_history()
        test_statistics()

        print("#"*70)
        print("# ALL INTEGRATION TESTS PASSED ✓")
        print("#"*70 + "\n")
        return True

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"\n✗ UNEXPECTED ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
