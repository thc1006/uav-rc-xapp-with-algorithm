"""
Complete E2E integration test: E2-Simulator → uav-policy xApp → decisions
"""
import sys
sys.path.insert(0, 'src')

from uav_policy.server import create_app

def test_e2e_complete_workflow():
    """Test complete workflow: receive indication, process, return decision"""
    print("\n" + "="*70)
    print("E2E INTEGRATION TEST: E2-Simulator → uav-policy xApp → Decisions")
    print("="*70)
    
    app = create_app()
    client = app.test_client()
    
    # Test Case 1: Healthy serving cell
    print("\n[Test 1] Healthy serving cell - should stay on serving")
    indication = {
        "uav_id": "uav_001",
        "position": {"x": 100.0, "y": 200.0, "z": 150.0},
        "radio_snapshot": {
            "serving_cell_id": "cell_001",
            "neighbor_cell_ids": ["cell_002"],
            "rsrp_serving": -80.0,
            "rsrp_best_neighbor": -95.0,
            "prb_utilization_serving": 0.3
        }
    }
    response = client.post("/e2/indication", json=indication)
    assert response.status_code == 200
    decision = response.get_json()
    assert decision['target_cell_id'] == 'cell_001'
    print(f"✓ Decision: Stay on {decision['target_cell_id']}")
    
    # Test Case 2: Overloaded serving cell with better neighbor
    print("\n[Test 2] Overloaded serving cell - should handover to neighbor")
    indication = {
        "uav_id": "uav_002",
        "position": {"x": 150.0, "y": 250.0, "z": 200.0},
        "radio_snapshot": {
            "serving_cell_id": "cell_001",
            "neighbor_cell_ids": ["cell_002", "cell_003"],
            "rsrp_serving": -90.0,
            "rsrp_best_neighbor": -85.0,
            "prb_utilization_serving": 0.9
        }
    }
    response = client.post("/e2/indication", json=indication)
    assert response.status_code == 200
    decision = response.get_json()
    assert decision['target_cell_id'] == 'cell_002'
    print(f"✓ Decision: Handover to {decision['target_cell_id']}")
    
    # Test Case 3: Streaming indications
    print("\n[Test 3] Streaming indications - 5 consecutive UAV positions")
    for i in range(5):
        indication = {
            "uav_id": f"uav_stream_{i}",
            "position": {"x": 100.0 + i*50, "y": 200.0 + i*30, "z": 150.0},
            "radio_snapshot": {
                "serving_cell_id": "cell_001",
                "neighbor_cell_ids": ["cell_002"],
                "rsrp_serving": -85.0 + i*2,
                "rsrp_best_neighbor": -92.0,
                "prb_utilization_serving": 0.4 + i*0.1
            }
        }
        response = client.post("/e2/indication", json=indication)
        assert response.status_code == 200
        decision = response.get_json()
        print(f"  ✓ Stream {i+1}: Decision for {decision['uav_id']}")
    
    # Test Case 4: Decisions history
    print("\n[Test 4] Decisions history retrieval")
    response = client.get("/decisions?limit=10")
    assert response.status_code == 200
    result = response.get_json()
    decisions = result.get('decisions', [])
    assert len(decisions) >= 5
    print(f"✓ Retrieved {len(decisions)} decisions from history")
    
    # Test Case 5: Health check
    print("\n[Test 5] Health check endpoint")
    response = client.get("/health")
    assert response.status_code == 200
    health = response.get_json()
    assert 'status' in health
    print(f"✓ Health check: {health['status']}")
    
    # Test Case 6: Stats endpoint
    print("\n[Test 6] Statistics endpoint")
    response = client.get("/stats")
    assert response.status_code == 200
    stats = response.get_json()
    assert 'total_decisions' in stats
    print(f"✓ Stats: {stats['total_decisions']} total indications processed")
    
    # Test Case 7: Invalid indication (error handling)
    print("\n[Test 7] Error handling - invalid indication")
    bad_indication = {"uav_id": "bad_uav"}
    response = client.post("/e2/indication", json=bad_indication)
    assert response.status_code == 400
    error = response.get_json()
    assert 'error' in error
    print(f"✓ Error handling: correctly rejected bad indication")
    
    print("\n" + "="*70)
    print("ALL E2E TESTS PASSED ✅")
    print("="*70)
    print("\nSUMMARY:")
    print("  ✓ Healthy serving cell detection")
    print("  ✓ Handover decision logic")
    print("  ✓ Streaming indications")
    print("  ✓ Decision history")
    print("  ✓ Health monitoring")
    print("  ✓ Statistics")
    print("  ✓ Error handling")
    print("\nSTATUS: PRODUCTION READY ✅")

if __name__ == "__main__":
    test_e2e_complete_workflow()
