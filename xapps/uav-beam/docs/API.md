# UAV Beam Tracking xApp - REST API Reference

This document provides complete documentation for the UAV Beam Tracking xApp REST API.

## Base URL

```
http://<host>:<port>/
```

Default: `http://localhost:5001/`

## Authentication

Currently, the API does not require authentication. For production deployments, it is recommended to:

1. Deploy behind an API gateway with authentication
2. Use O-RAN A1/E2 security mechanisms
3. Implement OAuth 2.0 or JWT-based authentication

## Rate Limiting

No rate limiting is currently enforced. For production:

| Endpoint Category | Recommended Limit |
|-------------------|-------------------|
| E2 Indications | 1000 req/s per UE |
| Statistics | 10 req/s |
| Configuration | 1 req/s |

## Content Type

All requests and responses use JSON format:

```
Content-Type: application/json
```

---

## Endpoints

### Health Check

Check xApp health status.

```
GET /health
```

#### Response

```json
{
  "status": "healthy",
  "xapp": "uav-beam",
  "version": "0.1.0"
}
```

#### Status Codes

| Code | Description |
|------|-------------|
| 200 | xApp is healthy |
| 503 | xApp is unhealthy |

---

### E2 Indication

Process beam measurement indications and receive beam control decisions.

```
POST /e2/indication
```

#### Request Body

```json
{
  "ue_id": "uav-001",
  "timestamp_ms": 1700000000000,
  "serving_cell_id": "cell-1",
  "serving_beam_id": 42,
  "beam_rsrp_dbm": -85.0,
  "neighbor_beams": {
    "41": -88.0,
    "43": -87.5,
    "44": -90.0
  },
  "cqi": 12,
  "position": [100.0, 200.0, 50.0],
  "velocity": [5.0, 2.0, 0.5]
}
```

#### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `ue_id` | string | Yes | Unique identifier for the UE (UAV) |
| `timestamp_ms` | number | No | Timestamp in milliseconds (default: current time) |
| `serving_cell_id` | string | No | Serving cell identifier |
| `serving_beam_id` | integer | Yes | Current serving beam index |
| `beam_rsrp_dbm` | number | Yes | Serving beam RSRP in dBm |
| `neighbor_beams` | object | No | Map of beam_id to RSRP for neighbor beams |
| `cqi` | integer | No | Channel Quality Indicator (0-15) |
| `position` | array | No | UAV position [x, y, z] in meters |
| `velocity` | array | No | UAV velocity [vx, vy, vz] in m/s |

#### Response - Success

```json
{
  "status": "success",
  "decision": {
    "ue_id": "uav-001",
    "action": "switch",
    "current_beam_id": 42,
    "target_beam_id": 43,
    "confidence": 0.85,
    "predicted_rsrp_dbm": -82.5,
    "reason": "Predictive switch: confidence=0.85"
  }
}
```

#### Decision Actions

| Action | Description |
|--------|-------------|
| `maintain` | Keep current beam (optimal) |
| `switch` | Proactive beam switch recommended |
| `refine` | Beam refinement (P2 procedure) |
| `recover` | Beam failure recovery (P3 procedure) |

#### Response - Error

```json
{
  "status": "error",
  "error": "Missing required field: serving_beam_id"
}
```

#### Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad request (invalid JSON or missing fields) |
| 500 | Internal server error |

#### Example

```bash
curl -X POST http://localhost:5001/e2/indication \
  -H "Content-Type: application/json" \
  -d '{
    "ue_id": "uav-001",
    "serving_beam_id": 42,
    "beam_rsrp_dbm": -85.0,
    "neighbor_beams": {"41": -88.0, "43": -82.0},
    "position": [100.0, 200.0, 50.0],
    "velocity": [10.0, 5.0, 0.0]
  }'
```

---

### Angle Estimation

Estimate Angle of Arrival (AoA) / Angle of Departure (AoD) from received signal samples.

```
POST /angle/estimate
```

#### Request Body

```json
{
  "ue_id": "uav-001",
  "timestamp_ms": 1700000000000,
  "received_signal_real": [
    [0.1, 0.2, 0.15, ...],
    [0.3, 0.1, 0.25, ...],
    ...
  ],
  "received_signal_imag": [
    [0.05, -0.1, 0.08, ...],
    [-0.02, 0.15, 0.12, ...],
    ...
  ]
}
```

#### Request Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `ue_id` | string | No | UE identifier |
| `timestamp_ms` | number | No | Timestamp in milliseconds |
| `received_signal_real` | array | Yes | Real part of received signal (num_elements x num_snapshots) |
| `received_signal_imag` | array | Yes | Imaginary part of received signal |

#### Response

```json
{
  "status": "success",
  "estimate": {
    "azimuth_deg": 23.5,
    "elevation_deg": 12.3,
    "confidence": 0.92,
    "method": "MUSIC"
  }
}
```

#### Response Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `azimuth_deg` | number | Estimated azimuth angle in degrees |
| `elevation_deg` | number | Estimated elevation angle in degrees |
| `confidence` | number | Estimation confidence [0, 1] |
| `method` | string | Estimation method used |

#### Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad request |
| 500 | Internal server error |

---

### Statistics

Retrieve comprehensive xApp statistics.

```
GET /statistics
```

#### Response

```json
{
  "indications_received": 15420,
  "decisions_made": 15420,
  "beam_switches": 342,
  "start_time": 1700000000.0,
  "uptime_seconds": 3600.5,
  "indications_per_second": 4.28,
  "beam_tracker_stats": {
    "beam_switches": 342,
    "beam_failures": 12,
    "successful_recoveries": 11,
    "active_ues": 5,
    "avg_prediction_accuracy": 0.87
  },
  "angle_estimator_stats": {
    "estimates": 8540,
    "avg_confidence": 0.89
  },
  "tracked_uavs": ["uav-001", "uav-002", "uav-003", "uav-004", "uav-005"]
}
```

#### Response Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `indications_received` | integer | Total E2 indications processed |
| `decisions_made` | integer | Total beam decisions made |
| `beam_switches` | integer | Total proactive beam switches |
| `start_time` | number | xApp start timestamp (Unix) |
| `uptime_seconds` | number | xApp uptime in seconds |
| `indications_per_second` | number | Average indication rate |
| `beam_tracker_stats` | object | BeamTracker module statistics |
| `angle_estimator_stats` | object | AngleEstimator module statistics |
| `tracked_uavs` | array | List of currently tracked UAV IDs |

---

### UE State

Get detailed state information for a specific UE.

```
GET /ue/<ue_id>
```

#### Path Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `ue_id` | string | UE identifier |

#### Response - Success

```json
{
  "ue_id": "uav-001",
  "beam_state": "tracking",
  "last_beam": 43,
  "failure_count": 0,
  "predicted_position": [150.0, 225.0, 52.0],
  "predicted_velocity": [10.0, 5.0, 0.5],
  "decision_history": [
    {
      "ue_id": "uav-001",
      "timestamp_ms": 1700000000000,
      "action": "maintain",
      "current_beam_id": 43,
      "target_beam_id": 43,
      "confidence": 0.95,
      "predicted_rsrp_dbm": -83.0,
      "reason": "Current beam optimal"
    }
  ]
}
```

#### Beam States

| State | Description |
|-------|-------------|
| `idle` | Not actively tracking |
| `acquiring` | P1: Initial beam acquisition |
| `refining` | P2: Beam refinement |
| `tracking` | P3: Active beam tracking |
| `recovery` | Beam failure recovery |
| `failed` | Beam link failed |

#### Response - Not Found

```json
{
  "status": "error",
  "error": "UE uav-999 not found"
}
```

#### Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 404 | UE not found |

---

### Configuration

Get or update xApp configuration.

#### Get Configuration

```
GET /config
```

#### Response

```json
{
  "beam_config": {
    "num_beams_h": 16,
    "num_beams_v": 8,
    "total_beams": 128,
    "beam_failure_threshold_db": -10.0,
    "prediction_horizon_ms": 20.0
  },
  "predictor_config": {
    "max_prediction_horizon_ms": 500.0,
    "max_velocity": 30.0
  },
  "estimator_config": {
    "num_elements_h": 8,
    "num_elements_v": 8
  }
}
```

#### Update Configuration

```
PUT /config
```

#### Request Body

```json
{
  "beam_failure_threshold_db": -15.0,
  "prediction_horizon_ms": 30.0
}
```

#### Updatable Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `beam_failure_threshold_db` | number | Beam failure RSRP threshold |
| `prediction_horizon_ms` | number | Prediction lookahead time |

#### Response

```json
{
  "status": "success",
  "message": "Configuration updated"
}
```

---

### Reset

Reset xApp state, clearing all UE tracking data and statistics.

```
POST /reset
```

#### Response

```json
{
  "status": "success",
  "message": "xApp state reset"
}
```

#### Status Codes

| Code | Description |
|------|-------------|
| 200 | Success |

---

## Error Codes

### Application Errors

| Error | Description |
|-------|-------------|
| `Missing required field: <field>` | Required request field is missing |
| `JSON required` | Request body must be valid JSON |
| `UE <ue_id> not found` | Specified UE is not being tracked |
| `Invalid signal dimensions` | Signal array dimensions are incorrect |

### HTTP Status Codes

| Code | Meaning |
|------|---------|
| 200 | OK - Request succeeded |
| 400 | Bad Request - Invalid request format |
| 404 | Not Found - Resource not found |
| 500 | Internal Server Error - Server-side error |

---

## Example Workflows

### Basic Beam Tracking

```bash
# 1. Start xApp
uav-beam-xapp --port 5001

# 2. Check health
curl http://localhost:5001/health

# 3. Send indication
curl -X POST http://localhost:5001/e2/indication \
  -H "Content-Type: application/json" \
  -d '{"ue_id": "uav-001", "serving_beam_id": 42, "beam_rsrp_dbm": -85.0}'

# 4. Check statistics
curl http://localhost:5001/statistics

# 5. Get UE state
curl http://localhost:5001/ue/uav-001
```

### Trajectory-Based Tracking

```python
import requests
import time

base_url = "http://localhost:5001"

# Simulate UAV movement with position updates
positions = [
    ([100.0, 0.0, 50.0], [10.0, 0.0, 0.0]),   # Moving east
    ([110.0, 0.0, 50.0], [10.0, 0.0, 0.0]),
    ([120.0, 0.0, 50.0], [10.0, 5.0, 0.0]),   # Turning northeast
    ([130.0, 5.0, 50.0], [10.0, 5.0, 0.0]),
    ([140.0, 10.0, 50.0], [10.0, 5.0, 1.0]),  # Ascending
]

for pos, vel in positions:
    response = requests.post(
        f"{base_url}/e2/indication",
        json={
            "ue_id": "uav-001",
            "timestamp_ms": time.time() * 1000,
            "serving_beam_id": 42,
            "beam_rsrp_dbm": -85.0,
            "position": pos,
            "velocity": vel,
        }
    )
    decision = response.json()["decision"]
    print(f"Action: {decision['action']}, Target: {decision['target_beam_id']}")
    time.sleep(0.1)
```

### Angle Estimation

```python
import requests
import numpy as np

base_url = "http://localhost:5001"

# Generate synthetic received signal
num_elements = 64  # 8x8 array
num_snapshots = 64

# Signal with noise
signal = np.random.randn(num_elements, num_snapshots) + \
         1j * np.random.randn(num_elements, num_snapshots)

response = requests.post(
    f"{base_url}/angle/estimate",
    json={
        "ue_id": "uav-001",
        "received_signal_real": signal.real.tolist(),
        "received_signal_imag": signal.imag.tolist(),
    }
)

estimate = response.json()["estimate"]
print(f"Azimuth: {estimate['azimuth_deg']:.1f} deg")
print(f"Elevation: {estimate['elevation_deg']:.1f} deg")
print(f"Confidence: {estimate['confidence']:.2f}")
```

---

## Integration with O-RAN

### E2 Interface Mapping

| xApp Endpoint | O-RAN E2 Service |
|---------------|------------------|
| `POST /e2/indication` | E2SM-KPM Indication |
| Response decision | E2SM-RC Control |

### SDL Integration

The xApp maintains state using an internal dictionary-based store. For production O-RAN deployment:

1. Replace with SDL (Shared Data Layer) for state persistence
2. Use RMR (RIC Message Router) for E2 communications
3. Integrate with A1 interface for policy management

### Example E2SM-KPM Mapping

```
E2SM-KPM Indication Message:
  - UE-id -> ue_id
  - L1 Measurement (RSRP) -> beam_rsrp_dbm
  - Serving SS/PBCH Block Index -> serving_beam_id
  - Neighbor Cell Info -> neighbor_beams
```
