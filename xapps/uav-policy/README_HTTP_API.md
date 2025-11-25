# UAV Policy xApp - HTTP API Server

This document describes the HTTP API for the UAV Policy xApp, which receives E2 indications and generates resource allocation decisions.

## Overview

The UAV Policy xApp is a Flask-based HTTP server that:
1. Receives E2 indications containing UAV position and radio measurements
2. Parses indication data into structured objects (UavState, RadioSnapshot)
3. Applies path-aware, QoS-aware policy algorithms to generate resource decisions
4. Maintains decision history for auditing and analysis

## Architecture

```
E2 Simulator/Controller
        |
        v
    /e2/indication (POST)
        |
        v
PolicyEngineHandler (parses indication)
        |
        v
path_aware_rc_policy (generates decision)
        |
        v
ResourceDecision (response)
```

## API Endpoints

### Health Check

**GET /health**

Health check endpoint for monitoring and Kubernetes probes.

Response:
```json
{
  "status": "healthy",
  "timestamp": "2025-11-21T12:34:56.789123",
  "service": "uav-policy-xapp"
}
```

Status: 200 OK

### E2 Indication Handler

**POST /e2/indication**

Main endpoint for receiving E2 indications and returning resource decisions.

Request:
```json
{
  "uav_id": "uav-001",
  "position": {
    "x": 100.0,
    "y": 50.0,
    "z": 120.0
  },
  "path_position": 0.5,
  "slice_id": "uav-hd-video",
  "radio_snapshot": {
    "serving_cell_id": "cell-A",
    "neighbor_cell_ids": ["cell-B", "cell-C"],
    "rsrp_serving": -88.0,
    "rsrp_best_neighbor": -82.0,
    "prb_utilization_serving": 0.75,
    "prb_utilization_slice": 0.6
  },
  "flight_plan": {
    "segments": [
      {
        "start_pos": 0.0,
        "end_pos": 0.5,
        "planned_cell_id": "cell-A",
        "slice_id": "uav-hd-video",
        "base_prb_quota": 20
      },
      {
        "start_pos": 0.5,
        "end_pos": 1.0,
        "planned_cell_id": "cell-B",
        "slice_id": "uav-hd-video",
        "base_prb_quota": 30
      }
    ]
  },
  "service_profile": {
    "name": "uav-hd-video",
    "target_bitrate_mbps": 10.0,
    "min_sinr_db": -5.0
  }
}
```

Response (Success):
```json
{
  "uav_id": "uav-001",
  "target_cell_id": "cell-B",
  "slice_id": "uav-hd-video",
  "prb_quota": 35,
  "reason": "Follow flight-plan cell; serving overloaded and neighbor stronger. Serving cell matches flight-plan segment. Using UAV slice_id=uav-hd-video. Estimating PRB from best-neighbor RSRP as SINR proxy. Service 'uav-hd-video' targets 10.00 Mbps; estimated required PRB quota ≈ 35.",
  "timestamp": "2025-11-21T12:34:56.789123"
}
```

Status: 200 OK

Response (Error):
```json
{
  "error": "Invalid indication data: Invalid position data: 'x' required"
}
```

Status: 400 Bad Request

**Field Descriptions:**

**Request Fields:**
- `uav_id` (required): Unique identifier for the UAV
- `position` (required): UAV 3D coordinates
  - `x`, `y`, `z`: Float values in meters
- `path_position` (optional): Normalized position on pre-planned path (0.0 to 1.0)
- `slice_id` (optional): Network slice identifier
- `radio_snapshot` (required): Current radio measurements
  - `serving_cell_id` (required): ID of current serving cell
  - `neighbor_cell_ids` (optional): List of neighbor cell IDs
  - `rsrp_serving` (required): Reference Signal Received Power on serving cell (dBm)
  - `rsrp_best_neighbor` (required): Best RSRP among neighbors (dBm)
  - `prb_utilization_serving` (required): PRB utilization on serving cell (0.0-1.0)
  - `prb_utilization_slice` (optional): PRB utilization in slice (0.0-1.0)
- `flight_plan` (optional): Pre-planned flight path with cell assignments
  - `segments`: Array of path segments with planned resource allocations
    - `start_pos`: Path segment start (0.0-1.0)
    - `end_pos`: Path segment end (0.0-1.0)
    - `planned_cell_id`: Cell to serve this segment
    - `slice_id`: Network slice for this segment
    - `base_prb_quota`: Base PRB quota for this segment
- `service_profile` (optional): QoS requirements
  - `name`: Service name
  - `target_bitrate_mbps`: Required bitrate in Mbps
  - `min_sinr_db`: Minimum SINR in dB

**Response Fields:**
- `uav_id`: UAV identifier (echoed)
- `target_cell_id`: Recommended cell for resource allocation
- `slice_id`: Network slice (if determined)
- `prb_quota`: Allocated PRB quota
- `reason`: Explanation of the decision (for auditing)
- `timestamp`: Decision timestamp

### Get Recent Decisions

**GET /decisions**

Retrieve recent decisions for auditing and analysis.

Query Parameters:
- `limit` (optional, default=100, max=1000): Maximum number of decisions to return

Response:
```json
{
  "decisions": [
    {
      "timestamp": "2025-11-21T12:34:56.789123",
      "uav_id": "uav-001",
      "target_cell_id": "cell-B",
      "slice_id": "uav-hd-video",
      "prb_quota": 35,
      "reason": "..."
    }
  ],
  "count": 1,
  "timestamp": "2025-11-21T12:34:57.123456"
}
```

Status: 200 OK

### Server Statistics

**GET /stats**

Get server statistics and operational metrics.

Response:
```json
{
  "total_decisions": 42,
  "unique_uavs": 3,
  "uav_list": ["uav-001", "uav-002", "uav-003"],
  "timestamp": "2025-11-21T12:34:57.123456"
}
```

Status: 200 OK

## Error Handling

The API returns standard HTTP status codes:

- **200 OK**: Request processed successfully
- **400 Bad Request**: Invalid request format or missing required fields
- **404 Not Found**: Endpoint not found
- **405 Method Not Allowed**: Wrong HTTP method for endpoint
- **500 Internal Server Error**: Unexpected server error

All error responses include a JSON body with an `error` field describing the issue.

## Running the Server

### Locally (Development)

```bash
# Install dependencies
pip install flask werkzeug

# Set environment variables
export LOG_LEVEL=INFO
export SERVER_HOST=0.0.0.0
export SERVER_PORT=5000
export DEBUG=false

# Run server
python3 -m uav_policy.main

# Or directly
python3 -c "from uav_policy.server import create_app; app = create_app(); app.run(host='0.0.0.0', port=5000)"
```

### With Docker

```bash
# Build image
docker build -t uav-policy-xapp:latest .

# Run container
docker run -d \
  --name uav-policy \
  -p 5000:5000 \
  -e LOG_LEVEL=INFO \
  -e SERVER_HOST=0.0.0.0 \
  -e SERVER_PORT=5000 \
  uav-policy-xapp:latest

# Check logs
docker logs -f uav-policy
```

### On Kubernetes

```bash
# Create namespace
kubectl create namespace oran-ric

# Apply all resources using kustomization
kubectl apply -k k8s/

# Or apply individually
kubectl apply -f k8s/serviceaccount.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/deployment.yaml
kubectl apply -f k8s/service.yaml

# Check deployment status
kubectl get deployment -n oran-ric uav-policy-xapp
kubectl get pods -n oran-ric -l app=uav-policy-xapp
kubectl logs -f -n oran-ric -l app=uav-policy-xapp

# Port forward for testing
kubectl port-forward -n oran-ric svc/uav-policy-xapp 5000:5000
```

## Testing

### Unit Tests

Run the test suite:

```bash
# Set PYTHONPATH for imports
export PYTHONPATH="$(pwd)/src"

# Run all tests
python3 -m pytest tests/ -v

# Run with coverage
python3 -m pytest tests/ -v --cov=src/uav_policy --cov-report=term-missing

# Run specific test
python3 -m pytest tests/test_http_server.py::test_health_check_endpoint -v
```

### Integration Tests

Send sample requests to the server:

```bash
# Health check
curl -X GET http://localhost:5000/health

# Submit E2 indication
curl -X POST http://localhost:5000/e2/indication \
  -H "Content-Type: application/json" \
  -d '{
    "uav_id": "uav-001",
    "position": {"x": 100.0, "y": 50.0, "z": 120.0},
    "path_position": 0.5,
    "radio_snapshot": {
      "serving_cell_id": "cell-A",
      "neighbor_cell_ids": ["cell-B"],
      "rsrp_serving": -88.0,
      "rsrp_best_neighbor": -82.0,
      "prb_utilization_serving": 0.75,
      "prb_utilization_slice": 0.6
    }
  }'

# Get recent decisions
curl -X GET http://localhost:5000/decisions?limit=10

# Get statistics
curl -X GET http://localhost:5000/stats
```

## Performance Considerations

1. **Decision History Limit**: The handler maintains up to 1000 decisions in memory. Older decisions are discarded.

2. **Threading**: The Flask server runs with `threaded=True` to handle concurrent requests.

3. **Request Processing**: Each indication is processed in O(n) where n is the number of path segments.

4. **Logging**: Structured logging with configurable verbosity via `LOG_LEVEL` environment variable.

## Configuration

Environment Variables:

- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR) - default: INFO
- `SERVER_HOST`: Server bind address - default: 0.0.0.0
- `SERVER_PORT`: Server port - default: 5000
- `DEBUG`: Flask debug mode (true/false) - default: false

Kubernetes ConfigMap:

- `uav-policy-config`: Service configuration
- `uav-policy-policies`: Policies and service profiles

## Assumptions and Dependencies

### Algorithm Assumptions

1. **Path Position Accuracy**: Assumes UAV path_position is accurate and normalized to [0.0, 1.0]
2. **RSRP as SINR Proxy**: Uses RSRP measurements as a proxy for SINR calculations
3. **Cell Handover Model**: Assumes immediate handover without connection loss
4. **PRB Model**: Uses Shannon capacity formula with simplified channel model

### External Dependencies

- Flask >= 2.3.0: HTTP framework
- Werkzeug >= 2.3.0: WSGI utilities
- Python >= 3.10: Runtime environment

### RAN Integration

The xApp assumes:
- E2 indications come from an external E2 simulator or RIC
- ResourceDecisions should be sent to RC xApp via separate integration (not included)
- Slice management is handled by network orchestration layer

## Code Structure

```
src/uav_policy/
├── policy_engine.py      # Core policy algorithms
├── server.py             # Flask HTTP server
└── main.py               # Entry point

tests/
├── test_policy_engine.py # Policy algorithm tests
└── test_http_server.py   # API endpoint tests

k8s/
├── deployment.yaml       # Kubernetes deployment
├── service.yaml          # Kubernetes service
├── serviceaccount.yaml   # RBAC configuration
├── configmap.yaml        # Configuration
└── kustomization.yaml    # Kustomize overlay
```

## Design Decisions

1. **Separation of Concerns**: Policy logic in `policy_engine.py` is independent of HTTP transport
2. **Immutable Data Classes**: Uses Python dataclasses for type safety
3. **Decision History**: In-memory history for auditing (bounded by max_history)
4. **Non-root Container**: Runs as unprivileged user for security
5. **Health Checks**: Kubernetes probes for automated restart on failure

## Future Enhancements

- Prometheus metrics export at `/metrics`
- gRPC interface in addition to HTTP
- Integration with Non-RT RIC for policy updates
- Decision caching for repeated indications
- Batch processing for multiple UAVs
- Machine learning model integration for policy optimization

## Related Documentation

- `docs/algorithms.md`: Detailed algorithm design
- `spec/uav_oran_usecase.md`: O-RAN use case specification
- `CLAUDE.md`: Project structure and ground rules
