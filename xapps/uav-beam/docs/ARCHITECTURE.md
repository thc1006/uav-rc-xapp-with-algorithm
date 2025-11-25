# UAV Beam Tracking xApp - Architecture Guide

This document describes the system architecture, module relationships, and integration patterns for the UAV Beam Tracking xApp.

## Table of Contents

1. [System Overview](#system-overview)
2. [Module Architecture](#module-architecture)
3. [Data Flow](#data-flow)
4. [O-RAN Integration](#o-ran-integration)
5. [Extension Guide](#extension-guide)

---

## System Overview

### High-Level Architecture

```
+===========================================================================+
||                          O-RAN Near-RT RIC Platform                     ||
+===========================================================================+
|                                                                           |
|  +---------------------------------------------------------------------+  |
|  |                      UAV Beam Tracking xApp                         |  |
|  +---------------------------------------------------------------------+  |
|  |                                                                     |  |
|  |   +-------------------+       +-------------------+                 |  |
|  |   |                   |       |                   |                 |  |
|  |   |   BeamTracker     |<----->|   UAVBeamXApp     |                 |  |
|  |   |                   |       |   (Coordinator)   |                 |  |
|  |   +-------------------+       +--------+----------+                 |  |
|  |           ^                            |                            |  |
|  |           |                            v                            |  |
|  |   +-------+-------+           +-------------------+                 |  |
|  |   |               |           |                   |                 |  |
|  |   | Trajectory    |<--------->|   Flask REST      |                 |  |
|  |   | Predictor     |           |   API Server      |                 |  |
|  |   |               |           |                   |                 |  |
|  |   +-------+-------+           +--------+----------+                 |  |
|  |           ^                            |                            |  |
|  |           |                            |                            |  |
|  |   +-------+-------+                    |                            |  |
|  |   |               |                    |                            |  |
|  |   | Angle         |                    |                            |  |
|  |   | Estimator     |                    |                            |  |
|  |   |               |                    |                            |  |
|  |   +---------------+                    |                            |  |
|  |                                        |                            |  |
|  +----------------------------------------|----------------------------+  |
|                                           |                               |
+-------------------------------------------|-------------------------------+
                                            |
              +-----------------------------+-----------------------------+
              |                             |                             |
              v                             v                             v
    +-----------------+         +-----------------+         +-----------------+
    |  E2 Interface   |         |  A1 Interface   |         |  O1 Interface   |
    |  (Indications/  |         |  (Policies)     |         |  (Management)   |
    |   Controls)     |         |                 |         |                 |
    +-----------------+         +-----------------+         +-----------------+
              |
              v
    +=========================================================+
    |                    E2 Nodes (gNB)                        |
    |                                                          |
    |  +------------------+  +------------------+              |
    |  |     gNB-CU       |  |     gNB-DU       |              |
    |  |  (Control Plane) |  |  (Data Plane)    |              |
    |  +------------------+  +--------+---------+              |
    |                                 |                        |
    |                       +---------+---------+              |
    |                       |   mmWave Radio    |              |
    |                       |   8x8 UPA Array   |              |
    |                       +-------------------+              |
    +=========================================================+
              |
              | (Air Interface - NR FR2 mmWave)
              v
    +=========================================================+
    |                      UAV Fleet                           |
    |                                                          |
    |    [UAV-001]      [UAV-002]      [UAV-003]              |
    |    Position:      Position:      Position:               |
    |    (x,y,z)        (x,y,z)        (x,y,z)                |
    |    Velocity:      Velocity:      Velocity:               |
    |    (vx,vy,vz)     (vx,vy,vz)     (vx,vy,vz)             |
    +=========================================================+
```

### Component Responsibilities

| Component | Responsibility |
|-----------|----------------|
| **UAVBeamXApp** | Central coordinator, integrates all modules |
| **BeamTracker** | Beam management decisions (P1/P2/P3) |
| **TrajectoryPredictor** | UAV state estimation and prediction |
| **AngleEstimator** | AoA/AoD calculation from signals |
| **Flask Server** | REST API and external interface |

---

## Module Architecture

### BeamTracker Module

```
+---------------------------------------------------------------------+
|                          BeamTracker                                 |
+---------------------------------------------------------------------+
|                                                                     |
|  Configuration (BeamConfig)                                         |
|  +---------------------------------------------------------------+  |
|  | - num_antenna_elements_h/v: Antenna array dimensions          |  |
|  | - num_beams_h/v: Codebook size                                |  |
|  | - beam_failure_threshold_db: L1-RSRP threshold                |  |
|  | - prediction_horizon_ms: Lookahead time                       |  |
|  +---------------------------------------------------------------+  |
|                                                                     |
|  State Management                                                   |
|  +---------------------------------------------------------------+  |
|  | ue_states: Dict[str, UEState]                                 |  |
|  |   - state: BeamState (IDLE/ACQUIRING/REFINING/TRACKING/etc)   |  |
|  |   - last_beam: Current serving beam                           |  |
|  |   - failure_count: Consecutive failures                       |  |
|  |                                                               |  |
|  | measurement_history: Dict[str, List[BeamMeasurement]]         |  |
|  |   - Per-UE measurement history (last 100)                     |  |
|  +---------------------------------------------------------------+  |
|                                                                     |
|  Core Functions                                                     |
|  +---------------------------------------------------------------+  |
|  | process_measurement(measurement) -> BeamDecision              |  |
|  |   1. Check for beam failure                                   |  |
|  |   2. Predict optimal beam                                     |  |
|  |   3. Decide action (maintain/switch/recover)                  |  |
|  |                                                               |  |
|  | _predict_optimal_beam(measurement) -> (beam_id, confidence)   |  |
|  |   - If position available: trajectory-based prediction        |  |
|  |   - Otherwise: RSRP trend analysis                            |  |
|  |                                                               |  |
|  | _handle_beam_failure(measurement) -> BeamDecision             |  |
|  |   - Find best alternative beam                                |  |
|  |   - Trigger recovery procedure                                |  |
|  +---------------------------------------------------------------+  |
|                                                                     |
|  Beam Codebook                                                      |
|  +---------------------------------------------------------------+  |
|  | codebook: np.ndarray (num_beams, num_elements)                |  |
|  |   - DFT-based steering vectors                                |  |
|  |   - Normalized for unit power                                 |  |
|  +---------------------------------------------------------------+  |
+---------------------------------------------------------------------+
```

### TrajectoryPredictor Module

```
+---------------------------------------------------------------------+
|                       TrajectoryPredictor                            |
+---------------------------------------------------------------------+
|                                                                     |
|  Per-UAV State                                                      |
|  +---------------------------------------------------------------+  |
|  | filters: Dict[str, KalmanFilter3D]                            |  |
|  |   - One Kalman filter per tracked UAV                         |  |
|  |                                                               |  |
|  | history: Dict[str, deque]                                     |  |
|  |   - Position history for polynomial fitting                   |  |
|  |                                                               |  |
|  | waypoints: Dict[str, List[np.ndarray]]                        |  |
|  |   - Mission waypoints (if available)                          |  |
|  +---------------------------------------------------------------+  |
|                                                                     |
|  KalmanFilter3D                                                     |
|  +---------------------------------------------------------------+  |
|  | State Vector: [x, y, z, vx, vy, vz]                           |  |
|  |                                                               |  |
|  | Matrices:                                                     |  |
|  |   - F: State transition (constant velocity)                   |  |
|  |   - Q: Process noise                                          |  |
|  |   - H: Measurement (observe position)                         |  |
|  |   - R: Measurement noise                                      |  |
|  |   - P: State covariance                                       |  |
|  |                                                               |  |
|  | Methods:                                                      |  |
|  |   - predict(dt_ms): Propagate state forward                   |  |
|  |   - update(measurement): Incorporate new observation          |  |
|  |   - predict_future(horizon_ms): Predict without update        |  |
|  +---------------------------------------------------------------+  |
|                                                                     |
|  Prediction Methods                                                 |
|  +---------------------------------------------------------------+  |
|  | 1. Kalman (constant velocity model)                           |  |
|  |    - Fast, smooth predictions                                 |  |
|  |    - Best for short horizons                                  |  |
|  |                                                               |  |
|  | 2. Polynomial (quadratic fit)                                 |  |
|  |    - Captures acceleration/maneuvers                          |  |
|  |    - Better for longer horizons                               |  |
|  |                                                               |  |
|  | 3. Hybrid (weighted combination)                              |  |
|  |    - Adapts weight based on horizon                           |  |
|  |    - Best overall accuracy                                    |  |
|  +---------------------------------------------------------------+  |
+---------------------------------------------------------------------+
```

### AngleEstimator Module

```
+---------------------------------------------------------------------+
|                         AngleEstimator                               |
+---------------------------------------------------------------------+
|                                                                     |
|  Pre-computed Data                                                  |
|  +---------------------------------------------------------------+  |
|  | array_response: np.ndarray (num_az, num_el, num_elements)     |  |
|  |   - Steering vectors for all search angles                    |  |
|  |                                                               |  |
|  | azimuth_grid: np.ndarray                                      |  |
|  |   - Search angles for azimuth                                 |  |
|  |                                                               |  |
|  | elevation_grid: np.ndarray                                    |  |
|  |   - Search angles for elevation                               |  |
|  +---------------------------------------------------------------+  |
|                                                                     |
|  Estimation Methods                                                 |
|  +---------------------------------------------------------------+  |
|  | MUSIC (Multiple Signal Classification)                        |  |
|  |   1. Compute covariance matrix R = XX^H / N                   |  |
|  |   2. Eigendecomposition: R = U * Lambda * U^H                 |  |
|  |   3. Separate signal/noise subspaces                          |  |
|  |   4. Compute pseudo-spectrum: P(theta) = 1/|a^H * En|^2       |  |
|  |   5. Find spectrum peaks                                      |  |
|  |   Pros: High resolution, accurate                             |  |
|  |   Cons: Computationally expensive                             |  |
|  +---------------------------------------------------------------+  |
|  | ESPRIT                                                        |  |
|  |   1. Compute signal subspace Es                               |  |
|  |   2. Use shift-invariance: J1*Es and J2*Es                    |  |
|  |   3. Solve for rotation matrix Phi                            |  |
|  |   4. Extract angles from Phi eigenvalues                      |  |
|  |   Pros: Closed-form, no spectrum search                       |  |
|  |   Cons: Requires uniform array, 1D only                       |  |
|  +---------------------------------------------------------------+  |
|  | Beamspace                                                     |  |
|  |   1. Apply DFT beamforming codebook                           |  |
|  |   2. Find beam with maximum power                             |  |
|  |   3. Map beam index to angle                                  |  |
|  |   Pros: Very fast, simple                                     |  |
|  |   Cons: Resolution limited by codebook                        |  |
|  +---------------------------------------------------------------+  |
+---------------------------------------------------------------------+
```

---

## Data Flow

### E2 Indication Processing Flow

```
+-------------+     +----------------+     +------------------+
|   gNB-DU    | --> | E2 Interface   | --> | POST /e2/indication |
| (Beam Meas) |     | (RMR/HTTP)     |     | (REST API)       |
+-------------+     +----------------+     +--------+---------+
                                                    |
                                                    v
                                          +------------------+
                                          | UAVBeamXApp      |
                                          | .process_e2_     |
                                          |  indication()    |
                                          +--------+---------+
                                                   |
                    +------------------------------+------------------------------+
                    |                              |                              |
                    v                              v                              v
          +------------------+          +------------------+          +------------------+
          | TrajectoryPred.  |          | BeamTracker      |          | Statistics       |
          | .update()        |          | .process_        |          | Update           |
          |                  |          |  measurement()   |          |                  |
          +--------+---------+          +--------+---------+          +------------------+
                   |                             |
                   v                             |
          +------------------+                   |
          | TrajectoryPred.  |                   |
          | .predict()       |                   |
          | (hybrid method)  |                   |
          +--------+---------+                   |
                   |                             |
                   +----------+------------------+
                              |
                              v
                    +------------------+
                    | BeamDecision     |
                    | - action         |
                    | - target_beam    |
                    | - confidence     |
                    +--------+---------+
                             |
                             v
                    +------------------+
                    | JSON Response    |
                    | to E2 Node       |
                    +------------------+
```

### Beam Decision Logic

```
                    +---------------------------+
                    |   New Beam Measurement    |
                    +-------------+-------------+
                                  |
                                  v
                    +---------------------------+
                    | RSRP < failure_threshold? |
                    +-------------+-------------+
                         |              |
                        YES            NO
                         |              |
                         v              v
            +----------------+  +------------------+
            | BEAM FAILURE   |  | Position/Velocity|
            | RECOVERY       |  | Available?       |
            +-------+--------+  +--------+---------+
                    |                |         |
                    |               YES       NO
                    |                |         |
                    |                v         v
                    |    +---------------+  +---------------+
                    |    | Trajectory-   |  | RSRP Trend    |
                    |    | Based         |  | Analysis      |
                    |    | Prediction    |  |               |
                    |    +-------+-------+  +-------+-------+
                    |            |                  |
                    |            +--------+---------+
                    |                     |
                    |                     v
                    |         +---------------------+
                    |         | Predicted Beam !=   |
                    |         | Current Beam AND    |
                    |         | Confidence > 0.7?   |
                    |         +----------+----------+
                    |              |          |
                    |             YES        NO
                    |              |          |
                    v              v          v
            +------------+  +------------+  +------------+
            | Action:    |  | Action:    |  | Action:    |
            | RECOVER    |  | SWITCH     |  | MAINTAIN   |
            +------------+  +------------+  +------------+
```

### State Diagram

```
                              +-------+
                              | IDLE  |
                              +---+---+
                                  |
                         First measurement
                                  |
                                  v
                           +------------+
                           | ACQUIRING  |
               +---------->|   (P1)     |
               |           +-----+------+
               |                 |
               |          Beam acquired
               |                 |
               |                 v
               |           +------------+
               |           | REFINING   |
               |    +----->|   (P2)     |
               |    |      +-----+------+
               |    |            |
               |   Poor      Refinement
               |   link        complete
               |    |            |
               |    |            v
               |    |      +------------+
               |    +------+ TRACKING   |<----+
               |           |   (P3)     |     |
               |           +-----+------+     |
               |                 |            |
               |           Beam failure       |
               |                 |        Recovery
               |                 v         success
               |           +------------+     |
               +---------->| RECOVERY   +-----+
                           |   (P3)     |
                           +-----+------+
                                 |
                          Recovery failed
                                 |
                                 v
                           +------------+
                           |  FAILED    |
                           +------------+
```

---

## O-RAN Integration

### E2 Interface Architecture

```
+===========================================================================+
|                           O-RAN Near-RT RIC                               |
+===========================================================================+
|                                                                           |
|   +-------------------+     +-------------------+     +----------------+   |
|   |   Subscription    |     |    Indication     |     |    Control     |   |
|   |    Manager        |<--->|    Handler        |<--->|    Handler     |   |
|   +--------+----------+     +--------+----------+     +-------+--------+   |
|            |                         |                        |            |
|            +-----------+-------------+------------------------+            |
|                        |                                                   |
|                        v                                                   |
|              +-------------------+                                         |
|              | E2 Termination    |                                         |
|              | (SCTP/E2AP)       |                                         |
|              +--------+----------+                                         |
|                       |                                                    |
+-----------------------|----------------------------------------------------+
                        |
         E2AP (ASN.1 encoded over SCTP)
                        |
+-----------------------|----------------------------------------------------+
|                       v                                                    |
|              +-------------------+                                         |
|              | E2 Agent          |                                         |
|              | (in gNB)          |                                         |
|              +--------+----------+                                         |
|                       |                                                    |
|   +-------------------+-------------------+                                |
|   |                                       |                                |
|   v                                       v                                |
|  +-------------------+         +-------------------+                       |
|  | E2SM-KPM          |         | E2SM-RC           |                       |
|  | (Key Performance  |         | (RAN Control)     |                       |
|  |  Monitoring)      |         |                   |                       |
|  +-------------------+         +-------------------+                       |
|                                                                            |
|                          gNB-DU / gNB-CU                                   |
+============================================================================+
```

### E2SM Service Model Mapping

#### E2SM-KPM (Monitoring)

```
E2SM-KPM Indication Message:
+------------------------------------------+
| Header                                    |
|   - Indication Type: REPORT              |
|   - Timestamp                             |
+------------------------------------------+
| Indication Message                        |
|   +--------------------------------------+
|   | Measurement Data                     |
|   |   - UE ID: "uav-001"                |
|   |   - Cell ID: "cell-1"               |
|   |   - Beam Index: 42                   |
|   |   - L1-RSRP: -85 dBm                |
|   |   - Neighbor Beams: [41, 43, 44]     |
|   +--------------------------------------+
+------------------------------------------+

Maps to xApp:
{
  "ue_id": "uav-001",
  "serving_cell_id": "cell-1",
  "serving_beam_id": 42,
  "beam_rsrp_dbm": -85.0,
  "neighbor_beams": {...}
}
```

#### E2SM-RC (Control)

```
E2SM-RC Control Message:
+------------------------------------------+
| Header                                    |
|   - Control Style: Beam Management       |
|   - RIC Control Action ID                 |
+------------------------------------------+
| Control Message                           |
|   +--------------------------------------+
|   | Beam Handover Command                |
|   |   - UE ID: "uav-001"                |
|   |   - Target Beam Index: 43            |
|   |   - Handover Type: PROACTIVE         |
|   +--------------------------------------+
+------------------------------------------+

Generated from xApp decision:
{
  "action": "switch",
  "target_beam_id": 43,
  "confidence": 0.85
}
```

### A1 Policy Interface

```
A1 Policy (from Non-RT RIC):
+------------------------------------------+
| Policy Type: Beam Management             |
| Policy ID: beam-policy-001               |
+------------------------------------------+
| Policy Content:                          |
|   - Max beam switch rate: 10/sec         |
|   - Min confidence threshold: 0.7        |
|   - Priority UEs: ["uav-001", "uav-002"] |
|   - Beam failure action: AGGRESSIVE      |
+------------------------------------------+

xApp uses policy to adjust:
- BeamConfig.prediction_horizon_ms
- BeamConfig.beam_failure_threshold_db
- Switch decision confidence threshold
```

---

## Extension Guide

### Adding New Prediction Methods

1. **Create method in TrajectoryPredictor**:

```python
# In trajectory_predictor.py

def _predict_waypoint_aware(self, uav_id: str, horizon_ms: float) -> UAVState:
    """Prediction using mission waypoint information"""
    waypoints = self.waypoints.get(uav_id, [])

    if not waypoints:
        return self._predict_kalman(uav_id, horizon_ms)

    # Find next waypoint
    current_pos, _ = self.filters[uav_id].get_state()
    next_wp = self._find_next_waypoint(current_pos, waypoints)

    # Predict path toward waypoint
    # ... implementation ...

    return UAVState(...)
```

2. **Register in predict() method**:

```python
def predict(self, uav_id: str, horizon_ms: float, method: str = "kalman"):
    # ... existing code ...
    elif method == "waypoint":
        return self._predict_waypoint_aware(uav_id, horizon_ms)
```

### Adding New Angle Estimation Algorithms

1. **Add method enum**:

```python
class EstimationMethod(Enum):
    MUSIC = "music"
    ESPRIT = "esprit"
    BEAMSPACE = "beamspace"
    ML = "ml"
    COMPRESSIVE = "compressive"  # New
```

2. **Implement estimation**:

```python
def _estimate_compressive(
    self,
    received_signal: np.ndarray,
    timestamp_ms: float
) -> AngleEstimate:
    """Compressive sensing based angle estimation"""
    # Use sparse recovery (e.g., OMP, LASSO)
    # ... implementation ...

    return AngleEstimate(
        timestamp_ms=timestamp_ms,
        azimuth_rad=azimuth_est,
        elevation_rad=elevation_est,
        confidence=confidence,
        method="Compressive"
    )
```

### Adding ML-Based Beam Prediction

1. **Create ML predictor class**:

```python
# In ml_predictor.py (new file)

import torch
import torch.nn as nn

class BeamPredictionNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_beams):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_beams)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out[:, -1, :])

class MLBeamPredictor:
    def __init__(self, model_path: str):
        self.model = BeamPredictionNet(...)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def predict(self, history: List[BeamMeasurement]) -> Tuple[int, float]:
        # Convert history to tensor
        features = self._extract_features(history)

        with torch.no_grad():
            probs = torch.softmax(self.model(features), dim=-1)

        best_beam = probs.argmax().item()
        confidence = probs.max().item()

        return best_beam, confidence
```

2. **Integrate with BeamTracker**:

```python
class BeamTracker:
    def __init__(self, config=None, ml_model_path=None):
        # ... existing init ...

        if ml_model_path:
            self.ml_predictor = MLBeamPredictor(ml_model_path)
        else:
            self.ml_predictor = None

    def _predict_optimal_beam(self, measurement):
        if self.ml_predictor and len(history) >= 10:
            return self.ml_predictor.predict(history)

        # Fall back to existing methods
        # ...
```

### Adding New API Endpoints

1. **Define endpoint in server.py**:

```python
@app.route('/beam/codebook', methods=['GET'])
def get_codebook():
    """Get beam codebook information"""
    xapp_instance = get_xapp()
    codebook = xapp_instance.beam_tracker.codebook

    return jsonify({
        "num_beams": xapp_instance.beam_tracker.config.total_beams,
        "num_elements": xapp_instance.beam_tracker.config.total_antenna_elements,
        "codebook_shape": list(codebook.shape),
        # Note: Full codebook may be too large to return
    })

@app.route('/prediction/<uav_id>', methods=['GET'])
def get_prediction(uav_id: str):
    """Get trajectory prediction for UAV"""
    horizon_ms = request.args.get('horizon_ms', 100.0, type=float)
    method = request.args.get('method', 'hybrid')

    xapp_instance = get_xapp()
    prediction = xapp_instance.trajectory_predictor.predict(
        uav_id, horizon_ms, method
    )

    if prediction is None:
        return jsonify({"error": f"UAV {uav_id} not tracked"}), 404

    return jsonify({
        "uav_id": uav_id,
        "horizon_ms": horizon_ms,
        "method": method,
        "predicted_position": prediction.position.tolist(),
        "predicted_velocity": prediction.velocity.tolist(),
        "confidence": xapp_instance.trajectory_predictor.get_prediction_confidence(
            uav_id, horizon_ms
        )
    })
```

### Integrating with SDL (Shared Data Layer)

```python
# Replace internal dictionaries with SDL calls

from ricsdl.syncstorage import SyncStorage

class UAVBeamXAppWithSDL(UAVBeamXApp):
    def __init__(self, ...):
        super().__init__(...)
        self.sdl = SyncStorage()
        self.namespace = "uav-beam-xapp"

    def _save_ue_state(self, ue_id: str, state: dict):
        key = f"ue:{ue_id}:state"
        self.sdl.set(self.namespace, {key: json.dumps(state)})

    def _load_ue_state(self, ue_id: str) -> Optional[dict]:
        key = f"ue:{ue_id}:state"
        result = self.sdl.get(self.namespace, key)
        if result:
            return json.loads(result[key])
        return None

    def _save_decision(self, ue_id: str, decision: BeamDecision):
        key = f"ue:{ue_id}:decisions"
        # Use SDL list operations for history
        self.sdl.add_member(self.namespace, key, json.dumps(asdict(decision)))
```
