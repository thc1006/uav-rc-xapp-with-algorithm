# UAV Beam Tracking xApp

A high-performance mmWave beam management xApp for O-RAN Near-RT RIC, designed specifically for UAV (Unmanned Aerial Vehicle) communication scenarios.

## Overview

The UAV Beam Tracking xApp provides intelligent beam management for 5G NR FR2 (mmWave) systems serving UAV users. It implements predictive beam tracking algorithms that leverage UAV trajectory information to maintain reliable high-throughput mmWave links despite the challenging mobility patterns of aerial vehicles.

### Key Features

- **Predictive Beam Tracking**: Proactive beam switching based on UAV trajectory prediction
- **3GPP Beam Procedures**: Full P1/P2/P3 beam management implementation
- **Multi-Algorithm AoA/AoD Estimation**: MUSIC, ESPRIT, and Beamspace methods
- **Kalman Filter Integration**: Accurate UAV state estimation and prediction
- **REST API Interface**: E2-compatible indication and control interface
- **Real-time Statistics**: Comprehensive monitoring and performance metrics

## System Architecture

```
+------------------------------------------------------------------+
|                        O-RAN Near-RT RIC                          |
+------------------------------------------------------------------+
|                                                                    |
|  +------------------------------------------------------------+  |
|  |                    UAV Beam Tracking xApp                   |  |
|  +------------------------------------------------------------+  |
|  |                                                              |  |
|  |  +------------------+  +------------------+  +------------+  |  |
|  |  |   BeamTracker    |  |   Trajectory     |  |   Angle    |  |  |
|  |  |                  |  |   Predictor      |  |  Estimator |  |  |
|  |  |  - P1/P2/P3      |  |                  |  |            |  |  |
|  |  |  - Beam Decision |  |  - Kalman Filter |  |  - MUSIC   |  |  |
|  |  |  - Codebook Mgmt |  |  - Polynomial    |  |  - ESPRIT  |  |  |
|  |  |  - Statistics    |  |  - Hybrid        |  |  - Beam-   |  |  |
|  |  +--------+---------+  +--------+---------+  |    space   |  |  |
|  |           |                     |            +------+-----+  |  |
|  |           +----------+----------+-------------------+        |  |
|  |                      |                                       |  |
|  |           +----------v----------+                            |  |
|  |           |    REST API Server  |                            |  |
|  |           |    (Flask-based)    |                            |  |
|  |           +----------+----------+                            |  |
|  +----------------------|--------------------------------------|--+
|                         |                                      |
+-----------------------+-|--------------------------------------+-+
                        | |
         +--------------+ +---------------+
         |                                |
+--------v--------+            +----------v---------+
|   E2 Interface  |            |    Management      |
| (Indications/   |            |    Interface       |
|  Control)       |            |    (Config/Stats)  |
+-----------------+            +--------------------+
         |
+--------v------------------------------------------+
|              E2 Nodes (gNB-DU/CU)                 |
|                                                   |
|  +---------------+  +---------------+             |
|  |   Cell 1      |  |   Cell 2      |   ...       |
|  | 8x8 UPA Array | | 8x8 UPA Array |             |
|  +---------------+  +---------------+             |
+---------------------------------------------------+
         |
+--------v------------------------------------------+
|                   UAV Fleet                       |
|  [UAV-001]    [UAV-002]    [UAV-003]    ...      |
+---------------------------------------------------+
```

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

```bash
# Clone or navigate to the xApp directory
cd xapps/uav-beam

# Install the package
pip install -e .

# For development dependencies
pip install -e ".[dev]"

# For ML-based features (optional)
pip install -e ".[ml]"
```

### Running the xApp

```bash
# Start with default configuration
uav-beam-xapp

# Custom host and port
uav-beam-xapp --host 0.0.0.0 --port 5001

# Custom beam configuration
uav-beam-xapp --num-beams-h 32 --num-beams-v 16

# Debug mode
uav-beam-xapp --debug
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=uav_beam --cov-report=html
```

## Configuration Options

### Beam Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_antenna_elements_h` | 8 | Horizontal antenna elements |
| `num_antenna_elements_v` | 8 | Vertical antenna elements |
| `num_beams_h` | 16 | Horizontal beam directions |
| `num_beams_v` | 8 | Vertical beam directions |
| `ssb_periodicity_ms` | 20.0 | SSB burst periodicity |
| `csi_rs_periodicity_ms` | 10.0 | CSI-RS periodicity |
| `beam_failure_threshold_db` | -10.0 | L1-RSRP failure threshold |
| `beam_recovery_timer_ms` | 50.0 | Recovery timer |
| `tracking_update_interval_ms` | 5.0 | Tracking update rate |
| `prediction_horizon_ms` | 20.0 | Prediction lookahead |

### Trajectory Predictor Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `process_noise_position` | 0.1 | Position process noise (m) |
| `process_noise_velocity` | 1.0 | Velocity process noise (m/s) |
| `measurement_noise_position` | 0.5 | GPS measurement noise (m) |
| `max_prediction_horizon_ms` | 500.0 | Maximum prediction time |
| `history_window_size` | 50 | States to keep in history |
| `max_acceleration` | 5.0 | Maximum UAV acceleration (m/s^2) |
| `max_velocity` | 30.0 | Maximum UAV velocity (m/s) |

### Angle Estimator Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_elements_h` | 8 | Horizontal antenna elements |
| `num_elements_v` | 8 | Vertical antenna elements |
| `element_spacing` | 0.5 | Spacing in wavelengths |
| `num_snapshots` | 64 | Signal snapshots |
| `subspace_dimension` | 4 | Signal subspace dimension |
| `angular_resolution` | 0.01 | Resolution in radians |

## API Overview

### Health Check
```
GET /health
```
Returns xApp health status and version.

### E2 Indication
```
POST /e2/indication
```
Process beam measurements and return beam decisions.

### Angle Estimation
```
POST /angle/estimate
```
Estimate AoA/AoD from received signal samples.

### Statistics
```
GET /statistics
```
Get comprehensive xApp statistics.

### UE State
```
GET /ue/<ue_id>
```
Get state information for a specific UE.

### Configuration
```
GET /config
PUT /config
```
Get or update xApp configuration.

### Reset
```
POST /reset
```
Reset xApp state.

For complete API documentation, see [docs/API.md](docs/API.md).

## Module Overview

| Module | Description |
|--------|-------------|
| `beam_tracker.py` | Core beam management with P1/P2/P3 procedures |
| `trajectory_predictor.py` | Kalman filter based UAV trajectory prediction |
| `angle_estimator.py` | AoA/AoD estimation (MUSIC, ESPRIT, Beamspace) |
| `server.py` | REST API server and xApp integration |
| `main.py` | Application entry point |

## Documentation

- [API Reference](docs/API.md) - Complete REST API documentation
- [Architecture Guide](docs/ARCHITECTURE.md) - System design and integration
- [Algorithms](docs/ALGORITHMS.md) - Beam tracking and estimation algorithms
- [Deployment Guide](docs/DEPLOYMENT.md) - Docker and Kubernetes deployment

## Dependencies

### Required
- Flask >= 2.0.0
- NumPy >= 1.20.0
- SciPy >= 1.7.0
- Requests >= 2.25.0

### Development
- pytest >= 6.0.0
- pytest-cov >= 2.0.0
- black >= 21.0
- flake8 >= 3.9.0

### Optional (ML Features)
- PyTorch >= 1.9.0
- scikit-learn >= 0.24.0

## References

- 3GPP TS 38.214: Physical layer procedures for data
- 3GPP TS 38.321: MAC protocol specification
- O-RAN.WG3.E2AP: E2 Application Protocol
- O-RAN.WG3.E2SM: E2 Service Models

## License

Apache License 2.0
