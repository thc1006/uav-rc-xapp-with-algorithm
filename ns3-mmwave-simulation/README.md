# ns-3 mmWave UAV Simulation

This directory contains ns-3 simulation scenarios for mmWave beam tracking with UAVs.

## Overview

The simulation demonstrates:
- 5G NR mmWave (28 GHz) channel modeling
- Beam management procedures (P1/P2/P3)
- UAV mobility with realistic 3D trajectories
- Integration with O-RAN Near-RT RIC via E2 interface

## Prerequisites

### Required ns-3 Modules

1. **ns-3 core** (version 3.36+)
2. **mmWave module** OR **NR module**
   - mmWave: https://github.com/nyuwireless-unipd/ns3-mmwave
   - NR: https://gitlab.com/cttc-lena/nr

### Installation

```bash
# Clone ns-3 with mmWave module
git clone https://github.com/nyuwireless-unipd/ns3-mmwave.git
cd ns3-mmwave

# Configure and build
./ns3 configure --enable-examples --enable-tests
./ns3 build

# Copy this scenario to scratch directory
cp mmwave_uav_scenario.cc scratch/
```

## Simulation Scenarios

### 1. Basic Beam Tracking (`mmwave_uav_scenario.cc`)

Multiple UAVs flying circular paths around a gNB with beam tracking.

```bash
./ns3 run "scratch/mmwave_uav_scenario --numUavs=3 --simTime=60 --uavSpeed=15"
```

Parameters:
- `--numUavs`: Number of UAVs (default: 3)
- `--simTime`: Simulation duration in seconds (default: 60)
- `--uavSpeed`: UAV speed in m/s (default: 15)
- `--uavHeight`: UAV altitude in meters (default: 50)
- `--enableXapp`: Enable xApp integration (default: false)
- `--xappHost`: xApp host address (default: 127.0.0.1)
- `--xappPort`: xApp port (default: 5001)

### 2. xApp Integration

To enable integration with UAV Beam xApp:

```bash
# Start the xApp first
cd ../xapps/uav-beam
python3 -m uav_beam.main --port 5001

# Then run simulation with xApp enabled
./ns3 run "scratch/mmwave_uav_scenario --enableXapp=true --xappHost=127.0.0.1 --xappPort=5001"
```

## Output Files

- `beam_tracking_results.csv`: Per-measurement beam tracking data
  - Columns: timestamp, ue_id, beam_id, rsrp_dbm, pos_x, pos_y, pos_z, vel_x, vel_y, vel_z

## Channel Model

The simulation uses a simplified mmWave channel model with:
- Free space path loss at 28 GHz
- Atmospheric absorption (~0.1 dB/km)
- Array gain from 8x8 UPA (~18 dB)
- Simplified fading model

For production use, configure ns-3's full mmWave channel model with:
- 3GPP 38.901 statistical channel
- Blockage modeling
- Spatial consistency

## Beam Management

The simulation implements simplified beam management:
- **P1**: Initial beam acquisition via angle-based selection
- **P2**: Beam refinement (simplified to nearest neighbor search)
- **P3**: Continuous tracking via periodic measurements

## Extending the Simulation

### Adding New Mobility Models

```cpp
// Example: Figure-8 pattern
Ptr<WaypointMobilityModel> CreateFigure8Path(
  Vector center,
  double width,
  double height,
  double altitude,
  double speed)
{
  // Implementation
}
```

### Custom Channel Models

```cpp
// Use ns-3 mmWave module's channel
Config::SetDefault("ns3::MmWaveHelper::ChannelModel",
                   StringValue("ns3::ThreeGppChannelModel"));
Config::SetDefault("ns3::ThreeGppChannelModel::Scenario",
                   StringValue("UMa"));  // Urban Macro
```

### E2 Integration

For full E2 integration with ns-O-RAN:

```cpp
#include "ns3/oran-interface.h"

// Setup E2 termination
Ptr<OranInterface> e2Interface = CreateObject<OranInterface>();
e2Interface->SetAttribute("RicAddress", StringValue("10.0.2.1"));
e2Interface->SetAttribute("E2Port", UintegerValue(36421));

// Register E2SM-KPM
e2Interface->RegisterServiceModel(CreateObject<E2SmKpm>());
```

## Performance Metrics

Key metrics for beam tracking evaluation:

1. **Average RSRP**: Signal quality during tracking
2. **Beam switch rate**: Number of beam switches per second
3. **Tracking latency**: Time from position change to beam update
4. **Outage probability**: Time spent below RSRP threshold

## Related Documentation

- [UAV Beam xApp](../xapps/uav-beam/README.md)
- [O-RAN E2 Interface](https://www.o-ran.org/specifications)
- [3GPP TS 38.214](https://www.3gpp.org/specifications): NR physical layer procedures
