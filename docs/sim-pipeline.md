# Simulation Pipeline Skeleton (ns-O-RAN style)

This document explains how `nonrt/`, `xapps/`, and `sim/` are meant to
fit together in an ns-O-RAN-like experiment.

## Pipeline overview

1. **Non-RT planning**
   - Input: discretized UAV flight path, predicted (or measured) radio map,
     ServiceProfile.
   - Module: `nonrt/uav-path-planner`.
   - Output: `FlightPlanPolicy` per UAV, typically written as JSON under
     `sim/artifacts/policies/`.

2. **Near-RT policy**
   - Input: `FlightPlanPolicy` + per-UAV `UavState` + `RadioSnapshot`
     (derived from E2SM_KPM reports).
   - Module: `xapps/uav-policy`.
   - Output: `ResourceDecision` objects per UAV at each decision time.

3. **RC control (not fully implemented here)**
   - Input: `ResourceDecision` objects.
   - Module: `xapps/rc-grpc-client` or external RC client.
   - Output: E2SM_RC (or vendor-specific) control messages to gNB/DU/RU.

4. **ns-O-RAN / RAN simulator**
   - Provides the RAN topology, UE configuration, and E2 interface.
   - Sends KPM reports and accepts RC actions.

The `sim/` directory contains YAML files and Python scripts that show how
these pieces could be wired together in an ns-O-RAN-style environment.
They are intentionally simple and meant to be adapted to a specific
testbed (ns-O-RAN, srsRAN+RIC, etc.).
