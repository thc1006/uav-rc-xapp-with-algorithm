# UAV Policy Algorithms for O-RAN RC Control

This document refines the textual O-RAN UAV use cases (Flight Path Based
Dynamic UAV Radio Resource Allocation and Radio Resource Allocation for UAV
Application Scenario) into concrete algorithms that can be implemented in the
`uav-policy` xApp and exercised via the RC xApp.

The guiding idea is to split the logic into:

- **Offline planning (Non-RT RIC / rApp)** – given a predicted flight path and
  a radio map, derive a per-path-segment plan: serving cell, slice, and base
  PRB quota.
- **Online control (Near-RT RIC / xApps)** – at runtime, track UAVs along their
  flight paths, adapt the plan using fresh KPM measurements, and send
  `ResourceDecision`s to the RC xApp.

## 1. Data model

We model the UAV and radio environment using the following structures (also
reflected in `uav_policy/policy_engine.py`):

- `UavState`:
  - `uav_id`: UAV identifier
  - `x, y, z`: current 3D position in some local coordinate system
  - `slice_id`: (optional) business slice for this UAV
  - `path_position`: (optional) scalar in `[0.0, 1.0]` representing the UAV’s
    progress along its planned path (0 = start, 1 = end).

- `RadioSnapshot`:
  - `serving_cell_id`
  - `neighbor_cell_ids` – ordered list by descending measured RSRP
  - `rsrp_serving`
  - `rsrp_best_neighbor`
  - `prb_utilization_serving` – fraction in `[0,1]`
  - `prb_utilization_slice` – (optional) slice-level utilization in `[0,1]`

- `PathSegmentPlan`:
  - `start_pos`, `end_pos` – path-position interval `[0,1]`
  - `planned_cell_id` – desired serving cell for this segment
  - `slice_id` – desired slice
  - `base_prb_quota` – nominal PRB quota (integer, e.g. out of 100)

- `FlightPlanPolicy`:
  - `uav_id`
  - `segments: List[PathSegmentPlan]`

- `ServiceProfile`:
  - `name` – label (e.g. `"uav-hd-video"`)
  - `target_bitrate_mbps` – desired end-to-end bitrate
  - `min_sinr_db` – minimum acceptable SINR (approximate threshold)

- `ResourceDecision`:
  - `uav_id`
  - `target_cell_id`
  - `slice_id`
  - `prb_quota`
  - `reason` – human-readable explanation for logging/debugging.

## 2. Offline planning algorithm (Non-RT RIC)

See inline comments in this file for the full description. In short, the
algorithm:

1. Enumerates candidate cells per waypoint based on SINR / RSRP thresholds.
2. Scores candidates with a utility function combining quality and load.
3. Uses dynamic programming across waypoints to find a cell sequence that
   maximizes total utility minus a handover penalty.
4. Compresses the resulting sequence into `PathSegmentPlan`s.
5. Packages them into a `FlightPlanPolicy` per UAV.

## 3. Online control algorithm (Near-RT RIC / `uav-policy` xApp)

The `path_aware_rc_policy` function in `policy_engine.py` implements the
online algorithm:

1. Determine the active path segment from the UAV's `path_position` and the
   `FlightPlanPolicy`.
2. Decide whether to stay on the serving cell or switch to the planned cell,
   based on PRB utilization and RSRP hysteresis.
3. Choose a slice (from UAV state or segment definition).
4. Estimate a required PRB quota using a Shannon-like formula and clamp it to
   reasonable bounds.
5. Return a `ResourceDecision` combining these choices and a textual reason.

A non-path-aware baseline policy (`simple_path_aware_policy`) is also kept for
comparison.
