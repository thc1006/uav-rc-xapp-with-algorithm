# uav-path-planner (Non-RT)

This package contains a **Non-RT path planner** for UAVs:

- Input: discretized flight path (`Waypoint` list), `RadioMap`, `ServiceProfile`.
- Output: `FlightPlanPolicy` (sequence of `PathSegmentPlan`s).

A simple dynamic programming (DP) algorithm chooses a serving cell per
waypoint to balance:
- Link quality (SINR),
- Cell load,
- Number of handovers (with an explicit penalty).

The resulting sequence is compressed into segments and can be consumed
by the Near-RT `uav-policy` xApp.
