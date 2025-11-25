# sim/ — ns-O-RAN-style UAV Scenario Skeleton

This directory contains high-level configuration and helper scripts that
show how to connect:

- `nonrt/uav-path-planner` (flight path -> FlightPlanPolicy),
- `xapps/uav-policy` (FlightPlanPolicy + KPM -> ResourceDecision),
- an ns-O-RAN-style topology (RAN + Near-RT RIC).

Nothing here talks directly to ns-3 or a real RIC; the files are meant
as *skeletons* that you can adapt to your specific environment.
