# UAV RC xApp Policy + Non-RT Planner + ns-O-RAN Sim Skeleton

This repository is scaffold for experimenting with
O-RAN WG1 UAV use cases (Flight Path Based Dynamic UAV Radio Resource Allocation /
Radio Resource Allocation for UAV Application Scenario) using:

- A **Non-RT path planner** (`nonrt/uav-path-planner`) that turns flight
  paths + radio maps into `FlightPlanPolicy` objects.
- A **Near-RT UAV policy xApp** (`xapps/uav-policy`) that uses those
  `FlightPlanPolicy` objects + KPM to make `ResourceDecision`s.
- A **stub RC gRPC client** (`xapps/rc-grpc-client`) to forward decisions
  towards an RC xApp.
- An **ns-O-RAN simulation skeleton** (`sim/`) that shows how the pieces
  fit into a simulated RAN topology.
- **Claude Code configuration** (CLAUDE.md, skills, subagents, commands)
  to develop and iterate on the logic with Anthropic tooling.

All code is deliberately lightweight and meant as a starting point for
experiments with ns-O-RAN, srsRAN + RIC, or other O-RAN testbeds.

See `docs/architecture.md` for an overview.
