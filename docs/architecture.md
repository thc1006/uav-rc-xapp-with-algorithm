# Architecture Overview — UAV RC xApp Policy Project

This project assumes a standard O-RAN split between:

- **Non-RT RIC / rApps**
  - Aggregate UAV flight paths, radio KPIs, and environment info.
  - Train path-aware models and derive `FlightPlanPolicy` objects.
  - Deliver policies to Near-RT RIC via A1 or other northbound channels.

- **Near-RT RIC / xApps**
  - `uav-policy` xApp:
    - Consumes `FlightPlanPolicy` per UAV (if available).
    - Consumes live E2SM_KPM measurements (via existing KPM xApps).
    - Produces `ResourceDecision`s per UAV.
  - RC xApp (not implemented here):
    - Accepts high-level decisions (target cell / slice / PRB hints).
    - Maps them into E2SM_RC / vendor-specific messages toward gNB/DU/RU.

- **RAN nodes (E2 Nodes)**
  - CU-CP / CU-UP / DU / RU with E2 support.
  - Implement beamforming / scheduler / handover logic; RC messages adjust
    configuration parameters, slice weights, PRB quotas, etc.

The **goal** of this repo is to focus on the **policy layer** at Near-RT RIC
(`uav-policy`) while remaining compatible with these architectural roles.
