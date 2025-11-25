"""Placeholder script for ns-O-RAN integration.

This script is intentionally minimal. It reads the scenario description
and prints a summary to stdout. In a real setup, you would:

- Launch or configure an ns-O-RAN experiment,
- Point the RIC / xApps to the correct policy/decision files,
- Orchestrate KPM subscriptions and RC control loops.
"""

import sys
from pathlib import Path

import yaml  # type: ignore

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    scenario_path = REPO_ROOT / "sim" / "nsoran" / "scenario.yaml"
    data = yaml.safe_load(scenario_path.read_text(encoding="utf-8"))
    scenario_id = data.get("scenario_id")
    description = data.get("description")
    print(f"[sim-glue] Scenario: {scenario_id}")
    print(f"[sim-glue] Description: {description}")
    print("[sim-glue] TODO: integrate with ns-O-RAN launcher here.")


if __name__ == "__main__":
    main()
