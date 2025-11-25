"""
Shared modules for UAV xApps

Provides common interfaces for:
- UAV state management
- xApp coordination
- SDL integration
"""

from .uav_state import (
    UAVStatus,
    ConnectionStatus,
    UAVPosition,
    UAVVelocity,
    RadioState,
    UAVState,
    UAVStateStore,
    get_state_store,
    build_policy_indication,
    build_beam_indication,
)

__all__ = [
    "UAVStatus",
    "ConnectionStatus",
    "UAVPosition",
    "UAVVelocity",
    "RadioState",
    "UAVState",
    "UAVStateStore",
    "get_state_store",
    "build_policy_indication",
    "build_beam_indication",
]
