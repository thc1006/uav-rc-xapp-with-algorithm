"""
Shared UAV State Module

Provides a common interface for UAV state management
that can be used by both UAV Policy xApp and UAV Beam xApp.

This enables coordination between xApps via:
1. Shared Data Layer (SDL) - Redis-based
2. REST API polling
3. Message queue (future)
"""

import time
import json
import logging
from dataclasses import dataclass, asdict, field
from typing import Optional, Dict, List, Any
from enum import Enum
import threading

logger = logging.getLogger(__name__)


class UAVStatus(Enum):
    """UAV operational status"""
    UNKNOWN = "unknown"
    IDLE = "idle"
    FLYING = "flying"
    HOVERING = "hovering"
    LANDING = "landing"
    EMERGENCY = "emergency"


class ConnectionStatus(Enum):
    """Radio connection status"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    HANDOVER = "handover"
    BEAM_FAILURE = "beam_failure"


@dataclass
class UAVPosition:
    """UAV 3D position"""
    x: float = 0.0      # meters (East)
    y: float = 0.0      # meters (North)
    z: float = 0.0      # meters (altitude AGL)
    latitude: Optional[float] = None   # degrees
    longitude: Optional[float] = None  # degrees
    altitude_msl: Optional[float] = None  # meters MSL

    def to_array(self) -> List[float]:
        return [self.x, self.y, self.z]

    @classmethod
    def from_array(cls, arr: List[float]) -> "UAVPosition":
        return cls(x=arr[0], y=arr[1], z=arr[2])


@dataclass
class UAVVelocity:
    """UAV velocity vector"""
    vx: float = 0.0     # m/s (East)
    vy: float = 0.0     # m/s (North)
    vz: float = 0.0     # m/s (Up)

    @property
    def speed(self) -> float:
        return (self.vx**2 + self.vy**2 + self.vz**2) ** 0.5

    def to_array(self) -> List[float]:
        return [self.vx, self.vy, self.vz]

    @classmethod
    def from_array(cls, arr: List[float]) -> "UAVVelocity":
        return cls(vx=arr[0], vy=arr[1], vz=arr[2])


@dataclass
class RadioState:
    """Radio connection state"""
    # LTE/NR cell info
    serving_cell_id: str = ""
    serving_enb_id: str = ""

    # Signal quality
    rsrp_dbm: float = -140.0
    rsrq_db: float = -20.0
    sinr_db: float = 0.0
    cqi: int = 0

    # Resource allocation
    allocated_prbs: int = 0
    total_prbs: int = 100

    # Beam info (for mmWave)
    serving_beam_id: int = 0
    beam_rsrp_dbm: float = -140.0

    # Connection status
    status: ConnectionStatus = ConnectionStatus.DISCONNECTED


@dataclass
class UAVState:
    """Complete UAV state"""
    uav_id: str
    timestamp_ms: float = field(default_factory=lambda: time.time() * 1000)

    # Kinematic state
    position: UAVPosition = field(default_factory=UAVPosition)
    velocity: UAVVelocity = field(default_factory=UAVVelocity)
    heading_deg: float = 0.0

    # Operational state
    status: UAVStatus = UAVStatus.UNKNOWN
    battery_percent: float = 100.0

    # Radio state
    radio: RadioState = field(default_factory=RadioState)

    # Mission info
    mission_id: Optional[str] = None
    waypoint_index: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return {
            "uav_id": self.uav_id,
            "timestamp_ms": self.timestamp_ms,
            "position": asdict(self.position),
            "velocity": asdict(self.velocity),
            "heading_deg": self.heading_deg,
            "status": self.status.value,
            "battery_percent": self.battery_percent,
            "radio": {
                **asdict(self.radio),
                "status": self.radio.status.value,
            },
            "mission_id": self.mission_id,
            "waypoint_index": self.waypoint_index,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UAVState":
        """Create from dictionary"""
        position = UAVPosition(**data.get("position", {}))
        velocity = UAVVelocity(**data.get("velocity", {}))

        radio_data = data.get("radio", {})
        radio_status = radio_data.pop("status", "disconnected")
        radio = RadioState(
            **radio_data,
            status=ConnectionStatus(radio_status)
        )

        return cls(
            uav_id=data["uav_id"],
            timestamp_ms=data.get("timestamp_ms", time.time() * 1000),
            position=position,
            velocity=velocity,
            heading_deg=data.get("heading_deg", 0.0),
            status=UAVStatus(data.get("status", "unknown")),
            battery_percent=data.get("battery_percent", 100.0),
            radio=radio,
            mission_id=data.get("mission_id"),
            waypoint_index=data.get("waypoint_index", 0),
        )


class UAVStateStore:
    """
    UAV State Storage

    In-memory store with optional SDL (Redis) persistence.
    Provides thread-safe access to UAV states.
    """

    def __init__(self, use_sdl: bool = False, redis_host: str = "localhost", redis_port: int = 6379):
        self._states: Dict[str, UAVState] = {}
        self._lock = threading.RLock()
        self._use_sdl = use_sdl
        self._redis_client = None

        if use_sdl:
            try:
                import redis
                self._redis_client = redis.Redis(host=redis_host, port=redis_port, decode_responses=True)
                self._redis_client.ping()
                logger.info(f"Connected to Redis SDL at {redis_host}:{redis_port}")
            except Exception as e:
                logger.warning(f"Failed to connect to Redis SDL: {e}")
                self._use_sdl = False

        self._subscribers: List[callable] = []

    def update(self, state: UAVState):
        """Update UAV state"""
        with self._lock:
            state.timestamp_ms = time.time() * 1000
            self._states[state.uav_id] = state

            if self._use_sdl and self._redis_client:
                key = f"uav:state:{state.uav_id}"
                self._redis_client.set(key, json.dumps(state.to_dict()))
                self._redis_client.expire(key, 60)  # 60s TTL

        # Notify subscribers
        for subscriber in self._subscribers:
            try:
                subscriber(state)
            except Exception as e:
                logger.error(f"Subscriber error: {e}")

    def get(self, uav_id: str) -> Optional[UAVState]:
        """Get UAV state"""
        with self._lock:
            # Try local cache first
            if uav_id in self._states:
                return self._states[uav_id]

            # Try SDL
            if self._use_sdl and self._redis_client:
                key = f"uav:state:{uav_id}"
                data = self._redis_client.get(key)
                if data:
                    state = UAVState.from_dict(json.loads(data))
                    self._states[uav_id] = state
                    return state

            return None

    def get_all(self) -> Dict[str, UAVState]:
        """Get all UAV states"""
        with self._lock:
            return dict(self._states)

    def remove(self, uav_id: str):
        """Remove UAV state"""
        with self._lock:
            self._states.pop(uav_id, None)

            if self._use_sdl and self._redis_client:
                key = f"uav:state:{uav_id}"
                self._redis_client.delete(key)

    def subscribe(self, callback: callable):
        """Subscribe to state updates"""
        self._subscribers.append(callback)

    def unsubscribe(self, callback: callable):
        """Unsubscribe from state updates"""
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def get_uav_ids(self) -> List[str]:
        """Get list of tracked UAV IDs"""
        with self._lock:
            return list(self._states.keys())


# Global state store instance
_global_store: Optional[UAVStateStore] = None


def get_state_store(use_sdl: bool = False) -> UAVStateStore:
    """Get or create global state store"""
    global _global_store
    if _global_store is None:
        _global_store = UAVStateStore(use_sdl=use_sdl)
    return _global_store


# Utility functions for xApp coordination

def build_policy_indication(state: UAVState) -> Dict[str, Any]:
    """Build indication message for UAV Policy xApp"""
    return {
        "ue_id": state.uav_id,
        "timestamp_ms": state.timestamp_ms,
        "cell_id": state.radio.serving_cell_id,
        "rsrp": state.radio.rsrp_dbm,
        "sinr": state.radio.sinr_db,
        "cqi": state.radio.cqi,
        "prb_usage": state.radio.allocated_prbs / max(state.radio.total_prbs, 1),
        "position": state.position.to_array(),
        "velocity": state.velocity.to_array(),
        "battery_level": state.battery_percent,
    }


def build_beam_indication(state: UAVState) -> Dict[str, Any]:
    """Build indication message for UAV Beam xApp"""
    return {
        "ue_id": state.uav_id,
        "timestamp_ms": state.timestamp_ms,
        "serving_cell_id": state.radio.serving_cell_id,
        "serving_beam_id": state.radio.serving_beam_id,
        "beam_rsrp_dbm": state.radio.beam_rsrp_dbm,
        "neighbor_beams": {},  # Would be populated from actual measurements
        "position": state.position.to_array(),
        "velocity": state.velocity.to_array(),
    }
