"""
E2 Message Definitions for UAV Beam xApp

Defines message formats for E2 interface communication:
- BeamIndication: Beam measurement reports from E2 nodes
- BeamControl: Beam control commands to E2 nodes
- Subscription management messages
- Message serialization/deserialization

Supports both JSON and binary (struct-based) encoding for flexibility.
"""

import struct
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any, Tuple, Union
from enum import IntEnum
import logging

logger = logging.getLogger(__name__)


# =============================================================================
# Message Type Definitions
# =============================================================================

class E2MessageType(IntEnum):
    """E2 message type identifiers"""
    # Indication messages (E2 Node -> xApp)
    BEAM_INDICATION = 0x01
    SUBSCRIPTION_RESPONSE = 0x02
    CONNECTION_ACK = 0x03
    HEARTBEAT_RESPONSE = 0x04
    ERROR_INDICATION = 0x05

    # Control messages (xApp -> E2 Node)
    BEAM_CONTROL = 0x11
    SUBSCRIPTION_REQUEST = 0x12
    CONNECTION_REQUEST = 0x13
    HEARTBEAT_REQUEST = 0x14
    SUBSCRIPTION_DELETE = 0x15

    # Internal messages
    UNKNOWN = 0xFF


class BeamAction(IntEnum):
    """Beam control action types"""
    MAINTAIN = 0x00
    SWITCH = 0x01
    REFINE = 0x02
    RECOVER = 0x03
    SWEEP = 0x04


class SubscriptionStatus(IntEnum):
    """Subscription status codes"""
    SUCCESS = 0x00
    FAILURE = 0x01
    TIMEOUT = 0x02
    DUPLICATE = 0x03
    NOT_FOUND = 0x04
    RESOURCE_UNAVAILABLE = 0x05


# =============================================================================
# Message Header
# =============================================================================

@dataclass
class E2MessageHeader:
    """
    Common header for all E2 messages

    Binary format (16 bytes):
    - magic (4 bytes): 0xE2E2E2E2
    - version (1 byte): Protocol version
    - message_type (1 byte): E2MessageType
    - flags (2 bytes): Reserved for future use
    - sequence_num (4 bytes): Message sequence number
    - payload_length (4 bytes): Length of payload in bytes
    """
    MAGIC = 0xE2E2E2E2
    VERSION = 0x01
    HEADER_SIZE = 16
    HEADER_FORMAT = ">IBBHII"  # Big-endian

    message_type: E2MessageType
    sequence_num: int = 0
    flags: int = 0
    payload_length: int = 0

    def to_bytes(self) -> bytes:
        """Serialize header to bytes"""
        return struct.pack(
            self.HEADER_FORMAT,
            self.MAGIC,
            self.VERSION,
            int(self.message_type),
            self.flags,
            self.sequence_num,
            self.payload_length
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> "E2MessageHeader":
        """Deserialize header from bytes"""
        if len(data) < cls.HEADER_SIZE:
            raise ValueError(f"Header data too short: {len(data)} < {cls.HEADER_SIZE}")

        magic, version, msg_type, flags, seq_num, payload_len = struct.unpack(
            cls.HEADER_FORMAT, data[:cls.HEADER_SIZE]
        )

        if magic != cls.MAGIC:
            raise ValueError(f"Invalid magic number: 0x{magic:08X}")

        if version != cls.VERSION:
            raise ValueError(f"Unsupported protocol version: {version}")

        return cls(
            message_type=E2MessageType(msg_type),
            sequence_num=seq_num,
            flags=flags,
            payload_length=payload_len
        )


# =============================================================================
# Indication Messages (E2 Node -> xApp)
# =============================================================================

@dataclass
class BeamIndication:
    """
    Beam measurement indication from E2 node

    Contains beam RSRP measurements and optional position/velocity data.
    Equivalent to E2SM-KPM indication for beam measurements.
    """
    ue_id: str
    timestamp_ms: float
    serving_cell_id: str
    serving_beam_id: int
    beam_rsrp_dbm: float
    neighbor_beams: Dict[int, float] = field(default_factory=dict)
    position: Optional[Tuple[float, float, float]] = None  # (x, y, z) in meters
    velocity: Optional[Tuple[float, float, float]] = None  # (vx, vy, vz) in m/s
    cqi: int = 0
    sinr_db: Optional[float] = None
    rnti: Optional[int] = None

    def to_json(self) -> str:
        """Serialize to JSON string"""
        data = asdict(self)
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> "BeamIndication":
        """Deserialize from JSON string"""
        data = json.loads(json_str)
        # Convert lists back to tuples for position/velocity
        if data.get("position"):
            data["position"] = tuple(data["position"])
        if data.get("velocity"):
            data["velocity"] = tuple(data["velocity"])
        # Convert neighbor_beams keys to int
        if data.get("neighbor_beams"):
            data["neighbor_beams"] = {
                int(k): float(v) for k, v in data["neighbor_beams"].items()
            }
        return cls(**data)

    def to_bytes(self) -> bytes:
        """
        Serialize to binary format

        Format:
        - ue_id_len (2 bytes) + ue_id (variable)
        - timestamp_ms (8 bytes, double)
        - cell_id_len (2 bytes) + cell_id (variable)
        - serving_beam_id (4 bytes)
        - beam_rsrp_dbm (8 bytes, double)
        - num_neighbors (2 bytes) + neighbor data
        - has_position (1 byte) + position (24 bytes if present)
        - has_velocity (1 byte) + velocity (24 bytes if present)
        - cqi (1 byte)
        - has_sinr (1 byte) + sinr (8 bytes if present)
        - has_rnti (1 byte) + rnti (2 bytes if present)
        """
        parts = []

        # UE ID
        ue_id_bytes = self.ue_id.encode("utf-8")
        parts.append(struct.pack(">H", len(ue_id_bytes)))
        parts.append(ue_id_bytes)

        # Timestamp
        parts.append(struct.pack(">d", self.timestamp_ms))

        # Cell ID
        cell_id_bytes = self.serving_cell_id.encode("utf-8")
        parts.append(struct.pack(">H", len(cell_id_bytes)))
        parts.append(cell_id_bytes)

        # Serving beam
        parts.append(struct.pack(">I", self.serving_beam_id))
        parts.append(struct.pack(">d", self.beam_rsrp_dbm))

        # Neighbor beams
        parts.append(struct.pack(">H", len(self.neighbor_beams)))
        for beam_id, rsrp in self.neighbor_beams.items():
            parts.append(struct.pack(">Id", beam_id, rsrp))

        # Position
        if self.position:
            parts.append(struct.pack(">B", 1))
            parts.append(struct.pack(">ddd", *self.position))
        else:
            parts.append(struct.pack(">B", 0))

        # Velocity
        if self.velocity:
            parts.append(struct.pack(">B", 1))
            parts.append(struct.pack(">ddd", *self.velocity))
        else:
            parts.append(struct.pack(">B", 0))

        # CQI
        parts.append(struct.pack(">B", self.cqi))

        # SINR
        if self.sinr_db is not None:
            parts.append(struct.pack(">B", 1))
            parts.append(struct.pack(">d", self.sinr_db))
        else:
            parts.append(struct.pack(">B", 0))

        # RNTI
        if self.rnti is not None:
            parts.append(struct.pack(">B", 1))
            parts.append(struct.pack(">H", self.rnti))
        else:
            parts.append(struct.pack(">B", 0))

        return b"".join(parts)

    @classmethod
    def from_bytes(cls, data: bytes) -> "BeamIndication":
        """Deserialize from binary format"""
        offset = 0

        # UE ID
        ue_id_len = struct.unpack_from(">H", data, offset)[0]
        offset += 2
        ue_id = data[offset:offset + ue_id_len].decode("utf-8")
        offset += ue_id_len

        # Timestamp
        timestamp_ms = struct.unpack_from(">d", data, offset)[0]
        offset += 8

        # Cell ID
        cell_id_len = struct.unpack_from(">H", data, offset)[0]
        offset += 2
        serving_cell_id = data[offset:offset + cell_id_len].decode("utf-8")
        offset += cell_id_len

        # Serving beam
        serving_beam_id = struct.unpack_from(">I", data, offset)[0]
        offset += 4
        beam_rsrp_dbm = struct.unpack_from(">d", data, offset)[0]
        offset += 8

        # Neighbor beams
        num_neighbors = struct.unpack_from(">H", data, offset)[0]
        offset += 2
        neighbor_beams = {}
        for _ in range(num_neighbors):
            beam_id, rsrp = struct.unpack_from(">Id", data, offset)
            neighbor_beams[beam_id] = rsrp
            offset += 12

        # Position
        has_position = struct.unpack_from(">B", data, offset)[0]
        offset += 1
        position = None
        if has_position:
            position = struct.unpack_from(">ddd", data, offset)
            offset += 24

        # Velocity
        has_velocity = struct.unpack_from(">B", data, offset)[0]
        offset += 1
        velocity = None
        if has_velocity:
            velocity = struct.unpack_from(">ddd", data, offset)
            offset += 24

        # CQI
        cqi = struct.unpack_from(">B", data, offset)[0]
        offset += 1

        # SINR
        has_sinr = struct.unpack_from(">B", data, offset)[0]
        offset += 1
        sinr_db = None
        if has_sinr:
            sinr_db = struct.unpack_from(">d", data, offset)[0]
            offset += 8

        # RNTI
        has_rnti = struct.unpack_from(">B", data, offset)[0]
        offset += 1
        rnti = None
        if has_rnti:
            rnti = struct.unpack_from(">H", data, offset)[0]

        return cls(
            ue_id=ue_id,
            timestamp_ms=timestamp_ms,
            serving_cell_id=serving_cell_id,
            serving_beam_id=serving_beam_id,
            beam_rsrp_dbm=beam_rsrp_dbm,
            neighbor_beams=neighbor_beams,
            position=position,
            velocity=velocity,
            cqi=cqi,
            sinr_db=sinr_db,
            rnti=rnti
        )


# =============================================================================
# Control Messages (xApp -> E2 Node)
# =============================================================================

@dataclass
class BeamControl:
    """
    Beam control command to E2 node

    Equivalent to E2SM-RC control message for beam management.
    """
    ue_id: str
    timestamp_ms: float
    target_beam_id: int
    action: BeamAction
    cell_id: str = ""
    confidence: float = 1.0
    predicted_rsrp_dbm: float = -100.0
    reason: str = ""
    priority: int = 0  # 0 = normal, 1 = high, 2 = urgent

    def to_json(self) -> str:
        """Serialize to JSON string"""
        data = asdict(self)
        data["action"] = int(self.action)
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> "BeamControl":
        """Deserialize from JSON string"""
        data = json.loads(json_str)
        data["action"] = BeamAction(data["action"])
        return cls(**data)

    def to_bytes(self) -> bytes:
        """Serialize to binary format"""
        parts = []

        # UE ID
        ue_id_bytes = self.ue_id.encode("utf-8")
        parts.append(struct.pack(">H", len(ue_id_bytes)))
        parts.append(ue_id_bytes)

        # Timestamp
        parts.append(struct.pack(">d", self.timestamp_ms))

        # Target beam and action
        parts.append(struct.pack(">IB", self.target_beam_id, int(self.action)))

        # Cell ID
        cell_id_bytes = self.cell_id.encode("utf-8")
        parts.append(struct.pack(">H", len(cell_id_bytes)))
        parts.append(cell_id_bytes)

        # Confidence and predicted RSRP
        parts.append(struct.pack(">dd", self.confidence, self.predicted_rsrp_dbm))

        # Reason
        reason_bytes = self.reason.encode("utf-8")
        parts.append(struct.pack(">H", len(reason_bytes)))
        parts.append(reason_bytes)

        # Priority
        parts.append(struct.pack(">B", self.priority))

        return b"".join(parts)

    @classmethod
    def from_bytes(cls, data: bytes) -> "BeamControl":
        """Deserialize from binary format"""
        offset = 0

        # UE ID
        ue_id_len = struct.unpack_from(">H", data, offset)[0]
        offset += 2
        ue_id = data[offset:offset + ue_id_len].decode("utf-8")
        offset += ue_id_len

        # Timestamp
        timestamp_ms = struct.unpack_from(">d", data, offset)[0]
        offset += 8

        # Target beam and action
        target_beam_id, action = struct.unpack_from(">IB", data, offset)
        offset += 5

        # Cell ID
        cell_id_len = struct.unpack_from(">H", data, offset)[0]
        offset += 2
        cell_id = data[offset:offset + cell_id_len].decode("utf-8")
        offset += cell_id_len

        # Confidence and predicted RSRP
        confidence, predicted_rsrp_dbm = struct.unpack_from(">dd", data, offset)
        offset += 16

        # Reason
        reason_len = struct.unpack_from(">H", data, offset)[0]
        offset += 2
        reason = data[offset:offset + reason_len].decode("utf-8")
        offset += reason_len

        # Priority
        priority = struct.unpack_from(">B", data, offset)[0]

        return cls(
            ue_id=ue_id,
            timestamp_ms=timestamp_ms,
            target_beam_id=target_beam_id,
            action=BeamAction(action),
            cell_id=cell_id,
            confidence=confidence,
            predicted_rsrp_dbm=predicted_rsrp_dbm,
            reason=reason,
            priority=priority
        )


# =============================================================================
# Subscription Messages
# =============================================================================

@dataclass
class SubscriptionRequest:
    """
    E2 subscription request

    Requests periodic beam indication reports from E2 node.
    """
    subscription_id: str
    xapp_id: str
    cell_ids: List[str] = field(default_factory=list)  # Empty = all cells
    reporting_period_ms: int = 10  # Report interval
    event_trigger_type: str = "periodic"  # "periodic", "threshold", "change"
    threshold_rsrp_db: Optional[float] = None  # For threshold-based trigger
    include_position: bool = True
    include_velocity: bool = True

    def to_json(self) -> str:
        """Serialize to JSON string"""
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> "SubscriptionRequest":
        """Deserialize from JSON string"""
        return cls(**json.loads(json_str))

    def to_bytes(self) -> bytes:
        """Serialize to binary format"""
        parts = []

        # Subscription ID
        sub_id_bytes = self.subscription_id.encode("utf-8")
        parts.append(struct.pack(">H", len(sub_id_bytes)))
        parts.append(sub_id_bytes)

        # xApp ID
        xapp_id_bytes = self.xapp_id.encode("utf-8")
        parts.append(struct.pack(">H", len(xapp_id_bytes)))
        parts.append(xapp_id_bytes)

        # Cell IDs
        parts.append(struct.pack(">H", len(self.cell_ids)))
        for cell_id in self.cell_ids:
            cell_bytes = cell_id.encode("utf-8")
            parts.append(struct.pack(">H", len(cell_bytes)))
            parts.append(cell_bytes)

        # Reporting period
        parts.append(struct.pack(">I", self.reporting_period_ms))

        # Event trigger type
        trigger_bytes = self.event_trigger_type.encode("utf-8")
        parts.append(struct.pack(">H", len(trigger_bytes)))
        parts.append(trigger_bytes)

        # Threshold
        if self.threshold_rsrp_db is not None:
            parts.append(struct.pack(">B", 1))
            parts.append(struct.pack(">d", self.threshold_rsrp_db))
        else:
            parts.append(struct.pack(">B", 0))

        # Flags
        flags = (1 if self.include_position else 0) | (2 if self.include_velocity else 0)
        parts.append(struct.pack(">B", flags))

        return b"".join(parts)

    @classmethod
    def from_bytes(cls, data: bytes) -> "SubscriptionRequest":
        """Deserialize from binary format"""
        offset = 0

        # Subscription ID
        sub_id_len = struct.unpack_from(">H", data, offset)[0]
        offset += 2
        subscription_id = data[offset:offset + sub_id_len].decode("utf-8")
        offset += sub_id_len

        # xApp ID
        xapp_id_len = struct.unpack_from(">H", data, offset)[0]
        offset += 2
        xapp_id = data[offset:offset + xapp_id_len].decode("utf-8")
        offset += xapp_id_len

        # Cell IDs
        num_cells = struct.unpack_from(">H", data, offset)[0]
        offset += 2
        cell_ids = []
        for _ in range(num_cells):
            cell_len = struct.unpack_from(">H", data, offset)[0]
            offset += 2
            cell_ids.append(data[offset:offset + cell_len].decode("utf-8"))
            offset += cell_len

        # Reporting period
        reporting_period_ms = struct.unpack_from(">I", data, offset)[0]
        offset += 4

        # Event trigger type
        trigger_len = struct.unpack_from(">H", data, offset)[0]
        offset += 2
        event_trigger_type = data[offset:offset + trigger_len].decode("utf-8")
        offset += trigger_len

        # Threshold
        has_threshold = struct.unpack_from(">B", data, offset)[0]
        offset += 1
        threshold_rsrp_db = None
        if has_threshold:
            threshold_rsrp_db = struct.unpack_from(">d", data, offset)[0]
            offset += 8

        # Flags
        flags = struct.unpack_from(">B", data, offset)[0]
        include_position = bool(flags & 1)
        include_velocity = bool(flags & 2)

        return cls(
            subscription_id=subscription_id,
            xapp_id=xapp_id,
            cell_ids=cell_ids,
            reporting_period_ms=reporting_period_ms,
            event_trigger_type=event_trigger_type,
            threshold_rsrp_db=threshold_rsrp_db,
            include_position=include_position,
            include_velocity=include_velocity
        )


@dataclass
class SubscriptionResponse:
    """
    E2 subscription response

    Confirms or rejects a subscription request.
    """
    subscription_id: str
    status: SubscriptionStatus
    e2_node_id: str = ""
    error_message: str = ""

    def to_json(self) -> str:
        """Serialize to JSON string"""
        data = asdict(self)
        data["status"] = int(self.status)
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> "SubscriptionResponse":
        """Deserialize from JSON string"""
        data = json.loads(json_str)
        data["status"] = SubscriptionStatus(data["status"])
        return cls(**data)

    def to_bytes(self) -> bytes:
        """Serialize to binary format"""
        parts = []

        # Subscription ID
        sub_id_bytes = self.subscription_id.encode("utf-8")
        parts.append(struct.pack(">H", len(sub_id_bytes)))
        parts.append(sub_id_bytes)

        # Status
        parts.append(struct.pack(">B", int(self.status)))

        # E2 Node ID
        node_id_bytes = self.e2_node_id.encode("utf-8")
        parts.append(struct.pack(">H", len(node_id_bytes)))
        parts.append(node_id_bytes)

        # Error message
        error_bytes = self.error_message.encode("utf-8")
        parts.append(struct.pack(">H", len(error_bytes)))
        parts.append(error_bytes)

        return b"".join(parts)

    @classmethod
    def from_bytes(cls, data: bytes) -> "SubscriptionResponse":
        """Deserialize from binary format"""
        offset = 0

        # Subscription ID
        sub_id_len = struct.unpack_from(">H", data, offset)[0]
        offset += 2
        subscription_id = data[offset:offset + sub_id_len].decode("utf-8")
        offset += sub_id_len

        # Status
        status = SubscriptionStatus(struct.unpack_from(">B", data, offset)[0])
        offset += 1

        # E2 Node ID
        node_id_len = struct.unpack_from(">H", data, offset)[0]
        offset += 2
        e2_node_id = data[offset:offset + node_id_len].decode("utf-8")
        offset += node_id_len

        # Error message
        error_len = struct.unpack_from(">H", data, offset)[0]
        offset += 2
        error_message = data[offset:offset + error_len].decode("utf-8")

        return cls(
            subscription_id=subscription_id,
            status=status,
            e2_node_id=e2_node_id,
            error_message=error_message
        )


# =============================================================================
# Connection and Heartbeat Messages
# =============================================================================

@dataclass
class ConnectionRequest:
    """Connection request to E2 termination"""
    xapp_id: str
    xapp_name: str = "uav-beam-xapp"
    version: str = "0.1.0"
    capabilities: List[str] = field(default_factory=lambda: ["beam-tracking", "beam-control"])

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> "ConnectionRequest":
        return cls(**json.loads(json_str))


@dataclass
class ConnectionAck:
    """Connection acknowledgment from E2 termination"""
    success: bool
    e2_term_id: str = ""
    session_id: str = ""
    error_message: str = ""
    available_e2_nodes: List[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> "ConnectionAck":
        return cls(**json.loads(json_str))


@dataclass
class HeartbeatRequest:
    """Heartbeat request"""
    xapp_id: str
    timestamp_ms: float = field(default_factory=lambda: time.time() * 1000)
    sequence_num: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> "HeartbeatRequest":
        return cls(**json.loads(json_str))


@dataclass
class HeartbeatResponse:
    """Heartbeat response"""
    e2_term_id: str
    timestamp_ms: float
    sequence_num: int
    healthy: bool = True

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, json_str: str) -> "HeartbeatResponse":
        return cls(**json.loads(json_str))


# =============================================================================
# Message Encoding/Decoding Functions
# =============================================================================

class E2MessageCodec:
    """
    Codec for encoding/decoding E2 messages

    Supports both JSON and binary formats.
    """

    @staticmethod
    def encode_message(
        message: Union[BeamIndication, BeamControl, SubscriptionRequest,
                       SubscriptionResponse, ConnectionRequest, ConnectionAck,
                       HeartbeatRequest, HeartbeatResponse],
        sequence_num: int = 0,
        use_binary: bool = False
    ) -> bytes:
        """
        Encode a message with header

        Args:
            message: Message to encode
            sequence_num: Sequence number for ordering
            use_binary: Use binary format (True) or JSON (False)

        Returns:
            Encoded message bytes
        """
        # Determine message type
        msg_type_map = {
            BeamIndication: E2MessageType.BEAM_INDICATION,
            BeamControl: E2MessageType.BEAM_CONTROL,
            SubscriptionRequest: E2MessageType.SUBSCRIPTION_REQUEST,
            SubscriptionResponse: E2MessageType.SUBSCRIPTION_RESPONSE,
            ConnectionRequest: E2MessageType.CONNECTION_REQUEST,
            ConnectionAck: E2MessageType.CONNECTION_ACK,
            HeartbeatRequest: E2MessageType.HEARTBEAT_REQUEST,
            HeartbeatResponse: E2MessageType.HEARTBEAT_RESPONSE,
        }

        msg_type = msg_type_map.get(type(message), E2MessageType.UNKNOWN)

        # Encode payload
        if use_binary and hasattr(message, "to_bytes"):
            payload = message.to_bytes()
            flags = 0x01  # Binary flag
        else:
            payload = message.to_json().encode("utf-8")
            flags = 0x00  # JSON flag

        # Create header
        header = E2MessageHeader(
            message_type=msg_type,
            sequence_num=sequence_num,
            flags=flags,
            payload_length=len(payload)
        )

        return header.to_bytes() + payload

    @staticmethod
    def decode_message(data: bytes) -> Tuple[E2MessageHeader, Any]:
        """
        Decode a message with header

        Args:
            data: Raw message bytes

        Returns:
            Tuple of (header, decoded message)
        """
        # Parse header
        header = E2MessageHeader.from_bytes(data)
        payload = data[E2MessageHeader.HEADER_SIZE:E2MessageHeader.HEADER_SIZE + header.payload_length]

        is_binary = header.flags & 0x01

        # Decode based on message type
        msg_class_map = {
            E2MessageType.BEAM_INDICATION: BeamIndication,
            E2MessageType.BEAM_CONTROL: BeamControl,
            E2MessageType.SUBSCRIPTION_REQUEST: SubscriptionRequest,
            E2MessageType.SUBSCRIPTION_RESPONSE: SubscriptionResponse,
            E2MessageType.CONNECTION_REQUEST: ConnectionRequest,
            E2MessageType.CONNECTION_ACK: ConnectionAck,
            E2MessageType.HEARTBEAT_REQUEST: HeartbeatRequest,
            E2MessageType.HEARTBEAT_RESPONSE: HeartbeatResponse,
        }

        msg_class = msg_class_map.get(header.message_type)
        if msg_class is None:
            raise ValueError(f"Unknown message type: {header.message_type}")

        if is_binary and hasattr(msg_class, "from_bytes"):
            message = msg_class.from_bytes(payload)
        else:
            message = msg_class.from_json(payload.decode("utf-8"))

        return header, message


# =============================================================================
# Utility Functions
# =============================================================================

def create_beam_indication(
    ue_id: str,
    serving_beam_id: int,
    beam_rsrp_dbm: float,
    cell_id: str = "cell-1",
    neighbor_beams: Optional[Dict[int, float]] = None,
    position: Optional[Tuple[float, float, float]] = None,
    velocity: Optional[Tuple[float, float, float]] = None
) -> BeamIndication:
    """Convenience function to create BeamIndication"""
    return BeamIndication(
        ue_id=ue_id,
        timestamp_ms=time.time() * 1000,
        serving_cell_id=cell_id,
        serving_beam_id=serving_beam_id,
        beam_rsrp_dbm=beam_rsrp_dbm,
        neighbor_beams=neighbor_beams or {},
        position=position,
        velocity=velocity
    )


def create_beam_control(
    ue_id: str,
    target_beam_id: int,
    action: BeamAction,
    confidence: float = 1.0,
    reason: str = ""
) -> BeamControl:
    """Convenience function to create BeamControl"""
    return BeamControl(
        ue_id=ue_id,
        timestamp_ms=time.time() * 1000,
        target_beam_id=target_beam_id,
        action=action,
        confidence=confidence,
        reason=reason
    )


def indication_to_dict(indication: BeamIndication) -> Dict[str, Any]:
    """Convert BeamIndication to dict format expected by server.py"""
    return {
        "ue_id": indication.ue_id,
        "timestamp_ms": indication.timestamp_ms,
        "serving_cell_id": indication.serving_cell_id,
        "serving_beam_id": indication.serving_beam_id,
        "beam_rsrp_dbm": indication.beam_rsrp_dbm,
        "neighbor_beams": {str(k): v for k, v in indication.neighbor_beams.items()},
        "position": list(indication.position) if indication.position else None,
        "velocity": list(indication.velocity) if indication.velocity else None,
        "cqi": indication.cqi,
    }


def dict_to_indication(data: Dict[str, Any]) -> BeamIndication:
    """Convert dict to BeamIndication"""
    return BeamIndication(
        ue_id=data["ue_id"],
        timestamp_ms=data.get("timestamp_ms", time.time() * 1000),
        serving_cell_id=data.get("serving_cell_id", "cell-1"),
        serving_beam_id=data["serving_beam_id"],
        beam_rsrp_dbm=data["beam_rsrp_dbm"],
        neighbor_beams={
            int(k): float(v) for k, v in data.get("neighbor_beams", {}).items()
        },
        position=tuple(data["position"]) if data.get("position") else None,
        velocity=tuple(data["velocity"]) if data.get("velocity") else None,
        cqi=data.get("cqi", 0),
        sinr_db=data.get("sinr_db"),
        rnti=data.get("rnti")
    )
