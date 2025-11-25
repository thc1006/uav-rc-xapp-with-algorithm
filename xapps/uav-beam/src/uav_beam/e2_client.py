"""
E2 Socket Client for UAV Beam xApp

Provides asynchronous TCP socket communication with E2 termination layer.

Features:
- Async I/O using asyncio
- Connection pooling for multiple E2 nodes
- Automatic reconnection with exponential backoff
- Heartbeat mechanism for connection health
- Message queuing and flow control
- Metrics collection (latency, message counts)
"""

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, Optional, Callable, Any, List, Awaitable
from enum import Enum
from collections import deque
import json

from .e2_messages import (
    E2MessageHeader,
    E2MessageType,
    E2MessageCodec,
    BeamIndication,
    BeamControl,
    SubscriptionRequest,
    SubscriptionResponse,
    ConnectionRequest,
    ConnectionAck,
    HeartbeatRequest,
    HeartbeatResponse,
    SubscriptionStatus,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Configuration and Types
# =============================================================================

class ConnectionState(Enum):
    """E2 connection state"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    AUTHENTICATED = "authenticated"
    SUBSCRIBED = "subscribed"
    ERROR = "error"
    CLOSING = "closing"


@dataclass
class E2ClientConfig:
    """E2 client configuration"""
    # Connection settings
    host: str = "localhost"
    port: int = 36422  # Default E2 port
    connection_timeout: float = 10.0
    read_timeout: float = 30.0

    # Reconnection settings
    auto_reconnect: bool = True
    reconnect_delay_initial: float = 1.0
    reconnect_delay_max: float = 60.0
    reconnect_max_attempts: int = 0  # 0 = unlimited

    # Heartbeat settings
    heartbeat_interval: float = 10.0
    heartbeat_timeout: float = 5.0
    max_missed_heartbeats: int = 3

    # Buffer settings
    read_buffer_size: int = 65536
    write_buffer_size: int = 65536
    max_queue_size: int = 1000

    # Message settings
    use_binary_encoding: bool = False  # JSON by default

    # xApp identity
    xapp_id: str = field(default_factory=lambda: f"uav-beam-{uuid.uuid4().hex[:8]}")
    xapp_name: str = "uav-beam-xapp"


@dataclass
class E2Metrics:
    """Metrics for E2 connection"""
    messages_sent: int = 0
    messages_received: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    indications_received: int = 0
    controls_sent: int = 0
    subscriptions_active: int = 0
    connection_errors: int = 0
    reconnection_count: int = 0
    heartbeats_sent: int = 0
    heartbeats_received: int = 0
    avg_latency_ms: float = 0.0
    last_indication_time: float = 0.0
    last_control_time: float = 0.0

    # Latency tracking
    _latency_samples: List[float] = field(default_factory=list)

    def record_latency(self, latency_ms: float):
        """Record a latency sample"""
        self._latency_samples.append(latency_ms)
        # Keep last 100 samples
        if len(self._latency_samples) > 100:
            self._latency_samples = self._latency_samples[-100:]
        self.avg_latency_ms = sum(self._latency_samples) / len(self._latency_samples)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "messages_sent": self.messages_sent,
            "messages_received": self.messages_received,
            "bytes_sent": self.bytes_sent,
            "bytes_received": self.bytes_received,
            "indications_received": self.indications_received,
            "controls_sent": self.controls_sent,
            "subscriptions_active": self.subscriptions_active,
            "connection_errors": self.connection_errors,
            "reconnection_count": self.reconnection_count,
            "heartbeats_sent": self.heartbeats_sent,
            "heartbeats_received": self.heartbeats_received,
            "avg_latency_ms": self.avg_latency_ms,
            "last_indication_time": self.last_indication_time,
            "last_control_time": self.last_control_time,
        }


# =============================================================================
# E2 Connection
# =============================================================================

class E2Connection:
    """
    Single E2 connection to an E2 termination endpoint

    Handles connection lifecycle, message sending/receiving,
    and heartbeat management.
    """

    def __init__(
        self,
        config: E2ClientConfig,
        on_indication: Optional[Callable[[BeamIndication], Awaitable[None]]] = None,
        on_disconnect: Optional[Callable[[], Awaitable[None]]] = None,
    ):
        self.config = config
        self.on_indication = on_indication
        self.on_disconnect = on_disconnect

        # Connection state
        self.state = ConnectionState.DISCONNECTED
        self.session_id: Optional[str] = None
        self.e2_term_id: Optional[str] = None

        # Asyncio components
        self._reader: Optional[asyncio.StreamReader] = None
        self._writer: Optional[asyncio.StreamWriter] = None
        self._read_task: Optional[asyncio.Task] = None
        self._heartbeat_task: Optional[asyncio.Task] = None
        self._reconnect_task: Optional[asyncio.Task] = None

        # Message handling
        self._sequence_num = 0
        self._pending_responses: Dict[int, asyncio.Future] = {}
        self._send_queue: deque = deque(maxlen=config.max_queue_size)
        self._send_lock = asyncio.Lock()

        # Subscriptions
        self._subscriptions: Dict[str, SubscriptionRequest] = {}

        # Heartbeat tracking
        self._last_heartbeat_sent: float = 0
        self._last_heartbeat_recv: float = 0
        self._missed_heartbeats: int = 0

        # Reconnection
        self._reconnect_attempts: int = 0
        self._reconnect_delay: float = config.reconnect_delay_initial

        # Metrics
        self.metrics = E2Metrics()

        # Shutdown flag
        self._shutdown = False

    @property
    def is_connected(self) -> bool:
        """Check if connection is active"""
        return self.state in (
            ConnectionState.CONNECTED,
            ConnectionState.AUTHENTICATED,
            ConnectionState.SUBSCRIBED
        )

    async def connect(self) -> bool:
        """
        Establish connection to E2 termination

        Returns:
            True if connection successful, False otherwise
        """
        if self.state == ConnectionState.CONNECTING:
            logger.warning("Connection already in progress")
            return False

        self.state = ConnectionState.CONNECTING
        self._shutdown = False

        try:
            logger.info(f"Connecting to E2 termination at {self.config.host}:{self.config.port}")

            # Establish TCP connection
            self._reader, self._writer = await asyncio.wait_for(
                asyncio.open_connection(self.config.host, self.config.port),
                timeout=self.config.connection_timeout
            )

            self.state = ConnectionState.CONNECTED
            logger.info("TCP connection established")

            # Send connection request
            await self._authenticate()

            # Start background tasks
            self._read_task = asyncio.create_task(self._read_loop())
            self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

            # Reset reconnection state
            self._reconnect_attempts = 0
            self._reconnect_delay = self.config.reconnect_delay_initial

            return True

        except asyncio.TimeoutError:
            logger.error(f"Connection timeout after {self.config.connection_timeout}s")
            self.metrics.connection_errors += 1
            self.state = ConnectionState.ERROR
            return False

        except ConnectionRefusedError:
            logger.error(f"Connection refused by {self.config.host}:{self.config.port}")
            self.metrics.connection_errors += 1
            self.state = ConnectionState.ERROR
            return False

        except Exception as e:
            logger.error(f"Connection error: {e}")
            self.metrics.connection_errors += 1
            self.state = ConnectionState.ERROR
            return False

    async def _authenticate(self):
        """Send connection request and wait for acknowledgment"""
        request = ConnectionRequest(
            xapp_id=self.config.xapp_id,
            xapp_name=self.config.xapp_name,
        )

        response = await self._send_and_wait(
            request,
            E2MessageType.CONNECTION_ACK,
            timeout=self.config.connection_timeout
        )

        if response and isinstance(response, ConnectionAck):
            if response.success:
                self.session_id = response.session_id
                self.e2_term_id = response.e2_term_id
                self.state = ConnectionState.AUTHENTICATED
                logger.info(f"Authenticated with E2 termination: session={self.session_id}")
            else:
                raise ConnectionError(f"Authentication failed: {response.error_message}")
        else:
            # If no response, assume connected (for simpler E2 endpoints)
            self.state = ConnectionState.AUTHENTICATED
            logger.info("Connected to E2 termination (no auth required)")

    async def disconnect(self):
        """Gracefully disconnect from E2 termination"""
        if self.state == ConnectionState.DISCONNECTED:
            return

        logger.info("Disconnecting from E2 termination")
        self._shutdown = True
        self.state = ConnectionState.CLOSING

        # Cancel background tasks
        for task in [self._read_task, self._heartbeat_task, self._reconnect_task]:
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Close socket
        if self._writer:
            try:
                self._writer.close()
                await self._writer.wait_closed()
            except Exception as e:
                logger.debug(f"Error closing writer: {e}")

        self._reader = None
        self._writer = None
        self.state = ConnectionState.DISCONNECTED

        # Clear pending responses
        for future in self._pending_responses.values():
            if not future.done():
                future.cancel()
        self._pending_responses.clear()

        logger.info("Disconnected from E2 termination")

    async def subscribe(
        self,
        cell_ids: Optional[List[str]] = None,
        reporting_period_ms: int = 10
    ) -> Optional[SubscriptionResponse]:
        """
        Subscribe to beam indications from E2 nodes

        Args:
            cell_ids: List of cell IDs to subscribe to (None = all)
            reporting_period_ms: Reporting interval in milliseconds

        Returns:
            SubscriptionResponse if successful, None otherwise
        """
        if not self.is_connected:
            logger.error("Cannot subscribe: not connected")
            return None

        subscription_id = f"sub-{uuid.uuid4().hex[:8]}"

        request = SubscriptionRequest(
            subscription_id=subscription_id,
            xapp_id=self.config.xapp_id,
            cell_ids=cell_ids or [],
            reporting_period_ms=reporting_period_ms,
        )

        response = await self._send_and_wait(
            request,
            E2MessageType.SUBSCRIPTION_RESPONSE,
            timeout=self.config.read_timeout
        )

        if response and isinstance(response, SubscriptionResponse):
            if response.status == SubscriptionStatus.SUCCESS:
                self._subscriptions[subscription_id] = request
                self.metrics.subscriptions_active = len(self._subscriptions)
                self.state = ConnectionState.SUBSCRIBED
                logger.info(f"Subscription established: {subscription_id}")
                return response
            else:
                logger.error(f"Subscription failed: {response.error_message}")
                return response

        return None

    async def unsubscribe(self, subscription_id: str) -> bool:
        """
        Unsubscribe from beam indications

        Args:
            subscription_id: ID of subscription to remove

        Returns:
            True if successful
        """
        if subscription_id in self._subscriptions:
            del self._subscriptions[subscription_id]
            self.metrics.subscriptions_active = len(self._subscriptions)
            logger.info(f"Unsubscribed: {subscription_id}")
            return True
        return False

    async def send_control(self, control: BeamControl) -> bool:
        """
        Send beam control message to E2 node

        Args:
            control: BeamControl message to send

        Returns:
            True if sent successfully
        """
        if not self.is_connected:
            logger.warning("Cannot send control: not connected")
            return False

        try:
            await self._send_message(control)
            self.metrics.controls_sent += 1
            self.metrics.last_control_time = time.time()
            logger.debug(f"Sent beam control: {control.ue_id} -> beam {control.target_beam_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to send control: {e}")
            return False

    async def _send_message(self, message: Any):
        """Send a message through the socket"""
        async with self._send_lock:
            if not self._writer:
                raise ConnectionError("Not connected")

            self._sequence_num += 1
            encoded = E2MessageCodec.encode_message(
                message,
                sequence_num=self._sequence_num,
                use_binary=self.config.use_binary_encoding
            )

            self._writer.write(encoded)
            await self._writer.drain()

            self.metrics.messages_sent += 1
            self.metrics.bytes_sent += len(encoded)

    async def _send_and_wait(
        self,
        message: Any,
        expected_type: E2MessageType,
        timeout: float = 10.0
    ) -> Optional[Any]:
        """Send a message and wait for response"""
        self._sequence_num += 1
        seq_num = self._sequence_num

        # Create future for response
        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending_responses[seq_num] = future

        try:
            await self._send_message(message)

            # Wait for response
            response = await asyncio.wait_for(future, timeout=timeout)
            return response

        except asyncio.TimeoutError:
            logger.warning(f"Timeout waiting for response (seq={seq_num})")
            return None
        finally:
            self._pending_responses.pop(seq_num, None)

    async def _read_loop(self):
        """Background task to read incoming messages"""
        logger.debug("Starting read loop")

        while not self._shutdown and self._reader:
            try:
                # Read header first
                header_data = await asyncio.wait_for(
                    self._reader.read(E2MessageHeader.HEADER_SIZE),
                    timeout=self.config.read_timeout
                )

                if not header_data:
                    logger.warning("Connection closed by remote")
                    break

                if len(header_data) < E2MessageHeader.HEADER_SIZE:
                    logger.warning(f"Incomplete header: {len(header_data)} bytes")
                    continue

                # Parse header
                try:
                    header = E2MessageHeader.from_bytes(header_data)
                except ValueError as e:
                    logger.error(f"Invalid header: {e}")
                    continue

                # Read payload
                payload_data = await self._reader.read(header.payload_length)
                if len(payload_data) < header.payload_length:
                    logger.warning(f"Incomplete payload: {len(payload_data)} < {header.payload_length}")
                    continue

                # Decode message
                full_data = header_data + payload_data
                _, message = E2MessageCodec.decode_message(full_data)

                self.metrics.messages_received += 1
                self.metrics.bytes_received += len(full_data)

                # Handle message
                await self._handle_message(header, message)

            except asyncio.TimeoutError:
                # Timeout is normal, check heartbeat
                continue

            except asyncio.CancelledError:
                break

            except Exception as e:
                logger.error(f"Read error: {e}")
                self.metrics.connection_errors += 1
                break

        # Connection lost
        if not self._shutdown:
            await self._handle_disconnect()

    async def _handle_message(self, header: E2MessageHeader, message: Any):
        """Process a received message"""
        msg_type = header.message_type

        if msg_type == E2MessageType.BEAM_INDICATION:
            self.metrics.indications_received += 1
            self.metrics.last_indication_time = time.time()

            if self.on_indication:
                try:
                    await self.on_indication(message)
                except Exception as e:
                    logger.error(f"Error in indication handler: {e}")

        elif msg_type == E2MessageType.HEARTBEAT_RESPONSE:
            self._last_heartbeat_recv = time.time()
            self._missed_heartbeats = 0
            self.metrics.heartbeats_received += 1

            # Calculate latency
            if isinstance(message, HeartbeatResponse):
                latency = (time.time() * 1000) - message.timestamp_ms
                self.metrics.record_latency(latency)

        elif msg_type == E2MessageType.SUBSCRIPTION_RESPONSE:
            # Check if there's a pending request
            if header.sequence_num in self._pending_responses:
                future = self._pending_responses[header.sequence_num]
                if not future.done():
                    future.set_result(message)

        elif msg_type == E2MessageType.CONNECTION_ACK:
            if header.sequence_num in self._pending_responses:
                future = self._pending_responses[header.sequence_num]
                if not future.done():
                    future.set_result(message)

        elif msg_type == E2MessageType.ERROR_INDICATION:
            logger.error(f"Received error indication: {message}")

        else:
            logger.debug(f"Unhandled message type: {msg_type}")

    async def _heartbeat_loop(self):
        """Background task to send periodic heartbeats"""
        logger.debug("Starting heartbeat loop")
        heartbeat_seq = 0

        while not self._shutdown:
            try:
                await asyncio.sleep(self.config.heartbeat_interval)

                if not self.is_connected:
                    continue

                # Check missed heartbeats
                if self._missed_heartbeats >= self.config.max_missed_heartbeats:
                    logger.error("Too many missed heartbeats, reconnecting")
                    await self._handle_disconnect()
                    break

                # Send heartbeat
                heartbeat_seq += 1
                request = HeartbeatRequest(
                    xapp_id=self.config.xapp_id,
                    timestamp_ms=time.time() * 1000,
                    sequence_num=heartbeat_seq
                )

                await self._send_message(request)
                self._last_heartbeat_sent = time.time()
                self._missed_heartbeats += 1
                self.metrics.heartbeats_sent += 1

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Heartbeat error: {e}")

    async def _handle_disconnect(self):
        """Handle disconnection and trigger reconnect if configured"""
        if self._shutdown:
            return

        logger.warning("Connection lost")
        self.state = ConnectionState.DISCONNECTED

        if self.on_disconnect:
            try:
                await self.on_disconnect()
            except Exception as e:
                logger.error(f"Error in disconnect handler: {e}")

        # Attempt reconnection
        if self.config.auto_reconnect:
            self._reconnect_task = asyncio.create_task(self._reconnect_loop())

    async def _reconnect_loop(self):
        """Background task for reconnection with exponential backoff"""
        while not self._shutdown:
            self._reconnect_attempts += 1
            self.metrics.reconnection_count += 1

            # Check max attempts
            if (self.config.reconnect_max_attempts > 0 and
                    self._reconnect_attempts > self.config.reconnect_max_attempts):
                logger.error("Max reconnection attempts reached")
                self.state = ConnectionState.ERROR
                break

            logger.info(
                f"Reconnection attempt {self._reconnect_attempts} "
                f"in {self._reconnect_delay:.1f}s"
            )

            await asyncio.sleep(self._reconnect_delay)

            # Exponential backoff
            self._reconnect_delay = min(
                self._reconnect_delay * 2,
                self.config.reconnect_delay_max
            )

            # Attempt reconnection
            if await self.connect():
                # Re-establish subscriptions
                for sub_request in list(self._subscriptions.values()):
                    await self.subscribe(
                        cell_ids=sub_request.cell_ids,
                        reporting_period_ms=sub_request.reporting_period_ms
                    )
                break


# =============================================================================
# E2 Client (Connection Pool)
# =============================================================================

class E2Client:
    """
    E2 Client with connection pooling

    Manages multiple E2 connections to different E2 termination endpoints.
    """

    def __init__(
        self,
        default_config: Optional[E2ClientConfig] = None,
        on_indication: Optional[Callable[[str, BeamIndication], Awaitable[None]]] = None,
    ):
        """
        Initialize E2 client

        Args:
            default_config: Default configuration for connections
            on_indication: Callback for indication messages (receives endpoint_id, indication)
        """
        self.default_config = default_config or E2ClientConfig()
        self._user_on_indication = on_indication

        # Connection pool: endpoint_id -> E2Connection
        self._connections: Dict[str, E2Connection] = {}

        # Combined metrics
        self._global_metrics = E2Metrics()

        # Running state
        self._running = False

    async def add_endpoint(
        self,
        endpoint_id: str,
        host: str,
        port: int,
        config_overrides: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Add a new E2 termination endpoint

        Args:
            endpoint_id: Unique identifier for this endpoint
            host: Hostname or IP address
            port: Port number
            config_overrides: Optional config overrides

        Returns:
            True if connection successful
        """
        if endpoint_id in self._connections:
            logger.warning(f"Endpoint {endpoint_id} already exists")
            return False

        # Create config
        config = E2ClientConfig(
            host=host,
            port=port,
            xapp_id=f"{self.default_config.xapp_id}-{endpoint_id}",
            **{
                k: v for k, v in vars(self.default_config).items()
                if k not in ("host", "port", "xapp_id")
            }
        )

        if config_overrides:
            for key, value in config_overrides.items():
                if hasattr(config, key):
                    setattr(config, key, value)

        # Create connection with wrapped callback
        async def on_indication(indication: BeamIndication):
            if self._user_on_indication:
                await self._user_on_indication(endpoint_id, indication)

        connection = E2Connection(
            config=config,
            on_indication=on_indication,
        )

        # Connect
        success = await connection.connect()
        if success:
            self._connections[endpoint_id] = connection
            logger.info(f"Added endpoint {endpoint_id} ({host}:{port})")
            return True

        return False

    async def remove_endpoint(self, endpoint_id: str):
        """Remove an E2 endpoint"""
        if endpoint_id in self._connections:
            await self._connections[endpoint_id].disconnect()
            del self._connections[endpoint_id]
            logger.info(f"Removed endpoint {endpoint_id}")

    async def connect_all(self) -> int:
        """
        Connect to all configured endpoints

        Returns:
            Number of successful connections
        """
        self._running = True
        successful = 0

        for endpoint_id, conn in self._connections.items():
            if not conn.is_connected:
                if await conn.connect():
                    successful += 1

        return successful

    async def disconnect_all(self):
        """Disconnect from all endpoints"""
        self._running = False

        for endpoint_id, conn in self._connections.items():
            await conn.disconnect()

    async def subscribe_all(
        self,
        cell_ids: Optional[List[str]] = None,
        reporting_period_ms: int = 10
    ) -> Dict[str, SubscriptionResponse]:
        """
        Subscribe to all connected endpoints

        Returns:
            Dict of endpoint_id -> SubscriptionResponse
        """
        results = {}

        for endpoint_id, conn in self._connections.items():
            if conn.is_connected:
                response = await conn.subscribe(cell_ids, reporting_period_ms)
                if response:
                    results[endpoint_id] = response

        return results

    async def send_control(
        self,
        control: BeamControl,
        endpoint_id: Optional[str] = None
    ) -> bool:
        """
        Send beam control message

        Args:
            control: BeamControl message
            endpoint_id: Target endpoint (None = broadcast to all)

        Returns:
            True if sent to at least one endpoint
        """
        if endpoint_id:
            if endpoint_id in self._connections:
                return await self._connections[endpoint_id].send_control(control)
            return False

        # Broadcast to all
        success = False
        for conn in self._connections.values():
            if await conn.send_control(control):
                success = True

        return success

    def get_metrics(self, endpoint_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get metrics for endpoint(s)

        Args:
            endpoint_id: Specific endpoint or None for all

        Returns:
            Metrics dictionary
        """
        if endpoint_id:
            if endpoint_id in self._connections:
                return self._connections[endpoint_id].metrics.to_dict()
            return {}

        # Aggregate all metrics
        aggregate = {
            "endpoints": {},
            "total_messages_sent": 0,
            "total_messages_received": 0,
            "total_indications": 0,
            "total_controls": 0,
            "connected_endpoints": 0,
        }

        for eid, conn in self._connections.items():
            metrics = conn.metrics.to_dict()
            aggregate["endpoints"][eid] = metrics
            aggregate["total_messages_sent"] += metrics["messages_sent"]
            aggregate["total_messages_received"] += metrics["messages_received"]
            aggregate["total_indications"] += metrics["indications_received"]
            aggregate["total_controls"] += metrics["controls_sent"]
            if conn.is_connected:
                aggregate["connected_endpoints"] += 1

        return aggregate

    def get_connection_states(self) -> Dict[str, str]:
        """Get connection state for all endpoints"""
        return {
            eid: conn.state.value
            for eid, conn in self._connections.items()
        }

    @property
    def is_connected(self) -> bool:
        """Check if at least one endpoint is connected"""
        return any(conn.is_connected for conn in self._connections.values())


# =============================================================================
# Async Context Manager and Factory
# =============================================================================

class E2ClientContext:
    """Async context manager for E2Client"""

    def __init__(self, client: E2Client):
        self.client = client

    async def __aenter__(self) -> E2Client:
        await self.client.connect_all()
        return self.client

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.client.disconnect_all()


def create_e2_client(
    endpoints: List[Dict[str, Any]],
    on_indication: Optional[Callable[[str, BeamIndication], Awaitable[None]]] = None,
    **config_kwargs
) -> E2Client:
    """
    Factory function to create E2Client with multiple endpoints

    Args:
        endpoints: List of endpoint configs [{"id": "e2-1", "host": "localhost", "port": 36422}, ...]
        on_indication: Indication callback
        **config_kwargs: Default config overrides

    Returns:
        Configured E2Client (not connected)
    """
    config = E2ClientConfig(**config_kwargs)
    client = E2Client(default_config=config, on_indication=on_indication)

    # Pre-configure endpoints (will connect when start() is called)
    for ep in endpoints:
        endpoint_id = ep.get("id", f"e2-{ep['host']}:{ep['port']}")
        # Store for later connection
        conn_config = E2ClientConfig(
            host=ep["host"],
            port=ep["port"],
            xapp_id=f"{config.xapp_id}-{endpoint_id}",
            **{k: v for k, v in vars(config).items() if k not in ("host", "port", "xapp_id")}
        )

        async def on_ind(indication: BeamIndication, eid=endpoint_id):
            if on_indication:
                await on_indication(eid, indication)

        client._connections[endpoint_id] = E2Connection(
            config=conn_config,
            on_indication=on_ind
        )

    return client


# =============================================================================
# Simple Test Server (for development/testing)
# =============================================================================

async def run_test_server(host: str = "localhost", port: int = 36422):
    """
    Run a simple E2 test server for development

    This simulates an E2 termination endpoint.
    """
    logger.info(f"Starting test E2 server on {host}:{port}")

    async def handle_client(reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        addr = writer.get_extra_info("peername")
        logger.info(f"Client connected: {addr}")

        try:
            while True:
                # Read header
                header_data = await reader.read(E2MessageHeader.HEADER_SIZE)
                if not header_data:
                    break

                header = E2MessageHeader.from_bytes(header_data)
                payload = await reader.read(header.payload_length)

                full_data = header_data + payload
                _, message = E2MessageCodec.decode_message(full_data)

                logger.debug(f"Received: {type(message).__name__}")

                # Respond based on message type
                if isinstance(message, ConnectionRequest):
                    response = ConnectionAck(
                        success=True,
                        e2_term_id="test-e2-term",
                        session_id=f"session-{uuid.uuid4().hex[:8]}"
                    )
                    encoded = E2MessageCodec.encode_message(response, header.sequence_num)
                    writer.write(encoded)

                elif isinstance(message, SubscriptionRequest):
                    response = SubscriptionResponse(
                        subscription_id=message.subscription_id,
                        status=SubscriptionStatus.SUCCESS,
                        e2_node_id="test-e2-node"
                    )
                    encoded = E2MessageCodec.encode_message(response, header.sequence_num)
                    writer.write(encoded)

                elif isinstance(message, HeartbeatRequest):
                    response = HeartbeatResponse(
                        e2_term_id="test-e2-term",
                        timestamp_ms=message.timestamp_ms,
                        sequence_num=message.sequence_num
                    )
                    encoded = E2MessageCodec.encode_message(response, header.sequence_num)
                    writer.write(encoded)

                await writer.drain()

        except Exception as e:
            logger.error(f"Error handling client: {e}")
        finally:
            writer.close()
            await writer.wait_closed()
            logger.info(f"Client disconnected: {addr}")

    server = await asyncio.start_server(handle_client, host, port)

    async with server:
        await server.serve_forever()


# =============================================================================
# Main entry point for testing
# =============================================================================

async def _test_client():
    """Test the E2 client"""
    logging.basicConfig(level=logging.DEBUG)

    # Create client
    async def on_indication(endpoint_id: str, indication: BeamIndication):
        logger.info(f"[{endpoint_id}] Indication: UE={indication.ue_id}, beam={indication.serving_beam_id}")

    client = create_e2_client(
        endpoints=[{"id": "e2-local", "host": "localhost", "port": 36422}],
        on_indication=on_indication
    )

    async with E2ClientContext(client):
        # Subscribe
        await client.subscribe_all(reporting_period_ms=100)

        # Send test control
        from .e2_messages import BeamAction
        control = BeamControl(
            ue_id="test-uav",
            timestamp_ms=time.time() * 1000,
            target_beam_id=42,
            action=BeamAction.SWITCH
        )
        await client.send_control(control)

        # Wait a bit
        await asyncio.sleep(5)

        # Print metrics
        print(json.dumps(client.get_metrics(), indent=2))


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1 and sys.argv[1] == "server":
        asyncio.run(run_test_server())
    else:
        asyncio.run(_test_client())
