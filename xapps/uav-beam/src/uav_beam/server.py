"""
UAV Beam Tracking xApp REST API Server

Provides dual-path E2 interface for:
- REST API: HTTP endpoints for manual testing and external integration
- Socket: TCP socket connection to E2 termination for ns-3 integration

Features:
- E2SM-KPM equivalent beam measurement reception
- E2SM-RC equivalent beam control command sending
- xApp coordination via SDL
- Unified message processing pipeline
"""

import logging
import asyncio
import threading
import time
from typing import Dict, Any, Optional, Callable, List
from dataclasses import asdict
import json
from queue import Queue

from flask import Flask, request, jsonify
import numpy as np

from .beam_tracker import BeamTracker, BeamConfig, BeamMeasurement, BeamDecision
from .trajectory_predictor import TrajectoryPredictor, PredictorConfig
from .angle_estimator import AngleEstimator, AngleEstimatorConfig, EstimationMethod
from .e2_messages import (
    BeamIndication,
    BeamControl,
    BeamAction,
    indication_to_dict,
    dict_to_indication,
)
from .e2_client import (
    E2Client,
    E2ClientConfig,
    E2Connection,
    create_e2_client,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class UAVBeamXApp:
    """
    UAV Beam Tracking xApp

    Integrates:
    - BeamTracker: Beam management decisions
    - TrajectoryPredictor: UAV position prediction
    - AngleEstimator: AoA/AoD calculation
    - E2Client: Socket communication with E2 termination
    """

    def __init__(
        self,
        beam_config: Optional[BeamConfig] = None,
        predictor_config: Optional[PredictorConfig] = None,
        estimator_config: Optional[AngleEstimatorConfig] = None,
        e2_config: Optional[E2ClientConfig] = None,
        on_decision: Optional[Callable[[BeamDecision], None]] = None
    ):
        self.beam_tracker = BeamTracker(beam_config)
        self.trajectory_predictor = TrajectoryPredictor(predictor_config)
        self.angle_estimator = AngleEstimator(estimator_config)
        self.e2_config = e2_config or E2ClientConfig()
        self.on_decision = on_decision

        # E2 client for socket communication
        self._e2_client: Optional[E2Client] = None
        self._e2_loop: Optional[asyncio.AbstractEventLoop] = None
        self._e2_thread: Optional[threading.Thread] = None
        self._e2_running = False

        # Control message queue (for async sending)
        self._control_queue: Queue = Queue(maxsize=1000)

        # Decision history
        self.decision_history: Dict[str, list] = {}

        # Statistics
        self.stats = {
            "indications_received": 0,
            "decisions_made": 0,
            "beam_switches": 0,
            "controls_sent": 0,
            "start_time": time.time(),
            "rest_requests": 0,
            "socket_messages": 0,
        }

        logger.info("UAVBeamXApp initialized")

    def process_e2_indication(self, indication: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process E2 indication message

        Expected format:
        {
            "ue_id": "uav-001",
            "timestamp_ms": 1234567890,
            "serving_cell_id": "cell-1",
            "serving_beam_id": 42,
            "beam_rsrp_dbm": -85.0,
            "neighbor_beams": {
                "43": -88.0,
                "41": -87.5
            },
            "position": [100.0, 200.0, 50.0],  # optional
            "velocity": [5.0, 2.0, 0.5]        # optional
        }

        Returns:
            Beam control decision
        """
        self.stats["indications_received"] += 1

        try:
            # Extract fields
            ue_id = indication["ue_id"]
            timestamp_ms = indication.get("timestamp_ms", time.time() * 1000)

            # Create BeamMeasurement
            measurement = BeamMeasurement(
                timestamp_ms=timestamp_ms,
                ue_id=ue_id,
                serving_beam_id=indication["serving_beam_id"],
                serving_rsrp_dbm=indication["beam_rsrp_dbm"],
                neighbor_beams={
                    int(k): float(v)
                    for k, v in indication.get("neighbor_beams", {}).items()
                },
                cqi=indication.get("cqi", 0),
                position=tuple(indication["position"]) if "position" in indication else None,
                velocity=tuple(indication["velocity"]) if "velocity" in indication else None,
            )

            # Update trajectory predictor if position available
            if measurement.position:
                self.trajectory_predictor.update(
                    uav_id=ue_id,
                    position=np.array(measurement.position),
                    timestamp_ms=timestamp_ms
                )

                # Get trajectory prediction
                predicted_state = self.trajectory_predictor.predict(
                    uav_id=ue_id,
                    horizon_ms=self.beam_tracker.config.prediction_horizon_ms,
                    method="hybrid"
                )

                if predicted_state:
                    # Update measurement with predicted info
                    measurement.position = tuple(predicted_state.position)
                    measurement.velocity = tuple(predicted_state.velocity)

            # Get beam decision
            decision = self.beam_tracker.process_measurement(measurement)

            self.stats["decisions_made"] += 1
            if decision.action == "switch":
                self.stats["beam_switches"] += 1

            # Store decision history
            if ue_id not in self.decision_history:
                self.decision_history[ue_id] = []
            self.decision_history[ue_id].append(asdict(decision))

            # Keep limited history
            if len(self.decision_history[ue_id]) > 100:
                self.decision_history[ue_id] = self.decision_history[ue_id][-100:]

            # Callback for external handling
            if self.on_decision:
                try:
                    self.on_decision(decision)
                except Exception as e:
                    logger.error(f"Error in decision callback: {e}")

            # Build response
            response = {
                "status": "success",
                "decision": {
                    "ue_id": decision.ue_id,
                    "action": decision.action,
                    "current_beam_id": decision.current_beam_id,
                    "target_beam_id": decision.target_beam_id,
                    "confidence": decision.confidence,
                    "predicted_rsrp_dbm": decision.predicted_rsrp_dbm,
                    "reason": decision.reason,
                },
            }

            logger.info(
                f"UE {ue_id}: {decision.action} beam {decision.current_beam_id} -> "
                f"{decision.target_beam_id} (confidence={decision.confidence:.2f})"
            )

            return response

        except Exception as e:
            logger.error(f"Error processing indication: {e}")
            return {
                "status": "error",
                "error": str(e),
            }

    def process_beam_indication(self, indication: BeamIndication) -> Optional[BeamControl]:
        """
        Process BeamIndication object and return BeamControl

        This is the internal processing method for socket messages.

        Args:
            indication: BeamIndication from E2 socket

        Returns:
            BeamControl decision or None
        """
        # Convert to dict format
        indication_dict = indication_to_dict(indication)

        # Process through unified pipeline
        result = self.process_e2_indication(indication_dict)

        if result["status"] != "success":
            return None

        decision = result["decision"]

        # Only return control if action requires it
        if decision["action"] in ("switch", "recover", "refine"):
            action_map = {
                "switch": BeamAction.SWITCH,
                "recover": BeamAction.RECOVER,
                "refine": BeamAction.REFINE,
                "maintain": BeamAction.MAINTAIN,
            }

            return BeamControl(
                ue_id=decision["ue_id"],
                timestamp_ms=time.time() * 1000,
                target_beam_id=decision["target_beam_id"],
                action=action_map.get(decision["action"], BeamAction.MAINTAIN),
                cell_id=indication.serving_cell_id,
                confidence=decision["confidence"],
                predicted_rsrp_dbm=decision["predicted_rsrp_dbm"],
                reason=decision["reason"],
            )

        return None

    def estimate_angle(self, request_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Estimate AoA/AoD from received signal

        Expected format:
        {
            "ue_id": "uav-001",
            "timestamp_ms": 1234567890,
            "received_signal_real": [[...], [...]],  # num_elements x num_snapshots
            "received_signal_imag": [[...], [...]]
        }
        """
        try:
            # Reconstruct complex signal
            signal_real = np.array(request_data["received_signal_real"])
            signal_imag = np.array(request_data["received_signal_imag"])
            received_signal = signal_real + 1j * signal_imag

            timestamp_ms = request_data.get("timestamp_ms", time.time() * 1000)

            # Estimate angles
            estimate = self.angle_estimator.estimate(
                received_signal=received_signal,
                timestamp_ms=timestamp_ms,
                method=EstimationMethod.MUSIC
            )

            return {
                "status": "success",
                "estimate": {
                    "azimuth_deg": np.degrees(estimate.azimuth_rad),
                    "elevation_deg": np.degrees(estimate.elevation_rad),
                    "confidence": estimate.confidence,
                    "method": estimate.method,
                }
            }

        except Exception as e:
            logger.error(f"Error estimating angle: {e}")
            return {
                "status": "error",
                "error": str(e),
            }

    def get_statistics(self) -> Dict[str, Any]:
        """Get xApp statistics"""
        uptime = time.time() - self.stats["start_time"]

        stats = {
            **self.stats,
            "uptime_seconds": uptime,
            "indications_per_second": (
                self.stats["indications_received"] / uptime if uptime > 0 else 0
            ),
            "beam_tracker_stats": self.beam_tracker.get_statistics(),
            "angle_estimator_stats": self.angle_estimator.get_statistics(),
            "tracked_uavs": self.trajectory_predictor.get_tracked_uavs(),
        }

        # Add E2 socket metrics if available
        if self._e2_client:
            stats["e2_socket_metrics"] = self._e2_client.get_metrics()
            stats["e2_connection_states"] = self._e2_client.get_connection_states()

        return stats

    def get_ue_state(self, ue_id: str) -> Optional[Dict[str, Any]]:
        """Get state for specific UE"""
        if ue_id not in self.beam_tracker.ue_states:
            return None

        state = self.beam_tracker.ue_states[ue_id]

        # Get trajectory prediction
        predicted = self.trajectory_predictor.predict(ue_id, horizon_ms=100.0)

        return {
            "ue_id": ue_id,
            "beam_state": state["state"].value,
            "last_beam": state["last_beam"],
            "failure_count": state["failure_count"],
            "predicted_position": predicted.position.tolist() if predicted else None,
            "predicted_velocity": predicted.velocity.tolist() if predicted else None,
            "decision_history": self.decision_history.get(ue_id, [])[-10:],
        }

    # =========================================================================
    # E2 Socket Communication
    # =========================================================================

    def start_e2_socket(
        self,
        endpoints: List[Dict[str, Any]],
        auto_subscribe: bool = True,
        reporting_period_ms: int = 10
    ):
        """
        Start E2 socket communication in background thread

        Args:
            endpoints: List of E2 endpoints [{"id": "e2-1", "host": "localhost", "port": 36422}]
            auto_subscribe: Automatically subscribe after connecting
            reporting_period_ms: Reporting period for subscriptions
        """
        if self._e2_running:
            logger.warning("E2 socket already running")
            return

        logger.info(f"Starting E2 socket with {len(endpoints)} endpoint(s)")

        # Create new event loop for E2 thread
        self._e2_loop = asyncio.new_event_loop()

        # Create E2 client
        async def on_indication(endpoint_id: str, indication: BeamIndication):
            self.stats["socket_messages"] += 1
            control = self.process_beam_indication(indication)

            if control:
                # Send control back
                await self._e2_client.send_control(control, endpoint_id)
                self.stats["controls_sent"] += 1

        self._e2_client = create_e2_client(
            endpoints=endpoints,
            on_indication=on_indication,
            auto_reconnect=True,
        )

        # Start background thread
        def run_e2_loop():
            asyncio.set_event_loop(self._e2_loop)

            async def main():
                # Connect to all endpoints
                connected = await self._e2_client.connect_all()
                logger.info(f"Connected to {connected} E2 endpoint(s)")

                if auto_subscribe and connected > 0:
                    subs = await self._e2_client.subscribe_all(
                        reporting_period_ms=reporting_period_ms
                    )
                    logger.info(f"Created {len(subs)} subscription(s)")

                # Run until stopped
                while self._e2_running:
                    # Process control queue
                    while not self._control_queue.empty():
                        try:
                            control = self._control_queue.get_nowait()
                            await self._e2_client.send_control(control)
                        except Exception as e:
                            logger.error(f"Error sending queued control: {e}")

                    await asyncio.sleep(0.01)

                # Cleanup
                await self._e2_client.disconnect_all()

            self._e2_loop.run_until_complete(main())

        self._e2_running = True
        self._e2_thread = threading.Thread(target=run_e2_loop, daemon=True)
        self._e2_thread.start()

        logger.info("E2 socket background thread started")

    def stop_e2_socket(self):
        """Stop E2 socket communication"""
        if not self._e2_running:
            return

        logger.info("Stopping E2 socket")
        self._e2_running = False

        # Wait for thread to finish
        if self._e2_thread:
            self._e2_thread.join(timeout=5.0)
            self._e2_thread = None

        if self._e2_loop:
            self._e2_loop.close()
            self._e2_loop = None

        self._e2_client = None
        logger.info("E2 socket stopped")

    def send_control_async(self, control: BeamControl):
        """
        Queue a control message for async sending via socket

        Args:
            control: BeamControl message to send
        """
        if not self._e2_running:
            logger.warning("E2 socket not running, cannot send control")
            return

        try:
            self._control_queue.put_nowait(control)
        except Exception as e:
            logger.error(f"Failed to queue control: {e}")

    def get_e2_status(self) -> Dict[str, Any]:
        """Get E2 socket connection status"""
        if not self._e2_client:
            return {
                "running": False,
                "connected": False,
                "endpoints": {}
            }

        return {
            "running": self._e2_running,
            "connected": self._e2_client.is_connected,
            "connection_states": self._e2_client.get_connection_states(),
            "metrics": self._e2_client.get_metrics(),
        }


# Flask application
app = Flask(__name__)
xapp: Optional[UAVBeamXApp] = None


def get_xapp() -> UAVBeamXApp:
    """Get or create xApp instance"""
    global xapp
    if xapp is None:
        xapp = UAVBeamXApp()
    return xapp


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    xapp_instance = get_xapp()
    e2_status = xapp_instance.get_e2_status()

    return jsonify({
        "status": "healthy",
        "xapp": "uav-beam",
        "version": "0.1.0",
        "e2_socket_connected": e2_status.get("connected", False),
    })


@app.route('/e2/indication', methods=['POST'])
def e2_indication():
    """
    E2 indication endpoint

    Receives beam measurements from E2 nodes and returns beam decisions.
    """
    if not request.is_json:
        return jsonify({"status": "error", "error": "JSON required"}), 400

    get_xapp().stats["rest_requests"] += 1
    indication = request.get_json()
    result = get_xapp().process_e2_indication(indication)

    status_code = 200 if result["status"] == "success" else 500
    return jsonify(result), status_code


@app.route('/e2/control', methods=['POST'])
def e2_control():
    """
    Send E2 control message via socket

    Request format:
    {
        "ue_id": "uav-001",
        "target_beam_id": 42,
        "action": "switch",  # switch, maintain, recover, refine
        "cell_id": "cell-1",
        "reason": "optional reason"
    }
    """
    if not request.is_json:
        return jsonify({"status": "error", "error": "JSON required"}), 400

    data = request.get_json()

    action_map = {
        "switch": BeamAction.SWITCH,
        "maintain": BeamAction.MAINTAIN,
        "recover": BeamAction.RECOVER,
        "refine": BeamAction.REFINE,
        "sweep": BeamAction.SWEEP,
    }

    try:
        control = BeamControl(
            ue_id=data["ue_id"],
            timestamp_ms=time.time() * 1000,
            target_beam_id=data["target_beam_id"],
            action=action_map.get(data.get("action", "switch"), BeamAction.SWITCH),
            cell_id=data.get("cell_id", ""),
            reason=data.get("reason", ""),
        )

        get_xapp().send_control_async(control)

        return jsonify({
            "status": "success",
            "message": "Control message queued",
            "control": {
                "ue_id": control.ue_id,
                "target_beam_id": control.target_beam_id,
                "action": control.action.name,
            }
        })

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 400


@app.route('/e2/socket/start', methods=['POST'])
def e2_socket_start():
    """
    Start E2 socket connection

    Request format:
    {
        "endpoints": [
            {"id": "e2-1", "host": "localhost", "port": 36422}
        ],
        "auto_subscribe": true,
        "reporting_period_ms": 10
    }
    """
    if not request.is_json:
        return jsonify({"status": "error", "error": "JSON required"}), 400

    data = request.get_json()
    endpoints = data.get("endpoints", [])

    if not endpoints:
        return jsonify({"status": "error", "error": "No endpoints provided"}), 400

    try:
        get_xapp().start_e2_socket(
            endpoints=endpoints,
            auto_subscribe=data.get("auto_subscribe", True),
            reporting_period_ms=data.get("reporting_period_ms", 10)
        )

        return jsonify({
            "status": "success",
            "message": f"E2 socket started with {len(endpoints)} endpoint(s)"
        })

    except Exception as e:
        return jsonify({"status": "error", "error": str(e)}), 500


@app.route('/e2/socket/stop', methods=['POST'])
def e2_socket_stop():
    """Stop E2 socket connection"""
    get_xapp().stop_e2_socket()
    return jsonify({"status": "success", "message": "E2 socket stopped"})


@app.route('/e2/socket/status', methods=['GET'])
def e2_socket_status():
    """Get E2 socket connection status"""
    return jsonify(get_xapp().get_e2_status())


@app.route('/angle/estimate', methods=['POST'])
def angle_estimate():
    """
    Angle estimation endpoint

    Estimates AoA/AoD from received signal samples.
    """
    if not request.is_json:
        return jsonify({"status": "error", "error": "JSON required"}), 400

    data = request.get_json()
    result = get_xapp().estimate_angle(data)

    status_code = 200 if result["status"] == "success" else 500
    return jsonify(result), status_code


@app.route('/statistics', methods=['GET'])
def statistics():
    """Get xApp statistics"""
    return jsonify(get_xapp().get_statistics())


@app.route('/ue/<ue_id>', methods=['GET'])
def ue_state(ue_id: str):
    """Get state for specific UE"""
    state = get_xapp().get_ue_state(ue_id)

    if state is None:
        return jsonify({"status": "error", "error": f"UE {ue_id} not found"}), 404

    return jsonify(state)


@app.route('/config', methods=['GET', 'PUT'])
def config():
    """Get or update xApp configuration"""
    xapp_instance = get_xapp()

    if request.method == 'GET':
        return jsonify({
            "beam_config": {
                "num_beams_h": xapp_instance.beam_tracker.config.num_beams_h,
                "num_beams_v": xapp_instance.beam_tracker.config.num_beams_v,
                "total_beams": xapp_instance.beam_tracker.config.total_beams,
                "beam_failure_threshold_db": xapp_instance.beam_tracker.config.beam_failure_threshold_db,
                "prediction_horizon_ms": xapp_instance.beam_tracker.config.prediction_horizon_ms,
            },
            "predictor_config": {
                "max_prediction_horizon_ms": xapp_instance.trajectory_predictor.config.max_prediction_horizon_ms,
                "max_velocity": xapp_instance.trajectory_predictor.config.max_velocity,
            },
            "estimator_config": {
                "num_elements_h": xapp_instance.angle_estimator.config.num_elements_h,
                "num_elements_v": xapp_instance.angle_estimator.config.num_elements_v,
            },
            "e2_socket": xapp_instance.get_e2_status(),
        })

    elif request.method == 'PUT':
        # Limited runtime configuration updates
        if not request.is_json:
            return jsonify({"status": "error", "error": "JSON required"}), 400

        updates = request.get_json()

        if "beam_failure_threshold_db" in updates:
            xapp_instance.beam_tracker.config.beam_failure_threshold_db = updates["beam_failure_threshold_db"]

        if "prediction_horizon_ms" in updates:
            xapp_instance.beam_tracker.config.prediction_horizon_ms = updates["prediction_horizon_ms"]

        return jsonify({"status": "success", "message": "Configuration updated"})


@app.route('/reset', methods=['POST'])
def reset():
    """Reset xApp state"""
    xapp_instance = get_xapp()
    xapp_instance.beam_tracker.reset()
    xapp_instance.decision_history.clear()
    xapp_instance.stats = {
        "indications_received": 0,
        "decisions_made": 0,
        "beam_switches": 0,
        "controls_sent": 0,
        "start_time": time.time(),
        "rest_requests": 0,
        "socket_messages": 0,
    }
    return jsonify({"status": "success", "message": "xApp state reset"})


def create_app(config: Optional[Dict] = None) -> Flask:
    """Create Flask application with optional configuration"""
    global xapp

    beam_config = BeamConfig(**(config.get("beam", {}) if config else {}))
    predictor_config = PredictorConfig(**(config.get("predictor", {}) if config else {}))
    estimator_config = AngleEstimatorConfig(**(config.get("estimator", {}) if config else {}))

    # E2 config
    e2_config = None
    if config and "e2" in config:
        e2_config = E2ClientConfig(**config["e2"])

    xapp = UAVBeamXApp(
        beam_config=beam_config,
        predictor_config=predictor_config,
        estimator_config=estimator_config,
        e2_config=e2_config
    )

    # Auto-start E2 socket if configured
    if config and "e2_endpoints" in config:
        xapp.start_e2_socket(
            endpoints=config["e2_endpoints"],
            auto_subscribe=config.get("e2_auto_subscribe", True),
            reporting_period_ms=config.get("e2_reporting_period_ms", 10)
        )

    return app


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="UAV Beam Tracking xApp")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=5001, help="Port to bind to")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")

    # E2 socket options
    parser.add_argument("--e2-host", default=None, help="E2 termination host")
    parser.add_argument("--e2-port", type=int, default=36422, help="E2 termination port")
    parser.add_argument("--e2-auto-connect", action="store_true", help="Auto-connect to E2 termination")

    args = parser.parse_args()

    config = {}

    # Configure E2 if specified
    if args.e2_host and args.e2_auto_connect:
        config["e2_endpoints"] = [
            {"id": "e2-default", "host": args.e2_host, "port": args.e2_port}
        ]

    app = create_app(config)
    app.run(host=args.host, port=args.port, debug=args.debug)
