"""HTTP server for receiving E2 indications and returning resource decisions.

This module provides a Flask-based HTTP API that:
1. Receives E2 indications containing UAV position and radio measurements
2. Parses indication data into UavState and RadioSnapshot objects
3. Applies path-aware RC policy to generate ResourceDecisions
4. Returns decisions and maintains decision history
"""

import json
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, request, jsonify

from uav_policy.policy_engine import (
    FlightPlanPolicy,
    PathSegmentPlan,
    RadioSnapshot,
    ResourceDecision,
    ServiceProfile,
    UavState,
    path_aware_rc_policy,
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class PolicyEngineHandler:
    """Handles parsing of E2 indications and calling policy engine."""

    def __init__(self, max_history: int = 1000):
        """Initialize handler with decision history tracking.

        Args:
            max_history: Maximum number of decisions to keep in history
        """
        self.max_history = max_history
        self.decision_history: List[Dict[str, Any]] = []

    def parse_indication(
        self, indication_json: Dict[str, Any]
    ) -> Tuple[UavState, RadioSnapshot]:
        """Parse HTTP E2 indication into UavState and RadioSnapshot.

        Args:
            indication_json: E2 indication data as dict

        Returns:
            Tuple of (UavState, RadioSnapshot)

        Raises:
            ValueError: If required fields are missing
        """
        # Parse position
        try:
            position = indication_json["position"]
            x = float(position["x"])
            y = float(position["y"])
            z = float(position["z"])
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"Invalid position data: {e}")

        # Parse UAV state
        uav_id = indication_json.get("uav_id", "unknown")
        path_position = indication_json.get("path_position")
        slice_id = indication_json.get("slice_id")

        if path_position is not None:
            try:
                path_position = float(path_position)
            except (TypeError, ValueError) as e:
                logger.warning(f"Invalid path_position for {uav_id}: {e}")
                path_position = None

        uav_state = UavState(
            uav_id=uav_id,
            x=x,
            y=y,
            z=z,
            slice_id=slice_id,
            path_position=path_position,
        )

        # Parse radio snapshot
        try:
            radio_data = indication_json["radio_snapshot"]
            radio_snapshot = RadioSnapshot(
                serving_cell_id=radio_data["serving_cell_id"],
                neighbor_cell_ids=radio_data.get("neighbor_cell_ids", []),
                rsrp_serving=float(radio_data["rsrp_serving"]),
                rsrp_best_neighbor=float(radio_data["rsrp_best_neighbor"]),
                prb_utilization_serving=float(radio_data["prb_utilization_serving"]),
                prb_utilization_slice=radio_data.get("prb_utilization_slice"),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"Invalid radio snapshot data: {e}")

        return uav_state, radio_snapshot

    def parse_flight_plan(self, plan_json: Dict[str, Any]) -> FlightPlanPolicy:
        """Parse flight plan from E2 indication.

        Args:
            plan_json: Flight plan data as dict

        Returns:
            FlightPlanPolicy object

        Raises:
            ValueError: If required fields are missing
        """
        try:
            segments = []
            for seg_data in plan_json.get("segments", []):
                segment = PathSegmentPlan(
                    start_pos=float(seg_data["start_pos"]),
                    end_pos=float(seg_data["end_pos"]),
                    planned_cell_id=seg_data["planned_cell_id"],
                    slice_id=seg_data["slice_id"],
                    base_prb_quota=int(seg_data["base_prb_quota"]),
                )
                segments.append(segment)

            return FlightPlanPolicy(
                uav_id=plan_json.get("uav_id", "unknown"),
                segments=segments,
            )
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"Invalid flight plan data: {e}")

    def parse_service_profile(self, profile_json: Dict[str, Any]) -> ServiceProfile:
        """Parse service profile from E2 indication.

        Args:
            profile_json: Service profile data as dict

        Returns:
            ServiceProfile object

        Raises:
            ValueError: If required fields are missing
        """
        try:
            return ServiceProfile(
                name=profile_json["name"],
                target_bitrate_mbps=float(profile_json["target_bitrate_mbps"]),
                min_sinr_db=float(profile_json.get("min_sinr_db", 0.0)),
            )
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"Invalid service profile data: {e}")

    def handle_indication(self, indication_json: Dict[str, Any]) -> ResourceDecision:
        """Process E2 indication and return resource decision.

        Args:
            indication_json: Complete E2 indication data

        Returns:
            ResourceDecision

        Raises:
            ValueError: If indication is malformed
        """
        logger.info(f"Processing indication for UAV: {indication_json.get('uav_id')}")

        # Parse basic UAV state and radio snapshot
        uav_state, radio_snapshot = self.parse_indication(indication_json)

        # Optionally parse flight plan
        flight_plan: Optional[FlightPlanPolicy] = None
        if "flight_plan" in indication_json:
            try:
                flight_plan = self.parse_flight_plan(indication_json["flight_plan"])
                logger.debug(f"Parsed flight plan for {uav_state.uav_id}")
            except ValueError as e:
                logger.warning(f"Failed to parse flight plan: {e}")

        # Optionally parse service profile
        service_profile: Optional[ServiceProfile] = None
        if "service_profile" in indication_json:
            try:
                service_profile = self.parse_service_profile(
                    indication_json["service_profile"]
                )
                logger.debug(f"Parsed service profile: {service_profile.name}")
            except ValueError as e:
                logger.warning(f"Failed to parse service profile: {e}")

        # Apply policy engine
        decision = path_aware_rc_policy(
            uav=uav_state,
            radio=radio_snapshot,
            plan=flight_plan,
            service=service_profile,
        )

        logger.info(
            f"Decision for {decision.uav_id}: cell={decision.target_cell_id}, "
            f"prb={decision.prb_quota}, reason={decision.reason[:50]}..."
        )

        return decision

    def record_decision(self, decision: ResourceDecision) -> None:
        """Record decision in history.

        Args:
            decision: ResourceDecision to record
        """
        record = {
            "timestamp": datetime.utcnow().isoformat(),
            "uav_id": decision.uav_id,
            "target_cell_id": decision.target_cell_id,
            "slice_id": decision.slice_id,
            "prb_quota": decision.prb_quota,
            "reason": decision.reason,
        }
        self.decision_history.append(record)

        # Maintain max history size
        if len(self.decision_history) > self.max_history:
            self.decision_history = self.decision_history[-self.max_history :]

    def get_recent_decisions(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent decisions from history.

        Args:
            limit: Maximum number of decisions to return

        Returns:
            List of decision records (most recent first)
        """
        return self.decision_history[-limit:][::-1]


def create_app() -> Flask:
    """Create and configure the Flask application.

    Returns:
        Configured Flask app instance
    """
    app = Flask(__name__)
    app.config["JSON_SORT_KEYS"] = False

    # Initialize handler
    handler = PolicyEngineHandler()

    # Routes
    @app.route("/health", methods=["GET"])
    def health_check():
        """Health check endpoint."""
        return jsonify(
            {
                "status": "healthy",
                "timestamp": datetime.utcnow().isoformat(),
                "service": "uav-policy-xapp",
            }
        ), 200

    @app.route("/api/v1/e2/indication", methods=["POST"])
    def handle_simulation_indication():
        """Receive E2 indication from simulation bridge and return control decision.

        This endpoint is designed for the ns-3/Python UAV simulator integration.

        Expected JSON format (from E2 HTTP Bridge):
        {
            "indication_type": "KPM",
            "ue_id": "UAV-001",
            "gnb_id": "gNB-001",
            "cell_id": 1,
            "timestamp": 10.5,
            "measurements": {
                "rsrp_serving_dbm": -85.0,
                "rsrq_serving_db": -12.0,
                "sinr_db": 15.0,
                "prb_utilization": 0.6
            },
            "neighbor_cells": [{"cell_id": 2, "rsrp": -90.0}],
            "ue_context": {
                "position": {"x": 100, "y": 200, "z": 100}
            }
        }

        Returns control decision:
        {
            "action": "handover" | "prb_allocation" | "no_action",
            "target_cell_id": 2,
            "allocated_prbs": 50,
            "reason": "..."
        }
        """
        try:
            if not request.is_json:
                return jsonify({"error": "Content-Type must be application/json"}), 400

            try:
                data = request.get_json(force=True)
            except Exception as e:
                logger.warning(f"Failed to parse JSON: {e}")
                return jsonify({"error": "Invalid JSON format"}), 400

            if data is None:
                return jsonify({"error": "Invalid JSON"}), 400

            # Extract fields from simulation format
            ue_id = data.get("ue_id", "UAV-001")
            cell_id = data.get("cell_id", 1)
            measurements = data.get("measurements", {})
            neighbor_cells = data.get("neighbor_cells", [])
            ue_context = data.get("ue_context", {})
            position = ue_context.get("position", {})

            # Convert to internal format
            rsrp_serving = measurements.get("rsrp_serving_dbm", -100.0)
            rsrq_serving = measurements.get("rsrq_serving_db", -15.0)
            prb_util = measurements.get("prb_utilization", 0.5)

            # Find best neighbor
            best_neighbor_rsrp = -140.0
            best_neighbor_id = None
            neighbor_ids = []
            for nc in neighbor_cells:
                nc_id = str(nc.get("cell_id", 0))
                nc_rsrp = nc.get("rsrp", -140.0)
                neighbor_ids.append(nc_id)
                if nc_rsrp > best_neighbor_rsrp:
                    best_neighbor_rsrp = nc_rsrp
                    best_neighbor_id = nc_id

            # Build internal indication format
            internal_indication = {
                "uav_id": ue_id,
                "position": {
                    "x": position.get("x", 0.0),
                    "y": position.get("y", 0.0),
                    "z": position.get("z", 100.0)
                },
                "radio_snapshot": {
                    "serving_cell_id": str(cell_id),
                    "neighbor_cell_ids": neighbor_ids,
                    "rsrp_serving": rsrp_serving,
                    "rsrp_best_neighbor": best_neighbor_rsrp,
                    "prb_utilization_serving": prb_util
                }
            }

            logger.info(f"Simulation indication: UE={ue_id}, cell={cell_id}, RSRP={rsrp_serving:.1f} dBm")

            # Process through policy engine
            try:
                decision = handler.handle_indication(internal_indication)
                handler.record_decision(decision)

                # Determine action type
                current_cell = str(cell_id)
                if decision.target_cell_id != current_cell:
                    action = "handover"
                elif decision.prb_quota and decision.prb_quota > 0:
                    action = "prb_allocation"
                else:
                    action = "no_action"

                response = {
                    "action": action,
                    "target_cell_id": int(decision.target_cell_id) if decision.target_cell_id.isdigit() else decision.target_cell_id,
                    "allocated_prbs": decision.prb_quota,
                    "reason": decision.reason,
                    "timestamp": datetime.utcnow().isoformat()
                }

                logger.info(f"Decision: action={action}, target_cell={decision.target_cell_id}, prb={decision.prb_quota}")
                return jsonify(response), 200

            except ValueError as e:
                logger.error(f"Invalid indication data: {e}")
                return jsonify({"error": f"Invalid indication data: {str(e)}"}), 400

        except Exception as e:
            logger.error(f"Unexpected error: {e}", exc_info=True)
            return jsonify({"error": f"Internal server error: {str(e)}"}), 500

    @app.route("/e2/indication", methods=["POST"])
    def handle_e2_indication():
        """Receive E2 indication and return resource decision.

        Expected JSON format:
        {
            "uav_id": "uav-001",
            "position": {"x": float, "y": float, "z": float},
            "path_position": float (optional),
            "slice_id": string (optional),
            "radio_snapshot": {
                "serving_cell_id": string,
                "neighbor_cell_ids": [string, ...],
                "rsrp_serving": float,
                "rsrp_best_neighbor": float,
                "prb_utilization_serving": float,
                "prb_utilization_slice": float (optional)
            },
            "flight_plan": {...} (optional),
            "service_profile": {...} (optional)
        }
        """
        try:
            # Parse JSON
            if not request.is_json:
                return jsonify({"error": "Content-Type must be application/json"}), 400

            try:
                data = request.get_json(force=True)
            except Exception as e:
                logger.warning(f"Failed to parse JSON: {e}")
                return jsonify({"error": "Invalid JSON format"}), 400

            if data is None:
                return jsonify({"error": "Invalid JSON"}), 400

            # Process indication
            try:
                decision = handler.handle_indication(data)
                handler.record_decision(decision)

                return jsonify(
                    {
                        "uav_id": decision.uav_id,
                        "target_cell_id": decision.target_cell_id,
                        "slice_id": decision.slice_id,
                        "prb_quota": decision.prb_quota,
                        "reason": decision.reason,
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                ), 200

            except ValueError as e:
                logger.error(f"Invalid indication data: {e}")
                return jsonify({"error": f"Invalid indication data: {str(e)}"}), 400

        except Exception as e:
            logger.error(f"Unexpected error processing indication: {e}", exc_info=True)
            return (
                jsonify({"error": f"Internal server error: {str(e)}"}),
                500,
            )

    @app.route("/decisions", methods=["GET"])
    def get_decisions():
        """Get recent decisions.

        Query parameters:
            limit: Maximum number of decisions to return (default 100)
        """
        try:
            limit = request.args.get("limit", 100, type=int)
            limit = max(1, min(limit, 1000))  # Clamp to [1, 1000]

            decisions = handler.get_recent_decisions(limit)
            return (
                jsonify(
                    {
                        "decisions": decisions,
                        "count": len(decisions),
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                ),
                200,
            )

        except Exception as e:
            logger.error(f"Error retrieving decisions: {e}", exc_info=True)
            return jsonify({"error": f"Failed to retrieve decisions: {str(e)}"}), 500

    @app.route("/stats", methods=["GET"])
    def get_stats():
        """Get server statistics."""
        try:
            history = handler.decision_history
            uav_ids = set(d["uav_id"] for d in history)

            return (
                jsonify(
                    {
                        "total_decisions": len(history),
                        "unique_uavs": len(uav_ids),
                        "uav_list": sorted(list(uav_ids)),
                        "timestamp": datetime.utcnow().isoformat(),
                    }
                ),
                200,
            )

        except Exception as e:
            logger.error(f"Error retrieving stats: {e}", exc_info=True)
            return jsonify({"error": f"Failed to retrieve stats: {str(e)}"}), 500

    @app.errorhandler(404)
    def not_found(e):
        """Handle 404 errors."""
        return jsonify({"error": "Endpoint not found"}), 404

    @app.errorhandler(405)
    def method_not_allowed(e):
        """Handle 405 errors."""
        return jsonify({"error": "Method not allowed"}), 405

    return app


if __name__ == "__main__":
    app = create_app()
    app.run(host="0.0.0.0", port=5000, debug=False)
