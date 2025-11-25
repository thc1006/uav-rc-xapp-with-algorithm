"""
UAV Beam Tracking xApp - Main Entry Point
"""

import logging
import argparse
from .server import create_app

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def main():
    """Main entry point for UAV Beam xApp"""
    parser = argparse.ArgumentParser(
        description="UAV Beam Tracking xApp for O-RAN Near-RT RIC"
    )
    parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5001,
        help="Port to bind to (default: 5001)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    parser.add_argument(
        "--num-beams-h",
        type=int,
        default=16,
        help="Number of horizontal beams (default: 16)"
    )
    parser.add_argument(
        "--num-beams-v",
        type=int,
        default=8,
        help="Number of vertical beams (default: 8)"
    )

    args = parser.parse_args()

    config = {
        "beam": {
            "num_beams_h": args.num_beams_h,
            "num_beams_v": args.num_beams_v,
        }
    }

    logger.info(f"Starting UAV Beam Tracking xApp on {args.host}:{args.port}")
    logger.info(f"Beam configuration: {args.num_beams_h}x{args.num_beams_v} beams")

    app = create_app(config)
    app.run(host=args.host, port=args.port, debug=args.debug)


if __name__ == "__main__":
    main()
