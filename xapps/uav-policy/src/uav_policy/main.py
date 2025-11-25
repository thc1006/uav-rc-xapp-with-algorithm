"""Main entry point for UAV policy xApp.

This module starts the HTTP server to receive E2 indications
and generate resource allocation decisions.
"""

import logging
import os
from uav_policy.server import create_app


# Configure logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    """Start the policy engine HTTP server."""
    logger.info("Starting UAV policy xApp server...")

    app = create_app()

    # Get configuration from environment
    host = os.environ.get("SERVER_HOST", "0.0.0.0")
    port = int(os.environ.get("SERVER_PORT", 5000))
    debug = os.environ.get("DEBUG", "false").lower() == "true"

    logger.info(f"Server configuration: host={host}, port={port}, debug={debug}")

    # Start server
    app.run(host=host, port=port, debug=debug, threaded=True)


if __name__ == "__main__":
    main()
