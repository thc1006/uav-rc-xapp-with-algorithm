#!/usr/bin/env python3
"""
TRACTOR → ns-O-RAN Traffic Converter

Converts real 5G traffic traces from TRACTOR dataset to ns-3 format
for integration with UAV Policy xApp testing
"""

import json
import csv
import argparse
from pathlib import Path
from typing import List, Dict, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrafficConverter:
    """Convert TRACTOR traffic format to ns-3 compatible format"""

    def __init__(self, input_file: str):
        self.input_file = Path(input_file)
        self.traffic_data = []

    def load_traffic(self) -> bool:
        """Load traffic data from TRACTOR dataset"""
        try:
            if self.input_file.suffix == '.json':
                with open(self.input_file) as f:
                    self.traffic_data = json.load(f)
            elif self.input_file.suffix == '.csv':
                with open(self.input_file) as f:
                    reader = csv.DictReader(f)
                    self.traffic_data = list(reader)
            else:
                logger.error(f"Unsupported format: {self.input_file.suffix}")
                return False

            logger.info(f"Loaded {len(self.traffic_data)} traffic records")
            return True
        except Exception as e:
            logger.error(f"Failed to load traffic: {e}")
            return False

    def convert_to_ns3_indication(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert TRACTOR traffic record to ns-3 E2 indication format
        Compatible with UAV Policy xApp
        """
        indication = {
            "timestamp": record.get("timestamp", 0),
            "uav_id": f"UAV-{record.get('ue_id', 'unknown')}",
            "position": {
                "x": float(record.get("pos_x", 0)),
                "y": float(record.get("pos_y", 0)),
                "z": float(record.get("pos_z", 50))  # Default altitude
            },
            "path_position": float(record.get("path_pos", 0)),
            "slice_id": record.get("slice_id", "slice-eMBB"),
            "radio_snapshot": {
                "serving_cell_id": record.get("serving_cell", "cell_001"),
                "neighbor_cell_ids": record.get("neighbor_cells", []).split(",")
                    if isinstance(record.get("neighbor_cells"), str)
                    else record.get("neighbor_cells", []),
                "rsrp_serving": float(record.get("rsrp_serving", -85.0)),
                "rsrp_best_neighbor": float(record.get("rsrp_best_neighbor", -90.0)),
                "prb_utilization_serving": float(record.get("dl_prb_util", 0.5)) / 100.0,
                "prb_utilization_slice": float(record.get("slice_prb_util", 0.3)) / 100.0
                    if "slice_prb_util" in record else None
            }
        }

        # Add service profile if traffic type is known
        if "traffic_type" in record:
            service_profiles = {
                "video_streaming": {
                    "name": "Video-Streaming",
                    "target_bitrate_mbps": 8.0,
                    "min_sinr_db": -5.0
                },
                "voip": {
                    "name": "VoIP",
                    "target_bitrate_mbps": 0.064,
                    "min_sinr_db": -10.0
                },
                "web_browsing": {
                    "name": "Web-Browsing",
                    "target_bitrate_mbps": 2.0,
                    "min_sinr_db": -8.0
                },
                "iot": {
                    "name": "IoT",
                    "target_bitrate_mbps": 0.1,
                    "min_sinr_db": -15.0
                }
            }

            traffic_type = record.get("traffic_type", "").lower()
            if traffic_type in service_profiles:
                indication["service_profile"] = service_profiles[traffic_type]

        return indication

    def convert_all(self) -> List[Dict[str, Any]]:
        """Convert all records"""
        logger.info("Converting traffic records...")
        converted = [self.convert_to_ns3_indication(record)
                    for record in self.traffic_data]
        logger.info(f"Converted {len(converted)} records")
        return converted

    def save_as_json(self, output_file: str):
        """Save converted traffic as JSON"""
        converted = self.convert_all()
        with open(output_file, 'w') as f:
            json.dump(converted, f, indent=2)
        logger.info(f"Saved to {output_file}")

    def save_as_jsonl(self, output_file: str):
        """Save converted traffic as JSONL (one JSON per line)"""
        converted = self.convert_all()
        with open(output_file, 'w') as f:
            for record in converted:
                f.write(json.dumps(record) + '\n')
        logger.info(f"Saved to {output_file} (JSONL format)")


class TrafficSimulator:
    """Simulate traffic using converted data with ns-O-RAN"""

    def __init__(self, converted_data: str, ns_oran_path: str = "/opt/ns-oran"):
        self.converted_data = converted_data
        self.ns_oran_path = Path(ns_oran_path)

    def create_ns3_scenario(self, output_dir: str):
        """Create ns-3 scenario file using traffic data"""
        import os
        os.makedirs(output_dir, exist_ok=True)

        script = f"""#!/usr/bin/env python3
# Auto-generated ns-3 scenario from TRACTOR dataset
import json

# Load converted traffic
with open('{self.converted_data}') as f:
    traffic = json.load(f)

print(f"Loaded {{len(traffic)}} traffic samples")
for i, record in enumerate(traffic[:10]):  # First 10 samples
    print(f"Sample {{i}}: UAV={{record['uav_id']}} → Cell={{record['radio_snapshot']['serving_cell_id']}}")
"""

        script_path = Path(output_dir) / "oran_traffic_scenario.py"
        with open(script_path, 'w') as f:
            f.write(script)
        logger.info(f"Created scenario script: {script_path}")
        return script_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert TRACTOR traffic to ns-3 format"
    )
    parser.add_argument("input", help="Input TRACTOR traffic file (JSON/CSV)")
    parser.add_argument("-o", "--output", help="Output file (JSON/JSONL)")
    parser.add_argument("--format", choices=["json", "jsonl"], default="jsonl",
                       help="Output format")

    args = parser.parse_args()

    # Convert
    converter = TrafficConverter(args.input)
    if not converter.load_traffic():
        return 1

    # Save
    output_file = args.output or f"{Path(args.input).stem}_converted.{args.format}"
    if args.format == "json":
        converter.save_as_json(output_file)
    else:
        converter.save_as_jsonl(output_file)

    logger.info(f"✓ Conversion complete: {output_file}")
    return 0


if __name__ == "__main__":
    exit(main())
