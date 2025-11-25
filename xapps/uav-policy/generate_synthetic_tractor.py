#!/usr/bin/env python3
"""
Synthetic TRACTOR Dataset Generator

Generates synthetic 5G traffic data compatible with the TRACTOR dataset format
for testing UAV Policy xApp without needing to download the full dataset.
"""

import csv
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import argparse
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SyntheticTractorGenerator:
    """Generate synthetic TRACTOR-compatible metrics"""

    def __init__(self, num_ues=8, num_samples=1000, seed=42):
        """
        Initialize generator

        num_ues: Number of UEs (1-4 eMBB, 5-8 URLLC)
        num_samples: Number of samples to generate
        seed: Random seed for reproducibility
        """
        self.num_ues = num_ues
        self.num_samples = num_samples
        np.random.seed(seed)

        # TRACTOR UE configuration
        self.ue_imsis = [
            "1010123456002", "1010123456003", "1010123456004", "1010123456005",
            "1010123456006", "1010123456007", "1010123456008", "1010123456009"
        ][:num_ues]

        self.ue_types = (
            ["eMBB"] * 4 + ["URLLC"] * 4
        )[:num_ues]

    def generate_ue_metrics(self, ue_id: int) -> list:
        """Generate metrics for a single UE"""
        ue_type = self.ue_types[ue_id - 1]
        metrics = []

        # Simulate mobility and changing radio conditions
        rsrp_baseline = -85.0 if ue_type == "eMBB" else -80.0
        prb_util_baseline = 0.6 if ue_type == "eMBB" else 0.3

        for t in range(self.num_samples):
            # Simulate RSRP variation (slow fading + fast fading)
            slow_fading = np.sin(t / 100) * 5  # Slow variation
            fast_fading = np.random.normal(0, 2)  # Fast fading
            rsrp = rsrp_baseline + slow_fading + fast_fading
            rsrp = np.clip(rsrp, -140, -40)

            # Simulate PRB utilization (based on traffic)
            traffic_pattern = 0.5 + 0.3 * np.sin(t / 50)
            prb_util = prb_util_baseline + traffic_pattern + np.random.normal(0, 0.05)
            prb_util = np.clip(prb_util, 0.0, 1.0)

            # SINR estimation (roughly RSRP + 100)
            sinr = rsrp + 100 + np.random.normal(0, 1)

            # Throughput estimation (depends on SINR and PRB allocation)
            # Shannon capacity: C = BW * log2(1 + SINR)
            # With 10 MHz = 50 PRBs, each PRB ≈ 180 kHz
            prb_count = int(50 * prb_util)
            bw_mhz = (prb_count / 50) * 10
            snr_linear = 10 ** (sinr / 10)
            throughput_mbps = max(0, bw_mhz * np.log2(1 + snr_linear))

            # Latency (depends on queue depth)
            queue_depth = int(prb_util * 100)
            latency_ms = 10 + queue_depth / 10 + np.random.exponential(2)

            metrics.append({
                "timestamp": t,
                "ue_imsi": self.ue_imsis[ue_id - 1],
                "ue_id": ue_id,
                "traffic_type": ue_type,
                "rsrp_serving_dbm": float(f"{rsrp:.1f}"),
                "sinr_db": float(f"{sinr:.1f}"),
                "prb_utilization": float(f"{prb_util:.2f}"),
                "prb_allocation": prb_count,
                "throughput_mbps": float(f"{throughput_mbps:.2f}"),
                "latency_ms": float(f"{latency_ms:.2f}"),
                "packet_loss_rate": max(0, min(0.1, prb_util - 0.8)) * 100,
                "handover_count": int(t / 500)  # One handover every 500 samples
            })

        return metrics

    def generate_enb_metrics(self) -> list:
        """Generate base station metrics (aggregated)"""
        metrics = []

        for t in range(self.num_samples):
            # Aggregate metrics across all UEs
            total_dl_bitrate = 0
            total_ul_bitrate = 0

            for ue_id in range(1, self.num_ues + 1):
                # Simulate per-UE throughput
                ue_type = self.ue_types[ue_id - 1]
                base_rate = 50 if ue_type == "eMBB" else 10  # Mbps
                variation = base_rate * 0.3 * np.sin(t / 100)
                dl_rate = base_rate + variation + np.random.normal(0, 5)
                ul_rate = dl_rate * 0.3  # Asymmetric UL/DL

                total_dl_bitrate += max(0, dl_rate)
                total_ul_bitrate += max(0, ul_rate)

            metrics.append({
                "timestamp": t,
                "dl_bitrate_mbps": float(f"{total_dl_bitrate:.2f}"),
                "ul_bitrate_mbps": float(f"{total_ul_bitrate:.2f}"),
                "total_prb_allocation": int(50 * (0.5 + 0.2 * np.sin(t / 50))),
                "slice_0_allocation": int(30),  # eMBB
                "slice_1_allocation": int(20),  # URLLC
            })

        return metrics

    def save_as_csv(self, output_dir: str):
        """Save metrics as CSV files (TRACTOR format)"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate and save UE metrics
        logger.info(f"Generating metrics for {self.num_ues} UEs...")
        for ue_id in range(1, self.num_ues + 1):
            metrics = self.generate_ue_metrics(ue_id)
            csv_file = output_path / f"{self.ue_imsis[ue_id - 1]}_metrics.csv"

            with open(csv_file, 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=metrics[0].keys())
                writer.writeheader()
                writer.writerows(metrics)

            logger.info(f"Saved {csv_file}")

        # Generate and save eNB metrics
        logger.info("Generating base station metrics...")
        enb_metrics = self.generate_enb_metrics()
        enb_file = output_path / "enb_metrics.csv"

        with open(enb_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=enb_metrics[0].keys())
            writer.writeheader()
            writer.writerows(enb_metrics)

        logger.info(f"Saved {enb_file}")

        # Save metadata
        metadata = {
            "generator": "synthetic_tractor",
            "num_ues": self.num_ues,
            "num_samples": self.num_samples,
            "ue_types": {
                "eMBB": [self.ue_imsis[i] for i in range(min(4, len(self.ue_imsis)))],
                "URLLC": [self.ue_imsis[i] for i in range(4, len(self.ue_imsis))]
            },
            "channel_bandwidth_mhz": 10,
            "prb_count": 50,
            "slice_count": 2,
            "generated_at": datetime.now().isoformat()
        }

        metadata_file = output_path / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved {metadata_file}")
        logger.info(f"✓ Synthetic TRACTOR dataset generated: {output_dir}")

        return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic TRACTOR-compatible dataset"
    )
    parser.add_argument("--output", default="/tmp/synthetic_tractor",
                       help="Output directory")
    parser.add_argument("--num-ues", type=int, default=8,
                       help="Number of UEs (default 8)")
    parser.add_argument("--num-samples", type=int, default=1000,
                       help="Number of time samples (default 1000)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    args = parser.parse_args()

    generator = SyntheticTractorGenerator(
        num_ues=args.num_ues,
        num_samples=args.num_samples,
        seed=args.seed
    )

    generator.save_as_csv(args.output)
    return 0


if __name__ == "__main__":
    exit(main())
