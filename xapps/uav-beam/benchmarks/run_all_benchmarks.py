#!/usr/bin/env python3
"""
UAV Beam xApp - Unified Benchmark Runner

Runs all performance benchmarks and generates comprehensive reports.
"""

import sys
import time
import json
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

# Benchmark modules
from benchmark_beam_tracker import run_benchmarks as run_beam_tracker_benchmarks
from benchmark_angle_estimation import run_benchmarks as run_angle_estimation_benchmarks
from benchmark_trajectory import run_benchmarks as run_trajectory_benchmarks


def generate_html_report(results: Dict[str, Any], output_path: str):
    """Generate HTML report from benchmark results"""

    html_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>UAV Beam xApp Performance Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        h1, h2, h3 {{
            color: #333;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 30px;
        }}
        .header h1 {{
            color: white;
            margin: 0;
        }}
        .header p {{
            margin: 10px 0 0 0;
            opacity: 0.9;
        }}
        .card {{
            background: white;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }}
        .summary-item {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .summary-item .value {{
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }}
        .summary-item .label {{
            color: #666;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 15px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
        }}
        tr:hover {{
            background: #f5f5f5;
        }}
        .pass {{
            color: #28a745;
            font-weight: bold;
        }}
        .fail {{
            color: #dc3545;
            font-weight: bold;
        }}
        .metric {{
            display: inline-block;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.85em;
            margin: 2px;
        }}
        .metric-good {{
            background: #d4edda;
            color: #155724;
        }}
        .metric-warning {{
            background: #fff3cd;
            color: #856404;
        }}
        .metric-bad {{
            background: #f8d7da;
            color: #721c24;
        }}
        .section-title {{
            border-bottom: 2px solid #667eea;
            padding-bottom: 10px;
            margin-bottom: 20px;
        }}
        .footer {{
            text-align: center;
            color: #666;
            margin-top: 40px;
            padding: 20px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>UAV Beam xApp Performance Report</h1>
        <p>Generated: {timestamp}</p>
    </div>

    <div class="summary-grid">
        <div class="summary-item">
            <div class="value">{total_benchmarks}</div>
            <div class="label">Total Benchmarks</div>
        </div>
        <div class="summary-item">
            <div class="value">{targets_met}</div>
            <div class="label">Targets Met</div>
        </div>
        <div class="summary-item">
            <div class="value">{avg_latency:.3f} ms</div>
            <div class="label">Avg Latency</div>
        </div>
        <div class="summary-item">
            <div class="value">{total_ops:.0f}/s</div>
            <div class="label">Total Throughput</div>
        </div>
    </div>

    {sections}

    <div class="footer">
        <p>UAV Beam xApp Performance Benchmarks</p>
        <p>Target: Single decision latency &lt; 1ms (p99)</p>
    </div>
</body>
</html>"""

    # Generate sections
    sections = []

    # Beam Tracker section
    if "beam_tracker" in results:
        beam_results = results["beam_tracker"]
        section = """
    <div class="card">
        <h2 class="section-title">Beam Tracker Performance</h2>
        <table>
            <tr>
                <th>Benchmark</th>
                <th>Iterations</th>
                <th>p50 (ms)</th>
                <th>p95 (ms)</th>
                <th>p99 (ms)</th>
                <th>Throughput</th>
                <th>Target</th>
            </tr>
            {rows}
        </table>
    </div>"""

        rows = []
        for r in beam_results.get("results", []):
            target_class = "pass" if r.get("target_met", False) else "fail"
            target_text = "PASS" if r.get("target_met", False) else "FAIL"
            rows.append(f"""
            <tr>
                <td>{r['name']}</td>
                <td>{r['iterations']}</td>
                <td>{r['latency_ms']['p50']:.4f}</td>
                <td>{r['latency_ms']['p95']:.4f}</td>
                <td>{r['latency_ms']['p99']:.4f}</td>
                <td>{r['throughput_ops']:.0f} ops/s</td>
                <td class="{target_class}">{target_text}</td>
            </tr>""")

        sections.append(section.format(rows="\n".join(rows)))

    # Angle Estimation section
    if "angle_estimation" in results:
        angle_results = results["angle_estimation"]
        section = """
    <div class="card">
        <h2 class="section-title">Angle Estimation Performance</h2>
        <table>
            <tr>
                <th>Method</th>
                <th>Antennas</th>
                <th>p50 (ms)</th>
                <th>p99 (ms)</th>
                <th>RMSE (deg)</th>
                <th>Throughput</th>
            </tr>
            {rows}
        </table>
    </div>"""

        rows = []
        for r in angle_results.get("results", []):
            rows.append(f"""
            <tr>
                <td>{r['method']}</td>
                <td>{r['antenna_elements']}</td>
                <td>{r['latency_ms']['p50']:.4f}</td>
                <td>{r['latency_ms']['p99']:.4f}</td>
                <td>{r['accuracy']['rmse_deg']:.4f}</td>
                <td>{r['throughput_ops']:.0f} ops/s</td>
            </tr>""")

        sections.append(section.format(rows="\n".join(rows)))

        # Method comparison
        if "method_comparison" in angle_results:
            section = """
    <div class="card">
        <h2 class="section-title">Method Comparison (64 elements)</h2>
        <table>
            <tr>
                <th>Method</th>
                <th>Mean Latency (ms)</th>
                <th>p95 Latency (ms)</th>
                <th>RMSE (deg)</th>
                <th>Throughput</th>
            </tr>
            {rows}
        </table>
    </div>"""
            rows = []
            for method, data in angle_results["method_comparison"].items():
                rows.append(f"""
            <tr>
                <td>{method.upper()}</td>
                <td>{data['latency_ms']['mean']:.4f}</td>
                <td>{data['latency_ms']['p95']:.4f}</td>
                <td>{data['accuracy']['rmse_deg']:.4f}</td>
                <td>{data['throughput_ops']:.0f} ops/s</td>
            </tr>""")
            sections.append(section.format(rows="\n".join(rows)))

    # Trajectory section
    if "trajectory" in results:
        traj_results = results["trajectory"]
        section = """
    <div class="card">
        <h2 class="section-title">Trajectory Predictor Performance</h2>
        <table>
            <tr>
                <th>Benchmark</th>
                <th>Iterations</th>
                <th>p50 (ms)</th>
                <th>p95 (ms)</th>
                <th>p99 (ms)</th>
                <th>Throughput</th>
            </tr>
            {rows}
        </table>
    </div>"""

        rows = []
        for r in traj_results.get("results", []):
            rows.append(f"""
            <tr>
                <td>{r['name']}</td>
                <td>{r['iterations']}</td>
                <td>{r['latency_ms']['p50']:.4f}</td>
                <td>{r['latency_ms']['p95']:.4f}</td>
                <td>{r['latency_ms']['p99']:.4f}</td>
                <td>{r['throughput_ops']:.0f} ops/s</td>
            </tr>""")

        sections.append(section.format(rows="\n".join(rows)))

        # Accuracy vs horizon
        if "accuracy_vs_horizon" in traj_results:
            section = """
    <div class="card">
        <h2 class="section-title">Prediction Accuracy vs Horizon</h2>
        <table>
            <tr>
                <th>Horizon (ms)</th>
                <th>Kalman RMSE (m)</th>
                <th>Polynomial RMSE (m)</th>
                <th>Hybrid RMSE (m)</th>
            </tr>
            {rows}
        </table>
    </div>"""
            rows = []
            for horizon, data in sorted(traj_results["accuracy_vs_horizon"].items(), key=lambda x: float(x[0])):
                rows.append(f"""
            <tr>
                <td>{horizon}</td>
                <td>{data['kalman_rmse_m']:.4f}</td>
                <td>{data['polynomial_rmse_m']:.4f}</td>
                <td>{data['hybrid_rmse_m']:.4f}</td>
            </tr>""")
            sections.append(section.format(rows="\n".join(rows)))

    # Calculate summary metrics
    total_benchmarks = 0
    targets_met = 0
    all_latencies = []
    all_throughputs = []

    for category in ["beam_tracker", "angle_estimation", "trajectory"]:
        if category in results:
            for r in results[category].get("results", []):
                total_benchmarks += 1
                if r.get("target_met", r.get("latency_ms", {}).get("p99", 999) < 1.0):
                    targets_met += 1
                all_latencies.append(r.get("latency_ms", {}).get("mean", 0))
                all_throughputs.append(r.get("throughput_ops", 0))

    avg_latency = sum(all_latencies) / len(all_latencies) if all_latencies else 0
    total_ops = sum(all_throughputs)

    # Generate final HTML
    html = html_template.format(
        timestamp=results.get("timestamp", datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
        total_benchmarks=total_benchmarks,
        targets_met=targets_met,
        avg_latency=avg_latency,
        total_ops=total_ops,
        sections="\n".join(sections)
    )

    with open(output_path, 'w') as f:
        f.write(html)

    print(f"HTML report generated: {output_path}")


def run_all_benchmarks(
    output_dir: str = "results",
    skip_slow: bool = False,
    generate_html: bool = True
) -> Dict[str, Any]:
    """Run all benchmarks and generate reports"""

    print("=" * 70)
    print("UAV Beam xApp - Comprehensive Performance Benchmark Suite")
    print("=" * 70)
    print(f"\nStart time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output directory: {output_dir}")
    print("-" * 70)

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "version": "1.0.0",
    }

    start_time = time.time()

    # Run beam tracker benchmarks
    print("\n[1/3] Running Beam Tracker Benchmarks...")
    try:
        beam_results = run_beam_tracker_benchmarks(
            str(output_path / "beam_tracker_results.json")
        )
        results["beam_tracker"] = beam_results
    except Exception as e:
        print(f"  ERROR: {e}")
        results["beam_tracker"] = {"error": str(e)}

    # Run angle estimation benchmarks
    print("\n[2/3] Running Angle Estimation Benchmarks...")
    if not skip_slow:
        try:
            angle_results = run_angle_estimation_benchmarks(
                str(output_path / "angle_estimation_results.json")
            )
            results["angle_estimation"] = angle_results
        except Exception as e:
            print(f"  ERROR: {e}")
            results["angle_estimation"] = {"error": str(e)}
    else:
        print("  SKIPPED (--skip-slow)")

    # Run trajectory benchmarks
    print("\n[3/3] Running Trajectory Predictor Benchmarks...")
    try:
        traj_results = run_trajectory_benchmarks(
            str(output_path / "trajectory_results.json")
        )
        results["trajectory"] = traj_results
    except Exception as e:
        print(f"  ERROR: {e}")
        results["trajectory"] = {"error": str(e)}

    total_time = time.time() - start_time
    results["total_runtime_seconds"] = round(total_time, 2)

    # Save combined results
    combined_path = output_path / "combined_results.json"
    with open(combined_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nCombined results saved to: {combined_path}")

    # Generate HTML report
    if generate_html:
        html_path = output_path / "performance_report.html"
        generate_html_report(results, str(html_path))

    # Print summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"Total runtime: {total_time:.1f} seconds")

    # Count results
    total = 0
    passed = 0
    for category in ["beam_tracker", "angle_estimation", "trajectory"]:
        if category in results and "results" in results[category]:
            for r in results[category]["results"]:
                total += 1
                if r.get("target_met", r.get("latency_ms", {}).get("p99", 999) < 1.0):
                    passed += 1

    print(f"Benchmarks run: {total}")
    print(f"Targets met (< 1ms p99): {passed}/{total}")
    print("=" * 70)

    return results


def main():
    parser = argparse.ArgumentParser(
        description="UAV Beam xApp Performance Benchmark Suite"
    )
    parser.add_argument(
        "--output-dir", "-o",
        default=str(Path(__file__).parent / "results"),
        help="Output directory for results"
    )
    parser.add_argument(
        "--skip-slow",
        action="store_true",
        help="Skip slow benchmarks (angle estimation)"
    )
    parser.add_argument(
        "--no-html",
        action="store_true",
        help="Skip HTML report generation"
    )
    parser.add_argument(
        "--json-only",
        action="store_true",
        help="Output JSON only (no console output)"
    )

    args = parser.parse_args()

    results = run_all_benchmarks(
        output_dir=args.output_dir,
        skip_slow=args.skip_slow,
        generate_html=not args.no_html
    )

    if args.json_only:
        print(json.dumps(results, indent=2))

    return 0 if results else 1


if __name__ == "__main__":
    sys.exit(main())
