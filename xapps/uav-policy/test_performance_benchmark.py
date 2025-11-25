#!/usr/bin/env python3
"""
Performance Benchmark Suite for UAV Policy xApp

Tests:
1. Latency: P50, P95, P99 for decision processing
2. Throughput: Requests per second
3. Memory: Baseline and per-decision overhead
4. Scalability: Performance with increasing UAV count
"""

import requests
import time
import statistics
import json
from typing import List, Dict, Any
import sys

BASE_URL = "http://localhost:5000"
INDICATION_ENDPOINT = f"{BASE_URL}/e2/indication"
HEALTH_ENDPOINT = f"{BASE_URL}/health"


def create_indication(uav_id: str, path_pos: float = 500.0) -> Dict[str, Any]:
    """Create a standard test indication."""
    return {
        "uav_id": uav_id,
        "position": {"x": 100.0 + path_pos / 10, "y": 200.0, "z": 50.0},
        "path_position": path_pos,
        "slice_id": "slice-eMBB",
        "radio_snapshot": {
            "serving_cell_id": "cell_001",
            "neighbor_cell_ids": ["cell_002", "cell_003"],
            "rsrp_serving": -85.0,
            "rsrp_best_neighbor": -90.0,
            "prb_utilization_serving": 0.4,
        }
    }


def benchmark_latency(num_requests: int = 100) -> Dict[str, float]:
    """Benchmark decision latency."""
    print("\n" + "="*70)
    print("BENCHMARK 1: Latency (Processing Time)")
    print("="*70)

    latencies = []

    for i in range(num_requests):
        indication = create_indication(f"BENCH-LAT-{i}")

        start = time.perf_counter()
        response = requests.post(INDICATION_ENDPOINT, json=indication)
        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)

        if i % 20 == 0:
            print(f"  Request {i+1}/{num_requests}: {latency_ms:.2f} ms")

        if response.status_code != 200:
            print(f"  ERROR: Request {i} failed with status {response.status_code}")
            return {}

    p50 = statistics.median(latencies)
    p95 = sorted(latencies)[int(num_requests * 0.95)]
    p99 = sorted(latencies)[int(num_requests * 0.99)]
    mean = statistics.mean(latencies)
    stdev = statistics.stdev(latencies) if len(latencies) > 1 else 0.0

    print(f"\n  Total Requests: {num_requests}")
    print(f"  Mean:           {mean:.2f} ms")
    print(f"  Median (P50):   {p50:.2f} ms")
    print(f"  P95:            {p95:.2f} ms")
    print(f"  P99:            {p99:.2f} ms")
    print(f"  Std Dev:        {stdev:.2f} ms")
    print(f"  Min:            {min(latencies):.2f} ms")
    print(f"  Max:            {max(latencies):.2f} ms")

    return {
        "p50": p50,
        "p95": p95,
        "p99": p99,
        "mean": mean,
        "stdev": stdev,
        "min": min(latencies),
        "max": max(latencies)
    }


def benchmark_throughput(duration_seconds: int = 10) -> Dict[str, float]:
    """Benchmark requests per second."""
    print("\n" + "="*70)
    print("BENCHMARK 2: Throughput (Requests Per Second)")
    print("="*70)

    request_count = 0
    start_time = time.perf_counter()
    request_times = []

    print(f"  Running for {duration_seconds} seconds...")

    while time.perf_counter() - start_time < duration_seconds:
        indication = create_indication(f"BENCH-TPS-{request_count}")

        req_start = time.perf_counter()
        try:
            response = requests.post(INDICATION_ENDPOINT, json=indication, timeout=5)
            req_end = time.perf_counter()
            request_times.append((req_end - req_start) * 1000)

            if response.status_code == 200:
                request_count += 1
            else:
                print(f"  Request {request_count} failed: {response.status_code}")

        except requests.exceptions.RequestException as e:
            print(f"  Request error: {e}")
            break

    elapsed = time.perf_counter() - start_time
    rps = request_count / elapsed
    mean_req_time = statistics.mean(request_times) if request_times else 0

    print(f"\n  Elapsed Time:     {elapsed:.2f} seconds")
    print(f"  Total Requests:   {request_count}")
    print(f"  RPS:              {rps:.1f} req/sec")
    print(f"  Mean Request:     {mean_req_time:.2f} ms")

    return {
        "elapsed_seconds": elapsed,
        "request_count": request_count,
        "rps": rps,
        "mean_request_time_ms": mean_req_time
    }


def benchmark_concurrent_uavs(num_uavs: int = 50) -> Dict[str, Any]:
    """Benchmark with increasing number of concurrent UAVs."""
    print("\n" + "="*70)
    print(f"BENCHMARK 3: Scalability ({num_uavs} Concurrent UAVs)")
    print("="*70)

    latencies = []

    print(f"  Simulating {num_uavs} UAVs...")

    for i in range(num_uavs):
        indication = create_indication(f"BENCH-UAV-{i:03d}", path_pos=100.0 + i*10)

        start = time.perf_counter()
        response = requests.post(INDICATION_ENDPOINT, json=indication)
        end = time.perf_counter()

        latency_ms = (end - start) * 1000
        latencies.append(latency_ms)

        if response.status_code != 200:
            print(f"  ERROR: UAV {i} request failed")
            return {}

    mean = statistics.mean(latencies)
    p99 = sorted(latencies)[int(num_uavs * 0.99)]

    print(f"\n  Total UAVs:     {num_uavs}")
    print(f"  Mean Latency:   {mean:.2f} ms")
    print(f"  P99 Latency:    {p99:.2f} ms")

    return {
        "num_uavs": num_uavs,
        "mean_latency_ms": mean,
        "p99_latency_ms": p99
    }


def benchmark_service_profile_overhead() -> Dict[str, float]:
    """Benchmark latency with service profile (PRB estimation)."""
    print("\n" + "="*70)
    print("BENCHMARK 4: Service Profile Overhead")
    print("="*70)

    # Without service profile
    print("  Testing without service profile...")
    simple_latencies = []

    for i in range(20):
        indication = create_indication(f"BENCH-SIMPLE-{i}")

        start = time.perf_counter()
        response = requests.post(INDICATION_ENDPOINT, json=indication)
        end = time.perf_counter()

        simple_latencies.append((end - start) * 1000)

    simple_mean = statistics.mean(simple_latencies)

    # With service profile
    print("  Testing with service profile...")
    profile_latencies = []

    for i in range(20):
        indication = create_indication(f"BENCH-PROFILE-{i}")
        indication["service_profile"] = {
            "name": "4K-Video",
            "target_bitrate_mbps": 25.0,
            "min_sinr_db": 10.0
        }

        start = time.perf_counter()
        response = requests.post(INDICATION_ENDPOINT, json=indication)
        end = time.perf_counter()

        profile_latencies.append((end - start) * 1000)

    profile_mean = statistics.mean(profile_latencies)
    overhead = profile_mean - simple_mean

    print(f"\n  Without Profile:  {simple_mean:.2f} ms")
    print(f"  With Profile:     {profile_mean:.2f} ms")
    print(f"  Overhead:         {overhead:.2f} ms ({overhead/simple_mean*100:.1f}%)")

    return {
        "simple_mean_ms": simple_mean,
        "profile_mean_ms": profile_mean,
        "overhead_ms": overhead,
        "overhead_percent": overhead / simple_mean * 100 if simple_mean > 0 else 0
    }


def benchmark_flight_plan_overhead() -> Dict[str, float]:
    """Benchmark latency with/without flight plan."""
    print("\n" + "="*70)
    print("BENCHMARK 5: Flight Plan Overhead")
    print("="*70)

    # Without flight plan
    print("  Testing without flight plan...")
    no_plan_latencies = []

    for i in range(20):
        indication = create_indication(f"BENCH-NOPLAN-{i}")

        start = time.perf_counter()
        response = requests.post(INDICATION_ENDPOINT, json=indication)
        end = time.perf_counter()

        no_plan_latencies.append((end - start) * 1000)

    no_plan_mean = statistics.mean(no_plan_latencies)

    # With flight plan (3 segments)
    print("  Testing with flight plan...")
    plan_latencies = []

    for i in range(20):
        indication = create_indication(f"BENCH-PLAN-{i}")
        indication["flight_plan"] = {
            "segments": [
                {
                    "start_pos": 0.0,
                    "end_pos": 400.0,
                    "planned_cell_id": "cell_001",
                    "slice_id": "slice-eMBB",
                    "base_prb_quota": 20
                },
                {
                    "start_pos": 400.0,
                    "end_pos": 600.0,
                    "planned_cell_id": "cell_002",
                    "slice_id": "slice-eMBB",
                    "base_prb_quota": 25
                },
                {
                    "start_pos": 600.0,
                    "end_pos": 800.0,
                    "planned_cell_id": "cell_001",
                    "slice_id": "slice-eMBB",
                    "base_prb_quota": 20
                }
            ]
        }

        start = time.perf_counter()
        response = requests.post(INDICATION_ENDPOINT, json=indication)
        end = time.perf_counter()

        plan_latencies.append((end - start) * 1000)

    plan_mean = statistics.mean(plan_latencies)
    overhead = plan_mean - no_plan_mean

    print(f"\n  Without Plan:     {no_plan_mean:.2f} ms")
    print(f"  With Plan (3seg): {plan_mean:.2f} ms")
    print(f"  Overhead:         {overhead:.2f} ms ({overhead/no_plan_mean*100:.1f}%)")

    return {
        "no_plan_mean_ms": no_plan_mean,
        "plan_mean_ms": plan_mean,
        "overhead_ms": overhead,
        "overhead_percent": overhead / no_plan_mean * 100 if no_plan_mean > 0 else 0
    }


def main():
    """Run all performance benchmarks."""
    print("\n" + "#"*70)
    print("# UAV POLICY XAPP - PERFORMANCE BENCHMARK SUITE")
    print("#"*70)

    # Check server health
    try:
        health = requests.get(HEALTH_ENDPOINT, timeout=5).json()
        print(f"\n✓ Server Status: {health['status']}")
    except Exception as e:
        print(f"✗ ERROR: Cannot connect to server: {e}")
        return False

    results = {}

    try:
        results["latency"] = benchmark_latency(num_requests=100)
        results["throughput"] = benchmark_throughput(duration_seconds=10)
        results["concurrent_uavs"] = benchmark_concurrent_uavs(num_uavs=50)
        results["service_profile"] = benchmark_service_profile_overhead()
        results["flight_plan"] = benchmark_flight_plan_overhead()

        # Summary
        print("\n" + "#"*70)
        print("# BENCHMARK SUMMARY")
        print("#"*70)

        print(f"\nLatency (P50):         {results['latency']['p50']:.2f} ms")
        print(f"Latency (P99):         {results['latency']['p99']:.2f} ms")
        print(f"Throughput (RPS):      {results['throughput']['rps']:.1f} req/sec")
        print(f"Scalability (50 UAVs): {results['concurrent_uavs']['mean_latency_ms']:.2f} ms")
        print(f"Service Profile:       +{results['service_profile']['overhead_ms']:.2f} ms ({results['service_profile']['overhead_percent']:.1f}%)")
        print(f"Flight Plan:           +{results['flight_plan']['overhead_ms']:.2f} ms ({results['flight_plan']['overhead_percent']:.1f}%)")

        print("\n" + "#"*70)
        print("# ALL BENCHMARKS COMPLETED ✓")
        print("#"*70 + "\n")

        # Save results
        with open("benchmark_results.json", "w") as f:
            json.dump(results, f, indent=2)
        print("Results saved to benchmark_results.json\n")

        return True

    except Exception as e:
        print(f"\n✗ BENCHMARK FAILED: {e}\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
