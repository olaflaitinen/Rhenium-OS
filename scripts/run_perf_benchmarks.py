#!/usr/bin/env python3
"""Run performance benchmarks for Rhenium OS.

Usage:
    python scripts/run_perf_benchmarks.py --smoke
    python scripts/run_perf_benchmarks.py --full --json-out
    python scripts/run_perf_benchmarks.py --baseline benchmarks/perf/baselines/v1.0.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import pytest


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run Rhenium OS performance benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--smoke",
        action="store_true",
        help="Run smoke subset (~20 fast tests)",
    )
    mode.add_argument(
        "--full",
        action="store_true",
        help="Run full benchmark suite (50+ tests)",
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda", "auto"],
        default="auto",
        help="Device to run benchmarks on (default: auto)",
    )

    parser.add_argument(
        "--json-out",
        action="store_true",
        help="Output JSON report to artifacts/perf/report.json",
    )

    parser.add_argument(
        "--baseline",
        type=Path,
        help="Baseline JSON file for regression comparison",
    )

    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="Exit with code 1 if regressions detected",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/perf"),
        help="Output directory (default: artifacts/perf)",
    )

    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    return parser.parse_args()


def detect_device() -> str:
    """Auto-detect best available device."""
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def run_benchmarks(args: argparse.Namespace) -> dict:
    """Run performance benchmarks using pytest."""
    # Determine marker
    if args.smoke:
        marker = "perf_smoke"
        mode = "smoke"
    else:
        marker = "perf"
        mode = "full"

    # Determine device
    device = args.device if args.device != "auto" else detect_device()

    # Build pytest args
    pytest_args = [
        "tests/perf/",
        "-m", marker,
    ]

    if args.verbose:
        pytest_args.append("-v")
    else:
        pytest_args.append("-q")

    # Set device via environment
    import os
    os.environ["RHENIUM_PERF_DEVICE"] = device

    print(f"Running {mode} performance benchmarks on {device}...")
    start_time = time.time()

    # Run pytest
    exit_code = pytest.main(pytest_args)

    duration = time.time() - start_time

    # Load results if JSON output requested
    results = {
        "mode": mode,
        "device": device,
        "duration_sec": round(duration, 2),
        "exit_code": exit_code,
    }

    # Try to load generated report
    report_path = args.output_dir / "report.json"
    if report_path.exists():
        try:
            with open(report_path) as f:
                results = json.load(f)
        except Exception:
            pass

    return results


def check_regressions(results: dict, baseline_path: Path) -> tuple[bool, list]:
    """Check for regressions against baseline."""
    if not baseline_path.exists():
        print(f"Baseline file not found: {baseline_path}")
        return False, []

    try:
        with open(baseline_path) as f:
            baseline = json.load(f)
    except Exception as e:
        print(f"Error loading baseline: {e}")
        return False, []

    regressions = results.get("summary", {}).get("regression_status", {}).get("regressions", [])
    return len(regressions) > 0, regressions


def print_summary(results: dict) -> None:
    """Print human-readable summary."""
    print("\n" + "=" * 60)
    print("PERFORMANCE BENCHMARK SUMMARY")
    print("=" * 60)

    summary = results.get("summary", {})
    top_line = summary.get("top_line_metrics", {})

    print(f"\nTotal benchmarks: {summary.get('total_benchmarks', 0)}")
    print(f"Passed: {summary.get('passed', 0)}")
    print(f"Failed: {summary.get('failed', 0)}")
    print(f"Skipped: {summary.get('skipped', 0)}")

    print("\n--- Top-Line Metrics ---")
    print(f"E2E Latency p50: {top_line.get('e2e_latency_p50_ms', 'N/A')} ms")
    print(f"E2E Latency p95: {top_line.get('e2e_latency_p95_ms', 'N/A')} ms")
    print(f"E2E Throughput: {top_line.get('e2e_throughput_items_per_sec', 'N/A')} items/sec")
    print(f"Peak Memory (RSS): {top_line.get('peak_memory_rss_mb', 'N/A')} MB")
    print(f"Peak Memory (VRAM): {top_line.get('peak_memory_vram_mb', 'N/A')} MB")

    regression_status = summary.get("regression_status", {})
    if regression_status.get("has_regressions"):
        print(f"\n[WARNING] {regression_status.get('regression_count', 0)} regressions detected!")
    else:
        print("\n[OK] No regressions detected.")

    print("=" * 60)


def main() -> int:
    """Main entry point."""
    args = parse_args()

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Run benchmarks
    results = run_benchmarks(args)

    # Compare baseline if provided
    if args.baseline:
        from rhenium.bench import PerfHarness
        harness = PerfHarness()
        # Load results and compare
        report_path = args.output_dir / "report.json"
        if report_path.exists():
            regression_status = harness.compare_baseline(args.baseline)
            results["summary"]["regression_status"] = regression_status

            # Update report
            with open(report_path, "w") as f:
                json.dump(results, f, indent=2)

    # Print summary
    print_summary(results)

    # Save JSON if requested
    if args.json_out:
        report_path = args.output_dir / "report.json"
        with open(report_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nReport saved to: {report_path}")

    # Check regression exit code
    if args.fail_on_regression:
        regression_status = results.get("summary", {}).get("regression_status", {})
        if regression_status.get("has_regressions"):
            print("\n[FAIL] Exiting with code 1 due to regressions.")
            return 1

    return 0 if results.get("exit_code", 0) == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
