#!/usr/bin/env python3
"""Render performance dashboard from benchmark results.

Reads artifacts/perf/report.json and generates:
- docs/perf/dashboard.md (tables)
- docs/perf/diagrams.md (Mermaid charts)
- artifacts/perf/perfcard.md (performance card)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Render performance dashboard")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/perf/report.json"),
        help="Input report JSON",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("docs/perf"),
        help="Output directory for markdown files",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        default=Path("artifacts/perf"),
        help="Output directory for perfcard",
    )
    return parser.parse_args()


def load_report(path: Path) -> dict:
    """Load benchmark report."""
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def render_dashboard(report: dict, output_path: Path) -> None:
    """Render dashboard.md with performance tables."""
    env = report.get("environment", {})
    model = report.get("model", {})
    summary = report.get("summary", {})
    top_line = summary.get("top_line_metrics", {})
    benchmarks = report.get("benchmarks", [])

    # Group benchmarks by category
    by_category = {}
    for b in benchmarks:
        cat = b.get("category", "unknown")
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(b)

    # Group by modality x task for matrix
    matrix = {}
    for b in benchmarks:
        task = b.get("task", "unknown")
        modality = b.get("modality", "N/A") or "N/A"
        if task not in matrix:
            matrix[task] = {}
        if modality not in matrix[task]:
            matrix[task][modality] = []
        matrix[task][modality].append(b)

    md = f"""# Performance Dashboard

> Generated: {report.get('timestamp', datetime.utcnow().isoformat())}

## Environment

| Component | Value |
|-----------|-------|
| Python | {env.get('python', 'N/A')} |
| PyTorch | {env.get('torch', 'N/A')} |
| CUDA | {env.get('cuda', 'N/A')} |
| OS | {env.get('os', 'N/A')} {env.get('os_version', '')} |
| CPU | {env.get('cpu', 'N/A')} |
| CPU Cores | {env.get('cpu_count', 'N/A')} |
| GPU | {env.get('gpu', 'N/A')} |
| RAM | {env.get('ram_gb', 'N/A')} GB |
| VRAM | {env.get('vram_gb', 'N/A')} GB |

## Model

| Field | Value |
|-------|-------|
| Version | {model.get('version', 'N/A')} |
| Core Model | {model.get('core_model_id', 'N/A')} |
| Git SHA | {model.get('git_sha', 'N/A')} |
| Git Branch | {model.get('git_branch', 'N/A')} |
| Config Hash | {model.get('config_hash', 'N/A')[:16] if model.get('config_hash') else 'N/A'} |

---

## Performance Summary

| Metric | Value |
|--------|-------|
| Total Benchmarks | {summary.get('total_benchmarks', 0)} |
| Passed | {summary.get('passed', 0)} |
| Failed | {summary.get('failed', 0)} |
| Skipped | {summary.get('skipped', 0)} |

### Top-Line Metrics

| Metric | Value |
|--------|-------|
| E2E Latency p50 | {top_line.get('e2e_latency_p50_ms', 'N/A')} ms |
| E2E Latency p95 | {top_line.get('e2e_latency_p95_ms', 'N/A')} ms |
| E2E Throughput | {top_line.get('e2e_throughput_items_per_sec', 'N/A')} items/sec |
| Peak Memory (RSS) | {top_line.get('peak_memory_rss_mb', 'N/A')} MB |
| Peak Memory (VRAM) | {top_line.get('peak_memory_vram_mb', 'N/A')} MB |

---

## Benchmark Results by Category

"""

    for category, results in sorted(by_category.items()):
        passed = sum(1 for r in results if r.get("success"))
        failed = len(results) - passed
        avg_p95 = 0
        count = 0
        for r in results:
            if r.get("latency_ms") and r.get("success"):
                avg_p95 += r["latency_ms"].get("p95", 0)
                count += 1
        avg_p95 = round(avg_p95 / count, 2) if count > 0 else "N/A"

        md += f"""### {category.replace('_', ' ').title()}

| Tests | Passed | Failed | Avg p95 |
|-------|--------|--------|---------|
| {len(results)} | {passed} | {failed} | {avg_p95} ms |

"""

    # Capability matrix
    modalities = sorted(set(b.get("modality", "N/A") or "N/A" for b in benchmarks))
    tasks = sorted(matrix.keys())

    if len(modalities) > 1 and len(tasks) > 1:
        md += """---

## Capability/Performance Matrix

| Task | """ + " | ".join(modalities) + """ |
|------|""" + "|".join(["------"] * len(modalities)) + """|
"""
        for task in tasks:
            row = [task]
            for mod in modalities:
                results = matrix.get(task, {}).get(mod, [])
                if results:
                    successful = [r for r in results if r.get("success") and r.get("latency_ms")]
                    if successful:
                        avg_p95 = sum(r["latency_ms"]["p95"] for r in successful) / len(successful)
                        row.append(f"{avg_p95:.1f} ms")
                    else:
                        row.append("FAIL")
                else:
                    row.append("-")
            md += "| " + " | ".join(row) + " |\n"

    # Regression status
    regression = summary.get("regression_status", {})
    if regression.get("baseline_file"):
        md += f"""
---

## Regression Status

Baseline: `{regression.get('baseline_file')}`

"""
        if regression.get("has_regressions"):
            md += f"""> [!WARNING]
> **{regression.get('regression_count', 0)} regressions detected**

| Benchmark | Metric | Baseline | Current | Change |
|-----------|--------|----------|---------|--------|
"""
            for r in regression.get("regressions", []):
                md += f"| {r['benchmark_id']} | {r['metric']} | {r['baseline_value']} | {r['current_value']} | {r['change_percent']}% |\n"
        else:
            md += """> [!NOTE]
> No regressions detected.
"""

    md += """
---

*This dashboard is auto-generated. Do not edit manually.*
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(md)


def render_diagrams(report: dict, output_path: Path) -> None:
    """Render diagrams.md with Mermaid charts."""
    benchmarks = report.get("benchmarks", [])

    # Group by category for pie chart
    by_category = {}
    for b in benchmarks:
        cat = b.get("category", "unknown")
        by_category[cat] = by_category.get(cat, 0) + 1

    md = """# Performance Diagrams

## Benchmark Distribution by Category

```mermaid
pie title Benchmarks by Category
"""
    for cat, count in sorted(by_category.items()):
        md += f'    "{cat}" : {count}\n'
    md += "```\n\n"

    # Pipeline timing breakdown
    md += """## Pipeline Timing Breakdown

```mermaid
flowchart LR
    A[Ingest] --> B[Preprocess]
    B --> C[Inference]
    C --> D[XAI Dossier]
    D --> E[Export]
    
    subgraph Timing
    A -.->|~50ms| B
    B -.->|~100ms| C
    C -.->|~500ms| D
    D -.->|~200ms| E
    end
```

## Benchmark Execution Flow

```mermaid
flowchart TD
    Start([Start]) --> Config[Load Config]
    Config --> Warmup[Warmup Runs]
    Warmup --> Measure[Measurement Runs]
    Measure --> Stats[Compute Stats]
    Stats --> Memory[Collect Memory]
    Memory --> Report[Generate Report]
    Report --> End([End])
    
    subgraph Timing Collection
    Warmup -->|discard| Measure
    Measure -->|record| Stats
    end
```

## Artifact/Reporting Flow

```mermaid
flowchart LR
    Tests[pytest tests/perf/] --> Harness[PerfHarness]
    Harness --> JSON[report.json]
    JSON --> Dashboard[dashboard.md]
    JSON --> Perfcard[perfcard.md]
    JSON --> Diagrams[diagrams.md]
    
    subgraph Artifacts
    JSON
    Dashboard
    Perfcard
    Diagrams
    end
```
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(md)


def render_perfcard(report: dict, output_path: Path) -> None:
    """Render performance card."""
    env = report.get("environment", {})
    model = report.get("model", {})
    summary = report.get("summary", {})
    top_line = summary.get("top_line_metrics", {})

    md = f"""# Performance Card

## Run Information

| Field | Value |
|-------|-------|
| **Timestamp** | {report.get('timestamp', 'N/A')} |
| **Version** | {model.get('version', 'N/A')} |
| **Git SHA** | {model.get('git_sha', 'N/A')} |
| **Device** | {env.get('gpu', 'CPU')} |

## Environment

| Component | Value |
|-----------|-------|
| Python | {env.get('python', 'N/A')} |
| PyTorch | {env.get('torch', 'N/A')} |
| CUDA | {env.get('cuda', 'N/A')} |
| OS | {env.get('os', 'N/A')} |
| CPU | {env.get('cpu', 'N/A')} |
| GPU | {env.get('gpu', 'N/A')} |
| RAM | {env.get('ram_gb', 'N/A')} GB |
| VRAM | {env.get('vram_gb', 'N/A')} GB |

## Summary Metrics

### Latency (End-to-End)

| Percentile | Value |
|------------|-------|
| p50 | {top_line.get('e2e_latency_p50_ms', 'N/A')} ms |
| p95 | {top_line.get('e2e_latency_p95_ms', 'N/A')} ms |

### Throughput

| Metric | Value |
|--------|-------|
| Items/sec | {top_line.get('e2e_throughput_items_per_sec', 'N/A')} |

### Memory

| Metric | Value |
|--------|-------|
| Peak RSS | {top_line.get('peak_memory_rss_mb', 'N/A')} MB |
| Peak VRAM | {top_line.get('peak_memory_vram_mb', 'N/A')} MB |

## Results

| Category | Tests | Passed | Failed |
|----------|-------|--------|--------|
| Total | {summary.get('total_benchmarks', 0)} | {summary.get('passed', 0)} | {summary.get('failed', 0)} |

---

> [!CAUTION]
> This is a research system. Metrics are for internal benchmarking only.
"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(md)


def main() -> int:
    """Main entry point."""
    args = parse_args()

    report = load_report(args.input)

    if not report:
        print(f"No report found at {args.input}")
        print("Run benchmarks first: python scripts/run_perf_benchmarks.py --smoke")
        return 1

    # Render outputs
    render_dashboard(report, args.output_dir / "dashboard.md")
    print(f"Dashboard: {args.output_dir / 'dashboard.md'}")

    render_diagrams(report, args.output_dir / "diagrams.md")
    print(f"Diagrams: {args.output_dir / 'diagrams.md'}")

    render_perfcard(report, args.artifacts_dir / "perfcard.md")
    print(f"Perfcard: {args.artifacts_dir / 'perfcard.md'}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
