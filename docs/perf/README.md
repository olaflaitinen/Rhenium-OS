# Performance Benchmarks

This directory contains documentation and artifacts for Rhenium OS performance benchmarking.

## Quick Start

```bash
# Run smoke tests (fast, ~20 tests)
python scripts/run_perf_benchmarks.py --smoke

# Run full suite (50+ tests)
python scripts/run_perf_benchmarks.py --full --json-out

# Generate dashboard
python scripts/render_perf_dashboard.py

# Run with pytest directly
pytest tests/perf/ -m perf_smoke -v
pytest tests/perf/ -m perf -v
```

## Directory Structure

```
docs/perf/
├── README.md           # This file
├── metrics.md          # Metric definitions
├── perfcard.md         # Performance card template
├── dashboard.md        # Generated dashboard (after running)
└── diagrams.md         # Generated Mermaid diagrams

benchmarks/perf/
├── schema_perf_results.json   # JSON schema for results
└── baselines/                 # Baseline files for regression

artifacts/perf/
├── report.json         # Generated benchmark report
└── perfcard.md         # Generated performance card
```

## Markers

| Marker | Description | Use Case |
|--------|-------------|----------|
| `perf_smoke` | ~20 fast tests | CI on every PR |
| `perf` | Full suite (50+) | Nightly/manual |
| `gpu` | GPU-only tests | When CUDA available |

## CLI Options

```bash
python scripts/run_perf_benchmarks.py --help

Options:
  --smoke               Run smoke subset
  --full                Run full suite
  --device {cpu,cuda,auto}
  --json-out            Output JSON report
  --baseline PATH       Compare against baseline
  --fail-on-regression  Exit 1 if regressions found
  --output-dir PATH     Output directory
  -v, --verbose         Verbose output
```

## Benchmark Categories

1. **End-to-end core model** — Full pipeline latency
2. **I/O & parsing** — Data loading overhead
3. **Preprocessing** — Normalization, resampling
4. **Perception inference** — Segmentation forward pass
5. **MRI reconstruction** — K-space to image
6. **CT reconstruction** — Sinogram to image
7. **PINN step** — Physics-informed step
8. **GAN inference** — Generator forward pass
9. **XAI dossier** — Evidence generation
10. **Serialization** — Output writing
11. **FastAPI readiness** — API overhead
12. **CLI overhead** — Command-line overhead
13. **Governance artifacts** — Card generation

## Metrics Collected

- **Latency**: p50, p95, p99 (milliseconds)
- **Throughput**: items/sec, voxels/sec
- **Memory**: RSS peak (CPU), VRAM peak (GPU)

See [metrics.md](metrics.md) for formal definitions.

## Regression Detection

```bash
# Run with baseline comparison
python scripts/run_perf_benchmarks.py --full \
  --baseline benchmarks/perf/baselines/v1.0.json \
  --fail-on-regression
```

Thresholds:
- Latency: 10% increase = regression
- Memory: 20% increase = regression
- Throughput: 10% decrease = regression

## Adding New Benchmarks

1. Add test to `tests/perf/test_perf_matrix.py`
2. Use `@pytest.mark.perf` marker
3. Use `PerfHarness` for consistent timing

```python
@pytest.mark.perf
def test_my_benchmark(perf_harness):
    def run():
        # Your benchmark code
        pass
    
    result = perf_harness.run_benchmark(
        func=run,
        benchmark_id="my_benchmark",
        category="custom",
        task="my_task",
    )
    assert result.success
```
