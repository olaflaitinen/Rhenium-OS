# Rhenium OS Benchmarks

This directory contains documentation for the Rhenium OS benchmark suite.

## Quick Start

```bash
# Run smoke tests (fast, for CI)
pytest tests/bench/ -m smoke -v

# Run full benchmark matrix (576+ tests)
pytest tests/bench/ -m bench -v

# Use the benchmark runner script
python scripts/run_benchmarks.py --smoke
python scripts/run_benchmarks.py --full --json
```

## Test Organization

| Directory | Description |
|-----------|-------------|
| `tests/smoke/` | Core model smoke tests |
| `tests/bench/` | Full benchmark test suite |

## Benchmark Categories

The benchmark suite covers 12 categories:

1. **Functional Correctness** - Schema validation, shape invariants
2. **Numerical Stability** - NaN/Inf detection, range preservation
3. **Determinism** - Same seed produces identical outputs
4. **Performance** - Throughput, latency, memory usage
5. **Robustness** - Noise, blur, missing data handling
6. **Calibration** - Confidence score sanity checks
7. **Fairness** - Subgroup metric infrastructure
8. **Privacy** - No PHI leakage verification
9. **API Readiness** - FastAPI endpoint tests
10. **CLI Readiness** - Command-line interface tests
11. **Governance** - Model card, risk register generation
12. **Generative Disclosure** - AI content disclosure compliance

## Test Matrix

The parametrized matrix generates 576+ test cases:

```
Modalities: [MRI, CT, US, XR]           = 4
Tasks:      [segmentation, ...]         = 6
Shapes:     [3D and 2D variants]        = 6
Dtypes:     [float32, float16]          = 2
Devices:    [cpu, cuda]                 = 2

Total: 4 × 6 × 6 × 2 × 2 = 576 cases
```

## Pytest Markers

| Marker | Description | Use Case |
|--------|-------------|----------|
| `smoke` | Fast subset (~50 tests) | CI on every PR |
| `bench` | Full matrix (576+ tests) | Nightly runs |
| `gpu` | GPU-only tests | When CUDA available |
| `slow` | Long-running tests | Optional |

## Running Specific Tests

```bash
# Only functional benchmarks
pytest tests/bench/test_functional_benchmarks.py -v

# Only determinism tests
pytest tests/bench/test_determinism.py -v

# GPU tests only (if available)
pytest tests/bench/ -m gpu -v

# Exclude slow tests
pytest tests/bench/ -m "not slow" -v
```

## Benchmark Report

The runner generates JSON reports:

```bash
python scripts/run_benchmarks.py --full --json
cat artifacts/benchmarks/report.json
```

Report structure:
```json
{
  "metadata": {
    "timestamp": "2025-01-01T00:00:00",
    "mode": "full",
    "device": "cpu",
    "rhenium_version": "1.0.0"
  },
  "summary": {
    "total": 576,
    "passed": 570,
    "failed": 0,
    "skipped": 6
  }
}
```

## Extending the Matrix

To add new test parameters:

1. Edit `tests/bench/conftest.py`
2. Add new values to parameter lists
3. Update `@pytest.mark.parametrize` decorators

Example adding a new modality:
```python
# In conftest.py
MODALITIES = ["MRI", "CT", "US", "XR", "PET"]  # Added PET
```

## Related Documentation

- [Benchmark Plan](benchmark-plan.md) - Detailed category definitions
- [Performance Methodology](performance-methodology.md) - Measurement methodology
- [Core Model Architecture](../architecture/core-model.md) - Model documentation
