# Rhenium OS Benchmark Plan

This document defines the benchmark strategy and acceptance criteria for the Rhenium OS platform.

> **IMPORTANT**: All benchmarks use synthetic data generated within the repository. No PHI/PII or patient data is used. Results are for R&D purposes only.

## Benchmark Categories

### 1. Functional Correctness

**Objective**: Verify outputs conform to expected schemas and invariants.

| Test | Criterion | Threshold |
|------|-----------|-----------|
| Output schema validation | JSON schema match | 100% |
| Shape invariants | Output shape matches input | 100% |
| Dtype preservation | float32 in -> float32 out | 100% |
| Segmentation mask values | Integer values only | 100% |
| Evidence dossier keys | Required keys present | 100% |

---

### 2. Numerical Stability

**Objective**: Ensure no NaN/Inf values and bounded outputs.

| Test | Criterion | Threshold |
|------|-----------|-----------|
| NaN detection | No NaN in outputs | 0 NaN |
| Inf detection | No Inf in outputs | 0 Inf |
| Range preservation | Normalized inputs -> [0,1] outputs | 100% |
| Gradient stability | No exploding gradients | max 1e6 |

---

### 3. Determinism/Reproducibility

**Objective**: Same seed produces identical outputs.

| Test | Criterion | Threshold |
|------|-----------|-----------|
| Same seed, same output | Bit-exact match | 100% |
| Cross-run consistency | Multiple runs identical | 100% |
| Documented nondeterminism | Listed exceptions | N/A |

---

### 4. Performance Throughput/Latency/Memory

**Objective**: Meet performance envelope constraints.

| Metric | CPU Target | GPU Target |
|--------|------------|------------|
| Smoke test runtime | < 1.0s per case | < 0.5s per case |
| Full volume (64^3) | < 5.0s | < 1.0s |
| Startup overhead | < 10s | < 15s |
| Peak memory | < 4GB | < 8GB |
| p50 latency | < 500ms | < 100ms |
| p95 latency | < 2000ms | < 500ms |

---

### 5. Robustness

**Objective**: Model handles degraded inputs gracefully.

| Test | Perturbation | Criterion |
|------|--------------|-----------|
| Gaussian noise | sigma=0.1, 0.2, 0.3 | No crash |
| Blur | kernel=3, 5, 7 | No crash |
| Intensity shift | +/- 20% | No crash |
| Missing slices | 10% random | No crash |
| Anisotropic spacing | 1:1:3 ratio | No crash |
| Orientation flip | LR, AP, SI | No crash |

---

### 6. Calibration/Uncertainty

**Objective**: Sanity check confidence scores.

| Metric | Threshold |
|--------|-----------|
| ECE (Expected Calibration Error) | Report only |
| Brier score | Report only |
| NLL (Negative Log Likelihood) | Report only |
| Confidence distribution | Documented |

*Note: These are sanity checks on synthetic data, not clinical validation.*

---

### 7. Fairness Scaffolding

**Objective**: Verify stratified metrics infrastructure.

| Test | Criterion |
|------|-----------|
| Subgroup label support | Metadata accepts subgroup fields |
| Stratified metric computation | Metrics computed per subgroup |
| Reporting template | Outputs subgroup breakdown |

*Note: Uses synthetic subgroup labels for testing infrastructure only.*

---

### 8. Privacy/Security

**Objective**: Ensure no PHI leakage.

| Test | Criterion |
|------|-----------|
| No PHI in outputs | Scan for PII patterns | 0 matches |
| De-identification hooks | Hook functions callable | 100% |
| Log scrubbing | Logs contain no patient IDs | 100% |
| Output sanitization | Paths sanitized | 100% |

---

### 9. API Readiness

**Objective**: FastAPI endpoints behave correctly.

| Test | Criterion |
|------|-----------|
| /health endpoint | Returns 200 | 100% |
| Schema validation | OpenAPI schema valid | 100% |
| Error responses | Stable error format | 100% |
| Pipeline endpoint | Returns structured result | 100% |

---

### 10. CLI Readiness

**Objective**: CLI commands work correctly.

| Test | Criterion |
|------|-----------|
| Exit codes | 0 on success, non-zero on error | 100% |
| Artifact creation | Files created in expected paths | 100% |
| Help text | All commands have --help | 100% |
| Synthetic ingest | Ingest synthetic data | Success |

---

### 11. Governance Readiness

**Objective**: Governance artifact generation works.

| Test | Criterion |
|------|-----------|
| Model card generation | Valid Markdown + JSON | 100% |
| Dataset card template | Renders correctly | 100% |
| Risk register template | Contains required sections | 100% |
| Incident template | Contains required fields | 100% |

---

### 12. Generative Disclosure Compliance

**Objective**: All AI-generated content is properly disclosed.

| Test | Criterion |
|------|-----------|
| SR output disclosure | `generation_metadata.disclosure` present | 100% |
| Denoise output disclosure | `generation_metadata.disclosure` present | 100% |
| GAN output disclosure | `generation_metadata.disclosure` present | 100% |
| Disclosure text | Contains "AI" or "generated" | 100% |

---

## Test Matrix Parameters

The benchmark suite uses parametrized tests to generate 500+ test cases:

```
Modalities: [MRI, CT, US, XR] = 4
Tasks: [segmentation, classification, detection, reconstruction, super_resolution, denoise] = 6
Shapes: [(16,32,32), (32,64,64), (64,128,128), (1,64,64), (1,128,128), (1,256,256)] = 6
Dtypes: [float32, float16] = 2
Devices: [cpu, cuda] = 2

Total: 4 × 6 × 6 × 2 × 2 = 576 test cases
```

## Smoke vs Full Benchmark

| Mode | Test Count | Runtime Target | CI Integration |
|------|------------|----------------|----------------|
| Smoke (`-m smoke`) | ~50 | < 120s CPU | Every PR/push |
| Full (`-m bench`) | 576+ | < 30min | Nightly optional |

## Running Benchmarks

```bash
# Smoke tests (CI)
pytest tests/bench/ -m smoke -v

# Full matrix (nightly)
python scripts/run_benchmarks.py --full --json

# Specific category
pytest tests/bench/test_numerical_stability.py -v

# With performance metrics
pytest tests/bench/ -m bench --benchmark-enable
```

## Report Format

Benchmark results are output to `artifacts/benchmarks/report.json`:

```json
{
  "metadata": {
    "timestamp": "2025-01-01T00:00:00",
    "device": "cpu",
    "python_version": "3.11",
    "rhenium_version": "1.0.0"
  },
  "summary": {
    "total_tests": 576,
    "passed": 570,
    "failed": 0,
    "skipped": 6,
    "duration_seconds": 1200
  },
  "categories": {
    "functional": {"passed": 50, "failed": 0},
    "numerical": {"passed": 48, "failed": 0},
    ...
  },
  "details": [...]
}
```
