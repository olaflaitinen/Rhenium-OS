# Performance Dashboard

> Generated: 2025-12-15T15:52:19.568807

## Environment

| Component | Value |
|-----------|-------|
| Python | 3.14.1 |
| PyTorch | 2.9.1+cpu |
| CUDA | None |
| OS | Windows 11 |
| CPU | AMD64 Family 23 Model 24 Stepping 1, AuthenticAMD |
| CPU Cores | 8 |
| GPU | None |
| RAM | 5.9 GB |
| VRAM | None GB |

## Model

| Field | Value |
|-------|-------|
| Version | 0.1.0 |
| Core Model | RheniumCoreModel |
| Git SHA | 75f0cbfc7feb |
| Git Branch | main |
| Config Hash | 9716c68902d06de9 |

---

## Performance Summary

| Metric | Value |
|--------|-------|
| Total Benchmarks | 49 |
| Passed | 49 |
| Failed | 0 |
| Skipped | 0 |

### Top-Line Metrics

| Metric | Value |
|--------|-------|
| E2E Latency p50 | 3.226 ms |
| E2E Latency p95 | 3.582 ms |
| E2E Throughput | 315.9 items/sec |
| Peak Memory (RSS) | 338.83 MB |
| Peak Memory (VRAM) | None MB |

---

## Benchmark Results by Category

### Cli Overhead

| Tests | Passed | Failed | Avg p95 |
|-------|--------|--------|---------|
| 1 | 1 | 0 | 2.49 ms |

### E2E Core Model

| Tests | Passed | Failed | Avg p95 |
|-------|--------|--------|---------|
| 8 | 8 | 0 | 3.58 ms |

### Fastapi Readiness

| Tests | Passed | Failed | Avg p95 |
|-------|--------|--------|---------|
| 1 | 1 | 0 | 0.01 ms |

### Gan Inference

| Tests | Passed | Failed | Avg p95 |
|-------|--------|--------|---------|
| 5 | 5 | 0 | 10.6 ms |

### Governance Artifacts

| Tests | Passed | Failed | Avg p95 |
|-------|--------|--------|---------|
| 2 | 2 | 0 | 0.02 ms |

### Io Parsing

| Tests | Passed | Failed | Avg p95 |
|-------|--------|--------|---------|
| 4 | 4 | 0 | 0.29 ms |

### Perception Inference

| Tests | Passed | Failed | Avg p95 |
|-------|--------|--------|---------|
| 10 | 10 | 0 | 3.51 ms |

### Pinn Step

| Tests | Passed | Failed | Avg p95 |
|-------|--------|--------|---------|
| 1 | 1 | 0 | 0.06 ms |

### Preprocessing

| Tests | Passed | Failed | Avg p95 |
|-------|--------|--------|---------|
| 4 | 4 | 0 | 0.31 ms |

### Reconstruction Ct

| Tests | Passed | Failed | Avg p95 |
|-------|--------|--------|---------|
| 2 | 2 | 0 | 3.42 ms |

### Reconstruction Mri

| Tests | Passed | Failed | Avg p95 |
|-------|--------|--------|---------|
| 2 | 2 | 0 | 3.3 ms |

### Scalability

| Tests | Passed | Failed | Avg p95 |
|-------|--------|--------|---------|
| 3 | 3 | 0 | 12.82 ms |

### Serialization

| Tests | Passed | Failed | Avg p95 |
|-------|--------|--------|---------|
| 2 | 2 | 0 | 0.04 ms |

### Stress Concurrency

| Tests | Passed | Failed | Avg p95 |
|-------|--------|--------|---------|
| 1 | 1 | 0 | 0.0 ms |

### Stress Stability

| Tests | Passed | Failed | Avg p95 |
|-------|--------|--------|---------|
| 1 | 1 | 0 | 0.0 ms |

### Xai Dossier

| Tests | Passed | Failed | Avg p95 |
|-------|--------|--------|---------|
| 2 | 2 | 0 | 12.76 ms |

---

## Capability/Performance Matrix

| Task | CT | MRI | N/A | US | XR |
|------|------|------|------|------|------|
| classification | - | 3.8 ms | - | - | - |
| ct_reconstruction | 3.4 ms | - | - | - | - |
| denoise | 4.0 ms | - | - | - | - |
| detection | 3.7 ms | - | - | - | - |
| dossier_generation | 3.5 ms | 22.0 ms | - | - | - |
| full_pipeline | 2.9 ms | 3.4 ms | - | 4.1 ms | 3.9 ms |
| health_check | - | - | 0.0 ms | - | - |
| json_export | - | - | 0.0 ms | - | - |
| model_card | - | - | 0.0 ms | - | - |
| module_import | - | - | 2.5 ms | - | - |
| mri_reconstruction | - | 3.3 ms | - | - | - |
| normalize_minmax | - | - | 0.1 ms | - | - |
| pde_residual | - | - | 0.1 ms | - | - |
| resample | - | - | 1.0 ms | - | - |
| risk_register | - | - | 0.0 ms | - | - |
| segmentation | 3.2 ms | 3.3 ms | 12.8 ms | 3.3 ms | 4.0 ms |
| segmentation_4threads | - | 0.0 ms | - | - | - |
| segmentation_soak | - | 0.0 ms | - | - | - |
| super_resolution | 12.4 ms | 12.1 ms | - | - | - |
| synthetic_volume | - | - | 1.0 ms | - | - |
| volume_creation | - | - | 0.1 ms | - | - |

---

*This dashboard is auto-generated. Do not edit manually.*
