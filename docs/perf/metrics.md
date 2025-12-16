# Performance Metrics Reference

This document defines the performance metrics collected by the Rhenium OS benchmark suite.

## Latency Metrics

### Definition

Latency measures the time to complete a single operation from start to finish.

### Measurement Methodology

1. **Warmup Phase**: `N_warmup` iterations discarded (default: 3)
2. **Measurement Phase**: `N_measure` timed iterations (default: 10)
3. **Timing**: Wall-clock time via `time.perf_counter()`

### Percentile Calculations

Given sorted latency measurements $L = [l_1, l_2, ..., l_n]$:

$$p_k = L[\lceil k \cdot n / 100 \rceil]$$

Where:
- $p_{50}$ (median): 50th percentile
- $p_{95}$: 95th percentile
- $p_{99}$: 99th percentile

### Statistics

| Metric | Formula |
|--------|---------|
| Mean | $\bar{L} = \frac{1}{n}\sum_{i=1}^{n} l_i$ |
| Std Dev | $\sigma = \sqrt{\frac{1}{n-1}\sum_{i=1}^{n}(l_i - \bar{L})^2}$ |
| Min | $\min(L)$ |
| Max | $\max(L)$ |

---

## Throughput Metrics

### Items Per Second

$$\text{items\_per\_sec} = \frac{n}{\sum_{i=1}^{n} l_i}$$

Where $n$ is the number of items processed and $l_i$ is the latency for each.

### Voxels Per Second (3D volumes)

$$\text{voxels\_per\_sec} = \frac{D \times H \times W}{\text{latency\_sec}}$$

Where $D$, $H$, $W$ are the depth, height, width of the volume.

---

## Memory Metrics

### CPU Memory (RSS)

Resident Set Size (RSS) measured via `psutil`:

```python
import psutil
process = psutil.Process()
rss_mb = process.memory_info().rss / (1024 * 1024)
```

| Metric | Description |
|--------|-------------|
| `rss_mb_before` | RSS before operation |
| `rss_mb_after` | RSS after operation |
| `rss_mb_peak` | Maximum RSS during operation |

### GPU Memory (VRAM)

VRAM measured via PyTorch CUDA APIs (optional `pynvml`):

```python
import torch
vram_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
```

| Metric | Description |
|--------|-------------|
| `vram_mb_before` | VRAM before operation |
| `vram_mb_after` | VRAM after operation |
| `vram_mb_peak` | Peak VRAM during operation |

---

## Regression Detection

### Threshold Calculation

A regression is flagged when:

$$\frac{\text{current} - \text{baseline}}{\text{baseline}} > \tau$$

Where $\tau$ is the regression threshold (default: 10% for latency, 20% for memory).

### Metrics Compared

| Metric | Threshold | Direction |
|--------|-----------|-----------|
| `latency_ms.p95` | 10% | Higher is worse |
| `memory.rss_mb_peak` | 20% | Higher is worse |
| `throughput.items_per_sec` | -10% | Lower is worse |

---

## Benchmark Categories

| Category | ID | Description |
|----------|-----|-------------|
| End-to-end | `e2e_core_model` | Full pipeline latency |
| I/O & Parsing | `io_parsing` | Data loading overhead |
| Preprocessing | `preprocessing` | Normalization, resampling |
| Perception | `perception_inference` | Segmentation forward pass |
| MRI Reconstruction | `reconstruction_mri` | K-space to image |
| CT Reconstruction | `reconstruction_ct` | Sinogram to image |
| PINN Step | `pinn_step` | Physics-informed step |
| GAN Inference | `gan_inference` | Generator forward pass |
| XAI Dossier | `xai_dossier` | Evidence generation |
| Serialization | `serialization` | Output writing |
| FastAPI | `fastapi_readiness` | API overhead |
| CLI | `cli_overhead` | Command-line overhead |
| Governance | `governance_artifacts` | Card generation |

---

## Environment Capture

The benchmark harness captures:

```json
{
  "python": "3.11.5",
  "torch": "2.1.0",
  "cuda": "12.1",
  "os": "Linux",
  "cpu": "AMD EPYC 7763",
  "cpu_count": 64,
  "gpu": "NVIDIA A100",
  "ram_gb": 512,
  "vram_gb": 80
}
```

---

## Reproducibility

For reproducible benchmarks:

1. Fixed random seed: `torch.manual_seed(42)`
2. Deterministic mode: `torch.use_deterministic_algorithms(True)`
3. Single-threaded: `torch.set_num_threads(1)`
4. GPU sync: `torch.cuda.synchronize()` before timing
