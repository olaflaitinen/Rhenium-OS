# Performance Card Template

A Performance Card provides a standardized summary of benchmark results for a single run.

---

## Run Information

| Field | Value |
|-------|-------|
| **Timestamp** | `{{timestamp}}` |
| **Version** | `{{version}}` |
| **Git SHA** | `{{git_sha}}` |
| **Device** | `{{device}}` |

---

## Environment

| Component | Value |
|-----------|-------|
| Python | `{{python}}` |
| PyTorch | `{{torch}}` |
| CUDA | `{{cuda}}` |
| OS | `{{os}}` |
| CPU | `{{cpu}}` |
| GPU | `{{gpu}}` |
| RAM | `{{ram_gb}}` GB |
| VRAM | `{{vram_gb}}` GB |

---

## Summary Metrics

### Latency (End-to-End)

| Percentile | Value |
|------------|-------|
| p50 | `{{e2e_p50}}` ms |
| p95 | `{{e2e_p95}}` ms |
| p99 | `{{e2e_p99}}` ms |

### Throughput

| Metric | Value |
|--------|-------|
| Items/sec | `{{throughput_items}}` |
| Voxels/sec | `{{throughput_voxels}}` |

### Memory

| Metric | Value |
|--------|-------|
| Peak RSS | `{{peak_rss}}` MB |
| Peak VRAM | `{{peak_vram}}` MB |

---

## Benchmark Results

| Category | Tests | Passed | Failed | p95 Latency |
|----------|-------|--------|--------|-------------|
| End-to-end | `{{e2e_total}}` | `{{e2e_passed}}` | `{{e2e_failed}}` | `{{e2e_p95}}` ms |
| I/O & Parsing | `{{io_total}}` | `{{io_passed}}` | `{{io_failed}}` | `{{io_p95}}` ms |
| Preprocessing | `{{prep_total}}` | `{{prep_passed}}` | `{{prep_failed}}` | `{{prep_p95}}` ms |
| Perception | `{{perc_total}}` | `{{perc_passed}}` | `{{perc_failed}}` | `{{perc_p95}}` ms |
| Reconstruction | `{{recon_total}}` | `{{recon_passed}}` | `{{recon_failed}}` | `{{recon_p95}}` ms |
| GAN Inference | `{{gan_total}}` | `{{gan_passed}}` | `{{gan_failed}}` | `{{gan_p95}}` ms |
| XAI Dossier | `{{xai_total}}` | `{{xai_passed}}` | `{{xai_failed}}` | `{{xai_p95}}` ms |
| Serialization | `{{ser_total}}` | `{{ser_passed}}` | `{{ser_failed}}` | `{{ser_p95}}` ms |
| API/CLI | `{{api_total}}` | `{{api_passed}}` | `{{api_failed}}` | `{{api_p95}}` ms |
| Governance | `{{gov_total}}` | `{{gov_passed}}` | `{{gov_failed}}` | `{{gov_p95}}` ms |

---

## Regression Status

{{#if has_regressions}}
> [!WARNING]
> **{{regression_count}} regressions detected** compared to baseline.

| Benchmark | Metric | Baseline | Current | Change |
|-----------|--------|----------|---------|--------|
{{#each regressions}}
| `{{benchmark_id}}` | {{metric}} | {{baseline_value}} | {{current_value}} | {{change_percent}}% |
{{/each}}
{{else}}
> [!NOTE]
> No regressions detected.
{{/if}}

---

## Disclaimer

> [!CAUTION]
> This is a research and development system. Performance metrics are for internal benchmarking only and do not represent clinical performance claims.
