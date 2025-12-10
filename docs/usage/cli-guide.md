# CLI Guide: Rhenium OS

**
---

## Overview

The Rhenium OS command-line interface provides access to all core functionality including data ingestion, pipeline execution, evaluation, and explanation generation.

## Installation

After installing Rhenium OS, the CLI is available as `rhenium`:

```bash
rhenium --help
```

---

## Global Options

```bash
rhenium [OPTIONS] COMMAND [ARGS]

Options:
  --version, -v    Show version
  --help           Show help message
```

---

## Commands

### ingest

Ingest medical imaging data into Rhenium OS.

#### ingest dicom

```bash
rhenium ingest dicom /path/to/dicom --output ./data --recursive

Arguments:
  SOURCE    Path to DICOM directory

Options:
  --output, -o    Output directory [default: ./data]
  --recursive     Search recursively [default: true]
```

#### ingest raw

```bash
rhenium ingest raw /path/to/kspace.h5 --data-type kspace --output ./data

Arguments:
  SOURCE    Path to raw data file

Options:
  --data-type     Data type: kspace, sinogram [default: kspace]
  --output, -o    Output directory [default: ./data]
```

---

### run-pipeline

Execute analysis pipelines.

#### run-pipeline run

```bash
rhenium run-pipeline run CONFIG --input /path/to/data --output ./results

Arguments:
  CONFIG    Pipeline configuration name

Options:
  --input, -i     Input data path (required)
  --output, -o    Output directory [default: ./results]
  --batch         Process multiple studies
  --generate-xai  Generate XAI dossiers [default: true]
```

#### run-pipeline list-configs

```bash
rhenium run-pipeline list-configs
```

Lists all available pipeline configurations.

---

### evaluate

Run evaluation and benchmarks.

#### evaluate run

```bash
rhenium evaluate run SUITE --data /path/to/test_data --output ./evaluation

Arguments:
  SUITE    Benchmark suite name

Options:
  --data, -d      Test data directory (required)
  --output, -o    Results output directory [default: ./evaluation]
  --fairness      Include fairness evaluation [default: false]
```

#### evaluate metrics

```bash
rhenium evaluate metrics PREDICTIONS GROUND_TRUTH --task segmentation

Arguments:
  PREDICTIONS     Predictions file
  GROUND_TRUTH    Ground truth file

Options:
  --task          Task type: segmentation, detection, classification
```

---

### explain

Generate explanations and evidence dossiers.

#### explain generate

```bash
rhenium explain generate /path/to/results --output ./explanations

Arguments:
  RESULTS    Path to pipeline results

Options:
  --output        Output directory [default: ./explanations]
  --format        Output format: json, html [default: json]
```

#### explain dossier

```bash
rhenium explain dossier FINDING_ID --results-dir ./results

Arguments:
  FINDING_ID    Finding identifier

Options:
  --results-dir    Results directory [default: ./results]
```

---

### inspect

Inspect registry, configuration, and models.

#### inspect registry

```bash
rhenium inspect registry
```

Lists all registered components (pipelines, models, organ modules).

#### inspect config

```bash
rhenium inspect config
```

Shows current configuration settings.

#### inspect version

```bash
rhenium inspect version
```

Shows version information.

---

### model-card

Generate or view model cards.

```bash
rhenium model-card show MODEL_NAME

Arguments:
  MODEL_NAME    Name of registered model

Options:
  --format      Output format: markdown, json [default: markdown]
  --output      Save to file instead of stdout
```

---

### fairness

Run fairness evaluation.

```bash
rhenium fairness evaluate --predictions pred.json --ground-truth gt.json --demographics demo.json

Options:
  --predictions     Predictions file (required)
  --ground-truth    Ground truth file (required)
  --demographics    Demographics file (required)
  --output          Output report path
  --format          Report format: markdown, json [default: markdown]
```

---

## Examples

### Complete Workflow

```bash
# 1. Ingest DICOM study
rhenium ingest dicom /data/patient001 --output ./ingested

# 2. Run knee MRI pipeline
rhenium run-pipeline run mri_knee_default \
    --input ./ingested/study_001 \
    --output ./results/study_001

# 3. Generate explanations
rhenium explain generate ./results/study_001 \
    --output ./explanations/study_001

# 4. Run evaluation on test set
rhenium evaluate run knee_mri_benchmark \
    --data ./test_data \
    --output ./evaluation

# 5. Generate fairness report
rhenium fairness evaluate \
    --predictions ./results/predictions.json \
    --ground-truth ./test_data/labels.json \
    --demographics ./test_data/demographics.json \
    --output ./fairness_report.md
```

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `RHENIUM_DATA_DIR` | Default data directory |
| `RHENIUM_MODELS_DIR` | Model weights directory |
| `RHENIUM_LOGS_DIR` | Log files directory |
| `RHENIUM_DEVICE` | Compute device (auto/cpu/cuda) |
| `RHENIUM_MEDGEMMA_BACKEND` | MedGemma backend (stub/local/remote) |

---

**Copyright (c) 2025 Skolyn LLC. All rights reserved.**

****
