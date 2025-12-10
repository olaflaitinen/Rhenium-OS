# CLI Reference

## Overview

Rhenium OS provides a command-line interface for data ingestion, pipeline execution, and evaluation.

```bash
rhenium --help
```

## Commands

### ingest

Ingest DICOM or raw acquisition data.

```bash
# Ingest DICOM directory
rhenium ingest dicom /path/to/dicom --output ./data

# Ingest k-space data
rhenium ingest raw /path/to/kspace.h5 --data-type kspace
```

### run-pipeline

Execute analysis pipelines.

```bash
# Run knee MRI pipeline
rhenium run-pipeline run mri_knee_default --input ./data/study --output ./results

# List available configurations
rhenium run-pipeline list-configs
```

### evaluate

Run evaluation benchmarks.

```bash
# Run benchmark suite
rhenium evaluate run knee_mri_benchmark --data ./test_data --output ./eval

# Compute metrics
rhenium evaluate metrics predictions.nii.gz ground_truth.nii.gz --task segmentation
```

### explain

Generate explanations and evidence dossiers.

```bash
# Generate explanations for results
rhenium explain generate ./results --output ./explanations

# Display evidence dossier
rhenium explain dossier finding_abc123
```

### inspect

Inspect registry and configuration.

```bash
# List registered components
rhenium inspect registry

# Show current configuration
rhenium inspect config

# Show version
rhenium inspect version
```
