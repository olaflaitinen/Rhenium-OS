# Quick Start

## Basic Usage

### Load and Preprocess Data

```python
from rhenium.data.nifti import load_nifti
from rhenium.data.preprocessing import PreprocessingPipeline

# Load volume
volume = load_nifti("brain_mri.nii.gz")
print(f"Shape: {volume.shape}, Spacing: {volume.spacing}")

# Preprocess
pipeline = PreprocessingPipeline.for_mri(
    target_spacing=(1.0, 1.0, 1.0)
)
processed = pipeline(volume)
```

### Run Segmentation

```python
from rhenium.perception.segmentation import UNet3D, load_segmentation_model
import torch

# Load model
model = UNet3D(in_channels=1, out_channels=3)

# Inference
tensor = processed.to_tensor(device="cuda")
with torch.no_grad():
    output = model(tensor)
    mask = output.argmax(dim=1)
```

### Generate Evidence Dossier

```python
from rhenium.xai import EvidenceDossier, Finding, VisualEvidence
from rhenium.xai.measurements import MeasurementExtractor
import numpy as np

# Extract measurements
extractor = MeasurementExtractor()
volume_ev = extractor.extract_volume(mask.cpu().numpy(), volume.spacing)

# Create finding
finding = Finding(
    finding_id="f001",
    finding_type="lesion",
    description="Hyperintense lesion in left temporal lobe",
    confidence=0.85,
    quantitative_evidence=[volume_ev],
)

# Create dossier
dossier = EvidenceDossier(
    dossier_id="d001",
    finding=finding,
    study_uid="1.2.3.4.5",
    series_uid="1.2.3.4.5.1",
    pipeline_name="brain_lesion_seg",
    pipeline_version="1.0.0",
)
dossier.save(Path("./output"))
```

## CLI Usage

```bash
# Generate synthetic test data
rhenium synthetic --output ./data/synthetic --num 5

# Run benchmark
rhenium benchmark

# Start API server
rhenium serve --port 8000
```

## Configuration

Set via environment variables:

```bash
export RHENIUM_DEVICE=cuda
export RHENIUM_SEED=42
export RHENIUM_LOG_LEVEL=INFO
```

Or in code:

```python
from rhenium.core.config import get_settings

settings = get_settings()
settings.data_dir = Path("./my_data")
```
