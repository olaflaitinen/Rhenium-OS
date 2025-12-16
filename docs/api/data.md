# Data API Reference

## ImageVolume

The core data structure for 3D medical images.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `array` | `np.ndarray` | Voxel data (D, H, W) or (D, H, W, C) |
| `spacing` | `tuple[float, ...]` | Voxel spacing in mm |
| `origin` | `tuple[float, ...]` | World coordinates of first voxel |
| `orientation` | `np.ndarray` | 3×3 direction cosine matrix |
| `modality` | `Modality` | Imaging modality enum |

### Modality Enum

```python
class Modality(str, Enum):
    MRI = "MRI"
    CT = "CT"
    US = "US"      # Ultrasound
    XRAY = "XRAY"
    PET = "PET"
    SPECT = "SPECT"
```

### Methods

```python
# Convert to PyTorch tensor
tensor = volume.to_tensor(device="cuda")  # Shape: (1, 1, D, H, W)

# Normalize
normalized = volume.normalize(method="zscore")

# Apply CT windowing
windowed = volume.apply_window(center=40, width=400)

# Resample to new spacing
resampled = volume.resample(target_spacing=(1.0, 1.0, 1.0))
```

---

## Data I/O

### DICOM

```python
from rhenium.data.dicom import load_dicom_directory, DICOMStudy

study: DICOMStudy = load_dicom_directory(
    "./dicom_folder",
    deidentify=True  # Remove PHI
)

for series in study.series:
    volume = series.to_volume()
```

### NIfTI

```python
from rhenium.data.nifti import load_nifti, save_nifti

volume = load_nifti("brain.nii.gz")
save_nifti(volume, "output.nii.gz")
```

---

## Preprocessing Pipeline

```mermaid
graph LR
    Input[Volume] --> Resample
    Resample --> Normalize
    Normalize --> CropOrPad
    CropOrPad --> Output[Processed]
```

### Built-in Pipelines

```python
from rhenium.data.preprocessing import PreprocessingPipeline

# For MRI
mri_pipeline = PreprocessingPipeline.for_mri(
    target_spacing=(1.0, 1.0, 1.0)
)

# For CT  
ct_pipeline = PreprocessingPipeline.for_ct(
    target_spacing=(1.0, 1.0, 3.0),
    window_preset="soft_tissue"
)
```
