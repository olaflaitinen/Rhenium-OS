# Reconstruction Guide

## Overview

Rhenium OS provides reconstruction methods for MRI, CT, Ultrasound, and X-ray.

## MRI Reconstruction

### Forward Model

The MRI acquisition can be modeled as:

\[
\mathbf{y} = \mathcal{P}_\Omega \mathcal{F} \mathbf{x} + \boldsymbol{\eta}
\]

Where \(\mathcal{P}_\Omega\) is the undersampling operator, \(\mathcal{F}\) is the Fourier transform.

### Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| Zero-filled | IFFT of undersampled data | Baseline/preview |
| Unrolled | Learned iterative network | High quality |
| VarNet | Variational network | Multi-coil data |

### Usage

```python
from rhenium.reconstruction.mri import UnrolledMRIRecon

model = UnrolledMRIRecon(
    num_cascades=10,
    features=64,
)

reconstructed = model(kspace, mask)
```

---

## CT Reconstruction

### Filtered Backprojection

```python
from rhenium.reconstruction.ct import FBPReconstructor
import torch

fbp = FBPReconstructor(filter_type="shepp-logan")

angles = torch.linspace(0, torch.pi, 180)
image = fbp(sinogram, angles)
```

### Learned Reconstruction

```python
from rhenium.reconstruction.ct import CTReconstructor

model = CTReconstructor(features=64, num_blocks=8)
image = model(sinogram, angles)
```

---

## Physics-Informed Neural Networks

Combine data fidelity with physics constraints:

\[
\mathcal{L} = \underbrace{\|\hat{x} - x_{gt}\|_2^2}_{\text{data}} + \lambda \underbrace{\|A\hat{x} - y\|_2^2}_{\text{physics}}
\]

```python
from rhenium.reconstruction.pinns import MRIPINNLoss

loss_fn = MRIPINNLoss(lambda_dc=1.0, lambda_smooth=0.01)

physics_loss = loss_fn(prediction, kspace, mask)
```

---

## Ultrasound Beamforming

```python
from rhenium.reconstruction.ultrasound import USReconstructor

beamformer = USReconstructor(
    speed_of_sound=1540.0,
    sampling_freq=40e6,
)

image = beamformer(channel_data)
```

---

## X-ray Enhancement

```python
from rhenium.reconstruction.xray import XRayEnhancer, BoneSuppressor

# Enhance image quality
enhancer = XRayEnhancer(features=64)
enhanced = enhancer(xray_image)

# Suppress bone structures
suppressor = BoneSuppressor()
soft_tissue = suppressor(chest_xray)
```
