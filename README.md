# Rhenium OS

**Multi-Modality AI Platform for Medical Imaging Research**

Rhenium OS is a comprehensive platform for medical imaging AI research, supporting reconstruction, perception, and generative tasks across MRI, CT, Ultrasound, and X-ray modalities.

---

## Features

- **Multi-Modality Support**: MRI, CT, Ultrasound, X-ray
- **Reconstruction**: Classical and deep learning methods, Physics-Informed Neural Networks (PINNs)
- **Perception**: Segmentation, detection, classification with state-of-the-art architectures
- **Generative Models**: Pix2Pix, CycleGAN, SRGAN for image translation and super-resolution
- **Explainability**: Evidence Dossier framework for transparent AI outputs
- **Governance**: Model cards, dataset cards, audit logging, fairness evaluation
- **Backend Integration**: FastAPI server with embeddable components

---

## Installation

```bash
# Basic installation
pip install rhenium-os

# Development installation
pip install -e ".[dev]"

# Full installation with all extras
pip install -e ".[all]"
```

---

## Quick Start

```python
from rhenium.core.config import get_settings
from rhenium.testing.synthetic import SyntheticDataGenerator

# Generate synthetic data
generator = SyntheticDataGenerator(seed=42)
volume = generator.generate_volume(shape=(64, 64, 32), modality="MRI")

# Run segmentation
from rhenium.perception.segmentation import UNet3D
model = UNet3D(in_channels=1, out_channels=2)
# ... inference code
```

### CLI Usage

```bash
# Show version
rhenium version

# Ingest synthetic data
rhenium ingest synthetic --output-dir ./data

# Run pipeline
rhenium pipeline run segmentation --input ./data/study_001

# Generate explanation
rhenium explain generate --job-id abc123
```

---

## Architecture

```
src/rhenium/
├── core/           # Configuration, registry, logging
├── data/           # DICOM, NIfTI, volume handling
├── reconstruction/ # MRI, CT, US, X-ray reconstruction + PINNs
├── perception/     # Segmentation, detection, classification
├── generative/     # GANs, super-resolution, denoising
├── xai/            # Evidence Dossier, explanations
├── medgemma/       # Vision-language model integration
├── pipelines/      # Orchestration and runners
├── cli/            # Command-line interface
├── evaluation/     # Metrics and benchmarks
├── governance/     # Model cards, audit, fairness
├── server/         # FastAPI backend
└── testing/        # Synthetic data generation
```

---

## Supported Modalities

| Modality | Reconstruction | Segmentation | Detection | GANs |
|----------|---------------|--------------|-----------|------|
| MRI | k-space, PINN | 3D U-Net, UNETR | CenterNet | Pix2Pix, SR |
| CT | FBP, PINN | 3D U-Net | CenterNet | Denoising |
| Ultrasound | Beamforming | 2D/3D | Detection | Enhancement |
| X-ray | Enhancement | 2D | Detection | Super-res |

---

## Backend Integration

The platform can be embedded in external systems:

```python
from rhenium.server.app import app
from rhenium.pipelines.runner import PipelineRunner

# Option 1: Mount FastAPI app
external_app.mount("/rhenium", app)

# Option 2: Direct pipeline access
runner = PipelineRunner()
result = runner.run(
    pipeline_name="brain_lesion_seg",
    study_uid="1.2.3.4.5",
)
```

---

## Development

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check src/
black --check src/

# Run type checking
mypy src/rhenium

# Start development server
uvicorn rhenium.server.app:app --reload
```

---

## Disclaimer

> **IMPORTANT**: Rhenium OS is intended for **research and development purposes only**.
> It has NOT been cleared or approved by any regulatory authority for clinical use.
> All AI-generated findings require verification by qualified medical professionals.
> This system does not provide medical advice, diagnosis, or treatment recommendations.

---

## Regulatory Alignment

This project is designed with awareness of regulatory frameworks:
- **EU MDR**: Alignment targets for medical device requirements
- **EU AI Act**: Alignment targets for high-risk AI system requirements

Note: "Alignment targets" indicate design considerations, not compliance certifications.

---

## License

This project is licensed under the **European Union Public License 1.1 (EUPL-1.1)**.
See [LICENSE](LICENSE) for details.

---

## Citation

```bibtex
@software{rhenium_os,
  title = {Rhenium OS: Multi-Modality AI Platform for Medical Imaging Research},
  year = {2025},
  url = {https://github.com/rhenium-os/rhenium-os}
}
```
