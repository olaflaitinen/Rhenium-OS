# Core Architecture: Rhenium OS

**Last Updated: December 2025**

---

## Package Structure

Rhenium OS is organized as a Python package with clearly defined modules:

```
rhenium/
├── __init__.py              # Package root, version, exports
├── core/                    # Foundation infrastructure
│   ├── config.py            # Pydantic settings
│   ├── logging.py           # Structured logging
│   ├── errors.py            # Error hierarchy
│   └── registry.py          # Component registry
├── data/                    # Data I/O and preprocessing
│   ├── dicom_io.py          # DICOM handling
│   ├── nifti_io.py          # NIfTI support
│   ├── raw_io.py            # Raw data (k-space, sinogram)
│   └── preprocessing.py     # Normalization, resampling
├── reconstruction/          # Rhenium Reconstruction Engine
│   ├── mri/                 # MRI-specific reconstruction
│   ├── ct/                  # CT reconstruction
│   ├── xray/                # X-ray enhancement
│   ├── us/                  # Ultrasound beamforming
│   └── pinns/               # Physics-informed models
├── perception/              # Rhenium Perception Engine
│   ├── segmentation/        # Semantic segmentation
│   ├── detection/           # Object detection
│   ├── classification/      # Classification models
│   └── organ/               # Organ-specific modules
├── generative/              # Generative models
│   ├── super_resolution.py  # Image upscaling
│   ├── denoising.py         # Noise reduction
│   └── anomaly_detection.py # Anomaly detection
├── xai/                     # Rhenium XAI Engine
│   ├── explanation_schema.py # Core data structures
│   ├── evidence_dossier.py  # Dossier container
│   ├── visual_explanations.py
│   └── quantitative_explanations.py
├── medgemma/                # MedGemma Integration
│   ├── adapter.py           # Client abstraction
│   ├── prompts/             # Prompt templates
│   ├── tools.py             # Tool-use framework
│   └── validators.py        # Safety validators
├── pipelines/               # Orchestration
│   ├── base_pipeline.py     # Abstract pipeline
│   ├── pipeline_runner.py   # Execution engine
│   └── configs/             # YAML configurations
├── cli/                     # Command-line interface
│   ├── main.py              # Typer application
│   └── commands/            # Subcommands
├── evaluation/              # Metrics and benchmarks
│   ├── metrics.py           # Evaluation metrics
│   ├── datasets.py          # Dataset abstractions
│   └── benchmark_suites.py  # Benchmark definitions
└── governance/              # Rhenium Governance Layer
    ├── audit_log.py         # Audit logging
    ├── model_card.py        # Model documentation
    ├── fairness_metrics.py  # Fairness evaluation
    └── bias_mitigation.py   # Mitigation strategies
```

---

## Registry Pattern

Rhenium OS uses a centralized registry for component discovery:

```python
from rhenium.core.registry import registry, ComponentType

# Register a model
registry.register(
    name="my_segmentation",
    component_type=ComponentType.PERCEPTION,
    component_class=MySegmentationModel,
    version="1.0.0",
    description="Custom segmentation model",
    tags=["segmentation", "mri", "brain"],
)

# Retrieve a component
model_class = registry.get("my_segmentation", ComponentType.PERCEPTION)
model = model_class()
```

### Component Types

| Type | Description |
|------|-------------|
| `RECONSTRUCTION` | Image reconstruction from raw data |
| `PERCEPTION` | Detection, segmentation, classification |
| `XAI` | Explainability components |
| `PIPELINE` | End-to-end pipelines |
| `ORGAN_MODULE` | Organ-specific analysis modules |

---

## Configuration Management

Settings are managed via Pydantic:

```python
from rhenium.core.config import get_settings

settings = get_settings()
print(settings.data_dir)
print(settings.device)
print(settings.medgemma_backend)
```

### Environment Variables

| Variable | Description |
|----------|-------------|
| `RHENIUM_DATA_DIR` | Default data directory |
| `RHENIUM_MODELS_DIR` | Model weights directory |
| `RHENIUM_DEVICE` | Compute device (auto/cpu/cuda) |
| `RHENIUM_LOG_LEVEL` | Logging level |
| `RHENIUM_MEDGEMMA_BACKEND` | MedGemma backend type |

---

## Error Handling

All errors inherit from `RheniumError`:

```python
from rhenium.core.errors import (
    RheniumError,
    DataIngestionError,
    ReconstructionError,
    ModelInferenceError,
    MedGemmaError,
)

try:
    result = pipeline.run(data)
except DataIngestionError as e:
    logger.error("Data ingestion failed", error=e.to_dict())
except ModelInferenceError as e:
    logger.error("Inference failed", error=e.to_dict())
```

---

## Logging

Structured logging with PHI sanitization:

```python
from rhenium.core.logging import get_logger

logger = get_logger("my_module")
logger.info("Processing study", study_id="HASHED_123", slices=50)
```

---

## Data Models

Core data structures used throughout:

### ImageVolume

```python
@dataclass
class ImageVolume:
    array: np.ndarray
    spacing: tuple[float, float, float]
    modality: Modality
    series_uid: str
    metadata: dict
```

### Finding

```python
@dataclass
class Finding:
    finding_id: str
    finding_type: str
    description: str
    confidence: float
    bounding_box: BoundingBox | None
    segmentation_mask: np.ndarray | None
    measurements: dict[str, float]
```

### EvidenceDossier

```python
@dataclass
class EvidenceDossier:
    finding: Finding
    visual_evidence: list[VisualEvidence]
    quantitative_evidence: list[QuantitativeEvidence]
    narrative_evidence: list[NarrativeEvidence]
```

---

## Interface Contracts

### Pipeline Interface

All pipelines implement:

```python
class BasePipeline(ABC):
    @abstractmethod
    def load_input(self, source: str) -> Any:
        pass
    
    @abstractmethod
    def preprocess(self, data: Any) -> Any:
        pass
    
    @abstractmethod
    def infer(self, data: Any) -> list[Finding]:
        pass
    
    @abstractmethod
    def generate_evidence(self, findings: list[Finding]) -> list[EvidenceDossier]:
        pass
    
    @abstractmethod
    def run(self, source: str) -> PipelineResult:
        pass
```

### Model Interface

Perception models implement:

```python
class BaseSegmentationModel(ABC):
    @abstractmethod
    def load(self) -> None:
        pass
    
    @abstractmethod
    def predict(self, image: np.ndarray) -> SegmentationResult:
        pass
```

---

## Extension Points

### Adding New Models

1. Create model class inheriting from appropriate base
2. Register with the registry
3. Configure in pipeline YAML

### Adding New Organ Modules

1. Create module under `perception/organ/`
2. Implement organ-specific models
3. Create pipeline configuration
4. Register module

### Adding New Validators

1. Inherit from `BaseValidator` in `medgemma/validators.py`
2. Implement `validate()` method
3. Add to `ValidationSuite`

---

**Copyright (c) 2025 Skolyn LLC. All rights reserved.**

**SPDX-License-Identifier: EUPL-1.1**
