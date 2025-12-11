# Copyright (c) 2025 Skolyn LLC. All rights reserved.

Developer Guide: Rhenium OS
===========================

This guide provides detailed instructions for developers working with
or extending Skolyn Rhenium OS.

# Developer Guide

## Overview

Rhenium OS is designed as a modular, extensible platform. This guide covers:

1. Setting up a development environment
2. Understanding the package structure
3. Adding new perception models
4. Creating new organ modules
5. Developing custom pipelines
6. Working with MedGemma integration
7. Testing and quality assurance

---

## Development Environment Setup

### Prerequisites

- Python 3.10 or higher
- Git
- CUDA toolkit (for GPU development)
- Virtual environment tool (venv, conda)

### Installation

```bash
# Clone repository
git clone https://github.com/skolyn/rhenium-os.git
cd rhenium-os

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install in development mode
pip install -e ".[dev]"

# Verify installation
rhenium --version
python -c "import rhenium; print(rhenium.__version__)"
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=rhenium --cov-report=html

# Run specific test module
pytest tests/test_config.py

# Run with verbose output
pytest -v
```

---

## Package Structure

```
rhenium/
├── __init__.py          # Version and root exports
├── core/                # Fundamental infrastructure
│   ├── config.py        # Pydantic settings
│   ├── logging.py       # Structured logging
│   ├── errors.py        # Error hierarchy
│   └── registry.py      # Component registry
├── data/                # Data I/O and preprocessing
├── reconstruction/      # Image reconstruction
│   └── pinns/           # Physics-informed models
├── perception/          # Perception models
│   └── organ/           # Organ-specific modules
├── generative/          # Generative models
├── xai/                 # Explainability
├── medgemma/            # MedGemma integration
├── pipelines/           # Orchestration
├── cli/                 # Command-line interface
├── evaluation/          # Metrics and benchmarks
└── governance/          # Audit and compliance
```

---

## Adding a New Perception Model

### Step 1: Define the Model Class

Create a new file or add to an existing module:

```python
# rhenium/perception/segmentation/my_model.py

from rhenium.perception.segmentation import BaseSegmentationModel, SegmentationResult
from rhenium.core.registry import registry
import numpy as np


class MySegmentationModel(BaseSegmentationModel):
    \"\"\"Custom segmentation model.\"\"\"
    
    LABELS = ["background", "target"]
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
    def load(self) -> None:
        # Load your model weights
        self._loaded = True
        
    def predict(self, image: np.ndarray) -> SegmentationResult:
        if not self._loaded:
            self.load()
        # Your inference logic here
        mask = np.zeros(image.shape[:3], dtype=np.int32)
        return SegmentationResult(mask=mask, class_labels=self.LABELS)


# Register the model
registry.register_model(
    "my_segmentation",
    MySegmentationModel,
    version="1.0.0",
    description="My custom segmentation model",
    tags=["segmentation", "custom"],
)
```

### Step 2: Add Tests

```python
# tests/test_my_model.py

import pytest
import numpy as np
from rhenium.perception.segmentation.my_model import MySegmentationModel


class TestMyModel:
    def test_predict(self):
        model = MySegmentationModel()
        image = np.random.rand(64, 64, 32).astype(np.float32)
        result = model.predict(image)
        assert result.mask.shape == image.shape
```

---

## Creating a New Organ Module

### Step 1: Create Module Structure

```
rhenium/perception/organ/liver/
├── __init__.py
└── models.py
```

### Step 2: Implement the Module

```python
# rhenium/perception/organ/liver/__init__.py

from rhenium.perception.organ.liver.models import LiverModule

__all__ = ["LiverModule"]
```

```python
# rhenium/perception/organ/liver/models.py

from dataclasses import dataclass, field
from typing import Any
import numpy as np

from rhenium.core.registry import registry
from rhenium.perception.segmentation import BaseSegmentationModel
from rhenium.perception.detection import BaseDetectionModel


class LiverSegmentationModel(BaseSegmentationModel):
    LABELS = ["background", "liver", "vessels", "lesion"]
    
    def load(self) -> None:
        self._loaded = True
        
    def predict(self, image: np.ndarray):
        # Implementation
        pass


@dataclass
class LiverModule:
    \"\"\"Complete liver analysis module.\"\"\"
    segmentation: LiverSegmentationModel = field(default_factory=LiverSegmentationModel)
    
    def analyze(self, image: np.ndarray) -> dict[str, Any]:
        return {"segmentation": self.segmentation.predict(image)}


# Register
registry.register_organ_module(
    "liver",
    LiverModule,
    version="1.0.0",
    description="Liver analysis module",
    tags=["liver", "abdomen"],
)
```

---

## Developing Custom Pipelines

### Step 1: Create Pipeline Configuration

```yaml
# rhenium/pipelines/configs/liver_ct.yaml

name: liver_ct_analysis
version: "1.0.0"
description: "Liver CT analysis pipeline"

pipeline_type: liver_ct

input:
  modality: CT
  body_part: ABDOMEN

perception:
  liver_segmentation:
    enabled: true
    model: liver_segmentation_v1

xai:
  visual:
    enabled: true
  quantitative:
    enabled: true

medgemma:
  enabled: true
  template: liver_ct
```

### Step 2: Create Pipeline Class

```python
from rhenium.pipelines.base_pipeline import BasePipeline, PipelineResult

class LiverCTPipeline(BasePipeline):
    name = "liver_ct"
    version = "1.0.0"
    
    def load_input(self, source):
        # Implementation
        pass
    
    # Implement other required methods
```

---

## Working with MedGemma

### Creating Custom Prompts

```python
from rhenium.medgemma.prompts.reporting_templates import ReportTemplate

LIVER_TEMPLATE = ReportTemplate(
    name="liver_ct",
    system_prompt=\"\"\"You are an expert radiologist assistant 
    specializing in liver imaging...\"\"\",
    user_prompt_template=\"\"\"
    INDICATION: {indication}
    FINDINGS: {findings}
    Generate a structured report.\"\"\",
    sections=["technique", "findings", "impression"],
)
```

### Using Tool-Use Framework

```python
from rhenium.medgemma.tools import get_tool_registry

registry = get_tool_registry()

# Invoke a tool
result = registry.invoke(
    "guideline_checker",
    finding=my_finding,
    guideline_set="acr",
)
```

---

## Code Quality Standards

### Style Guidelines

- Follow PEP 8
- Use type hints for all function signatures
- Write docstrings for all public functions and classes
- Maximum line length: 100 characters

### Pre-commit Checks

```bash
# Format code
black rhenium tests

# Sort imports
isort rhenium tests

# Type checking
mypy rhenium

# Linting
ruff check rhenium
```

---

## Submitting Changes

1. Create a feature branch
2. Make changes with tests
3. Ensure all tests pass
4. Update documentation
5. Submit pull request

---

**Copyright (c) 2025 Skolyn LLC. All rights reserved.**

