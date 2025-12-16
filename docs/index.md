# Rhenium OS

[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-EUPL--1.1-green)](https://eupl.eu/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c)](https://pytorch.org/)
[![MONAI](https://img.shields.io/badge/MONAI-1.3+-blueviolet)](https://monai.io/)

**Multi-Modality AI Platform for Medical Imaging Research**

---

## Overview

Rhenium OS is a comprehensive platform for medical imaging AI research, supporting:

- **MRI, CT, Ultrasound, X-ray** modalities
- **Reconstruction** with Physics-Informed Neural Networks
- **Perception** (segmentation, detection, classification)
- **Generative** models with disclosure tracking
- **Evidence Dossier** framework for transparent XAI

## Architecture

```mermaid
graph TB
    subgraph Input["Data Layer"]
        DICOM[DICOM I/O]
        NIfTI[NIfTI I/O]
        Volume[ImageVolume]
    end
    
    subgraph Core["Core Framework"]
        Config[Configuration]
        Registry[Component Registry]
        Errors[Error Taxonomy]
    end
    
    subgraph Models["AI Models"]
        Recon[Reconstruction]
        Seg[Segmentation]
        Det[Detection]
        Gen[Generative]
    end
    
    subgraph Output["Output Layer"]
        XAI[Evidence Dossier]
        API[REST API]
        CLI[CLI]
    end
    
    DICOM --> Volume
    NIfTI --> Volume
    Volume --> Models
    Core --> Models
    Models --> XAI
    XAI --> API
    XAI --> CLI
```

## Quick Start

```bash
pip install rhenium-os

# Run benchmark with synthetic data
rhenium benchmark

# Start API server
rhenium serve
```

## Documentation

- [Installation Guide](getting-started/installation.md)
- [Quick Start](getting-started/quickstart.md)
- [API Reference](api/core.md)
- [Mathematical Foundations](math/foundations.md)

## Disclaimer

> ⚠️ **Research Use Only**: This software is not approved for clinical use. All AI findings require verification by qualified medical professionals.
