# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2025-01-01

### Added

- **Core Module**
  - Pydantic-based configuration system with environment variable support
  - Versioned component registry for models, pipelines, and preprocessors
  - Hierarchical error taxonomy
  - Structured logging with structlog
  - Reproducibility utilities (seeding, determinism)

- **Data Module**
  - `ImageVolume` class for 3D medical imaging data
  - DICOM I/O with de-identification support
  - NIfTI I/O
  - Preprocessing pipeline (resample, normalize, crop/pad)

- **Reconstruction Module**
  - MRI: Zero-filled, unrolled networks, VarNet
  - CT: FBP, SIRT, learned reconstruction
  - Ultrasound: Beamforming, speckle reduction
  - X-ray: Enhancement, bone suppression
  - Physics-Informed Neural Networks (PINNs)

- **Perception Module**
  - 3D U-Net and UNETR segmentation
  - CenterNet-style 3D detection
  - ResNet3D and DenseNet3D classification
  - Ordinal classification for grading scales

- **Generative Module**
  - Pix2Pix paired translation
  - CycleGAN unpaired translation
  - SRGAN/ESRGAN super-resolution
  - Generated image disclosure stamping

- **XAI Module**
  - Evidence Dossier framework
  - Gradient and Grad-CAM saliency maps
  - Quantitative measurement extraction

- **Governance Module**
  - Model Card and Dataset Card templates
  - Risk Register
  - Audit logging

- **Evaluation Module**
  - Dice, IoU, Hausdorff distance metrics
  - PSNR, SSIM for reconstruction
  - AUROC, Expected Calibration Error

- **CLI**
  - Typer-based command-line interface
  - Commands: version, ingest, synthetic, benchmark, serve

- **Server**
  - FastAPI backend with REST API
  - Health, ingest, pipeline, registry endpoints
  - Docker and docker-compose support

- **Testing**
  - Synthetic data generator
  - pytest test suite

### Documentation

- Comprehensive README
- Technical Bible (internal specification)
- CONTRIBUTING, CODE_OF_CONDUCT, SECURITY

[Unreleased]: https://github.com/rhenium-os/rhenium-os/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/rhenium-os/rhenium-os/releases/tag/v0.1.0
