# Rhenium OS Architecture

## Overview

Rhenium OS is a state-of-the-art AI Operating System for diagnostic imaging, built on proprietary deep learning models including PINNs, GANs, U-Net, Vision Transformers, and 3D CNNs. This document describes the core architecture and key design decisions.

## Component Layers

```
┌─────────────────────────────────────────────────────────────┐
│                     CLI / Python API                        │
├─────────────────────────────────────────────────────────────┤
│                  Pipeline Orchestration                      │
│     (Configuration-driven, YAML-based workflows)            │
├─────────────────────────────────────────────────────────────┤
│  MedGemma       │      XAI Layer      │    Governance       │
│  Integration    │  (Evidence Dossiers)│  (Audit, Risk)      │
├─────────────────────────────────────────────────────────────┤
│                    Perception Layer                          │
│  (Segmentation, Detection, Classification)                  │
│  (Organ Modules: Knee, Brain, Prostate, Breast)             │
├─────────────────────────────────────────────────────────────┤
│                  Reconstruction Layer                        │
│  (MRI k-space, CT sinograms, X-ray, Ultrasound)             │
│  (Rhenium Reconstruction Engine)                             │
├─────────────────────────────────────────────────────────────┤
│                      Data Layer                              │
│  (DICOM, NIfTI, Raw IO, Preprocessing, Metadata)            │
├─────────────────────────────────────────────────────────────┤
│                      Core Module                             │
│  (Config, Logging, Errors, Registry)                        │
└─────────────────────────────────────────────────────────────┘
```

## Key Design Principles

### 1. Explainability by Design
Every finding must have an Evidence Dossier with:
- **Visual evidence**: Saliency maps, segmentation overlays
- **Quantitative evidence**: Measurements, radiomics features
- **Narrative evidence**: MedGemma-generated explanations

### 2. Regulatory Preparedness
- GDPR-compliant PHI handling
- EU MDR and EU AI Act readiness
- Comprehensive audit logging
- Model cards for all perception models

### 3. Modular Architecture
- Plugin-based organ modules
- Version-aware component registry
- Configuration-driven pipelines

### 4. MedGemma Integration
Abstract adapter pattern supporting:
- Stub client for testing
- Local deployment
- Remote API access

## Directory Structure

```
rhenium/
├── core/           # Configuration, logging, errors, registry
├── data/           # DICOM, NIfTI, raw IO, preprocessing
├── reconstruction/ # MRI, CT, X-ray, Ultrasound processing
├── perception/     # Detection, segmentation, classification
│   └── organ/      # Knee, brain, prostate, breast modules
├── xai/            # Evidence dossiers, explanations
├── medgemma/       # MedGemma adapter and prompts
├── pipelines/      # Orchestration framework
├── cli/            # Command-line interface
├── evaluation/     # Metrics and benchmarks
└── governance/     # Audit, model cards, risk tracking
```
