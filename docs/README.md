# Rhenium OS Documentation

## Overview

Welcome to the official documentation for **Rhenium OS**, the state-of-the-art AI operating system for diagnostic medical imaging.

## Documentation Index

### 1. Vision and Clinical Context
- [**Vision**](vision/vision.md): The mission, clinical context, and roadmap for Rhenium OS.
- [**Feature Taxonomy**](vision/feature-taxonomy.md): Detailed breakdown of capabilities across modalities and organs.
- [**Clinical Procedures**](clinical_procedures.md): Reference for clinical reasoning steps (staging, prognosis, safety flags).
- [**Team**](team/AI_AND_RHENIUM_OS_TEAM.md): The engineering, AI, and medical teams behind Rhenium OS.

### 2. Architecture
- [**Architecture Overview**](architecture/architecture-overview.md): High-level system design and layers.
- [**Core Architecture**](architecture/core-architecture.md): Deep dive into core components and package structure.
- [**Pipeline Architecture**](architecture/pipeline-architecture.md): Configuration-driven orchestration details.
- [**Modality Architectures**]:
    - [**MRI**](architecture/modality-architecture-mri.md)
    - [**CT**](architecture/modality-architecture-ct.md)
    - [**Ultrasound**](architecture/modality-architecture-ultrasound.md)
    - [**X-ray**](architecture/modality-architecture-xray.md)
- [**XAI Architecture**](architecture/xai-architecture.md): The engine behind Evidence Dossiers.
- [**MedGemma Integration**](architecture/medgemma-integration.md): LLM-based reasoning and reporting.
- [**Generative & PINNs**](architecture/generative-and-pinns-architecture.md): Advanced reconstruction and generation models.

### 3. Usage & Development
- [**CLI Guide**](usage/cli-guide.md): Command-line interface reference and examples.
- [**Developer Guide**](usage/developer-guide.md): Environment setup, testing, and contribution workflows.
- [**Configuration**](configuration.md): System configuration options and environment variables.
- [**CLI Reference**](../docs/cli_reference.md): Automated CLI command reference.

### 4. Benchmarks & Evaluation
- [**Benchmark Plans**](evaluation/benchmark-plans.md): Evaluation strategies and dataset definitions.
- [**Metrics Appendix**](evaluation/metrics-mathematical-appendix.md): Mathematical definitions of strict performance metrics.
- [**Modality Benchmarks**]:
    - [**MRI**](benchmarks/mri-benchmarks.md)
    - [**CT**](benchmarks/ct-benchmarks.md)
    - [**Ultrasound**](benchmarks/us-benchmarks.md)
    - [**X-ray**](benchmarks/xray-benchmarks.md)

### 5. Regulatory & Governance
- [**Regulatory Overview**](regulatory/regulatory-overview.md): Alignment with EU MDR and EU AI Act.
- [**XAI Design Principles**](xai/xai-design-principles.md): Framework for transparent, explainable AI.

---

## Quick Links

- [**Root README**](../README.md)
- [**Contributing Guidelines**](../CONTRIBUTING.md)
- [**Operations (Ops)**](../ops/README.md)

---

**Copyright (c) 2025 Skolyn LLC. All rights reserved.**
