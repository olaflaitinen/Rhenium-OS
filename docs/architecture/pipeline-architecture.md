# Pipeline Architecture

---

## Overview

This document describes the pipeline architecture in Rhenium OS, including composition patterns, execution flow, and configuration.

---

## Pipeline Execution Flow

```mermaid
flowchart TD
    subgraph Input
        A[Data Source] --> B[Data Ingestion]
    end
    
    subgraph Preprocessing
        B --> C[Format Validation]
        C --> D[Metadata Extraction]
        D --> E[Preprocessing Steps]
    end
    
    subgraph Reconstruction
        E --> F{Raw Data?}
        F -->|Yes| G[Reconstruction Engine]
        F -->|No| H[Pass Through]
        G --> I[Enhancement/Denoising]
        H --> I
    end
    
    subgraph Perception
        I --> J[Segmentation Models]
        I --> K[Detection Models]
        I --> L[Classification Models]
        J & K & L --> M[Finding Aggregation]
    end
    
    subgraph XAI
        M --> N[Visual Evidence]
        M --> O[Quantitative Evidence]
        N & O --> P[Evidence Dossier]
    end
    
    subgraph DiseaseReasoning
        P --> DR[Disease Assessor]
        DR --> DH[Disease Hypotheses]
        DH --> SF[Safety Flags]
    end
    
    subgraph Reasoning
        SF --> Q[MedGemma Adapter]
        Q --> R[Validators]
        R --> S[Narrative Generation]
    end
    
    subgraph Output
        S --> T[Report Draft]
        P --> U[Evidence Export]
        DH --> W[Disease Output]
        T & U & W --> V[Pipeline Result]
    end
```

---

## Pipeline Configuration

### YAML Structure

```yaml
# Example: mri_knee_meniscus.yaml

name: mri_knee_meniscus
version: "1.0.0"
description: "Knee MRI meniscus analysis pipeline"
pipeline_type: mri_knee

input:
  modality: MR
  body_part: KNEE
  required_sequences:
    - PD_FS
    - T2_SAGITTAL

preprocessing:
  steps:
    - type: intensity_normalization
      params:
        method: percentile
        lower: 1
        upper: 99
    - type: bias_field_correction
      params:
        n4itk: true
    - type: registration
      params:
        reference: PD_FS

reconstruction:
  enabled: false

perception:
  models:
    - name: meniscus_segmentation_v2
      type: segmentation
      targets: [medial_meniscus, lateral_meniscus]
    - name: meniscus_tear_detection_v1
      type: detection
      threshold: 0.5

xai:
  visual:
    enabled: true
    types: [segmentation_overlay, saliency_map]
  quantitative:
    enabled: true
    measurements: [tear_length, meniscus_volume]

medgemma:
  enabled: true
  template: mri_knee_meniscus
  generate_report: true
  validators: [consistency, laterality, high_risk]

output:
  format: json
  include_dossiers: true
```

---

## Pipeline Types

### By Modality

```mermaid
graph TD
    subgraph MRI[MRI Pipelines]
        M1[mri_knee_default]
        M2[mri_brain_lesions]
        M3[mri_prostate_pirads]
        M4[mri_breast_lesions]
        M5[mri_cardiac_function]
    end
    
    subgraph CT[CT Pipelines]
        C1[ct_head_ich_detection]
        C2[ct_lung_nodule_detection]
        C3[cta_vessel_analysis]
        C4[ct_cardiac_cta]
    end
    
    subgraph XR[X-ray Pipelines]
        X1[xray_chest_abnormality]
        X2[xray_bone_fracture]
        X3[mammo_lesion_detection]
    end
    
    subgraph US[Ultrasound Pipelines]
        U1[us_liver_lesion_detection]
        U2[us_carotid_plaque]
        U3[us_liver_elastography]
    end
```

### By Task Type

| Task Type | Pipeline Examples |
|-----------|-------------------|
| Segmentation | knee_meniscus, brain_lesions, prostate_zones |
| Detection | lung_nodule, ich_detection, lesion_detection |
| Classification | pirads_scoring, birads_classification |
| Quantification | atrophy_volume, fibrosis_staging |

---

## Abstract Pipeline Interface

```python
class BasePipeline(ABC):
    """Abstract base class for all Rhenium OS pipelines."""
    
    name: str
    version: str
    modality: Modality
    
    @abstractmethod
    def load_input(self, source: str | Path) -> Any:
        """Load and validate input data."""
        pass
    
    @abstractmethod
    def preprocess(self, data: Any) -> Any:
        """Apply preprocessing steps."""
        pass
    
    @abstractmethod
    def reconstruct(self, data: Any) -> ImageVolume:
        """Reconstruct from raw data if applicable."""
        pass
    
    @abstractmethod
    def infer(self, volume: ImageVolume) -> list[Finding]:
        """Run perception models."""
        pass
    
    @abstractmethod
    def generate_evidence(
        self, 
        volume: ImageVolume, 
        findings: list[Finding]
    ) -> list[EvidenceDossier]:
        """Generate XAI evidence dossiers."""
        pass
    
    @abstractmethod
    def reason(
        self, 
        findings: list[Finding], 
        dossiers: list[EvidenceDossier]
    ) -> ReportDraft:
        """Generate MedGemma narrative."""
        pass
    
    def run(self, source: str | Path) -> PipelineResult:
        """Execute full pipeline."""
        data = self.load_input(source)
        data = self.preprocess(data)
        volume = self.reconstruct(data)
        findings = self.infer(volume)
        dossiers = self.generate_evidence(volume, findings)
        report = self.reason(findings, dossiers)
        
        return PipelineResult(
            findings=findings,
            dossiers=dossiers,
            report=report,
            metadata=self._build_metadata(),
        )
```

---

## Step Logging and Audit

```mermaid
sequenceDiagram
    participant CLI
    participant Pipeline
    participant AuditLog
    participant Model
    participant MedGemma
    
    CLI->>Pipeline: run(source)
    Pipeline->>AuditLog: log_start(pipeline_id)
    
    Pipeline->>Pipeline: load_input()
    Pipeline->>AuditLog: log_step("load_input")
    
    Pipeline->>Pipeline: preprocess()
    Pipeline->>AuditLog: log_step("preprocess")
    
    Pipeline->>Model: infer()
    Model-->>Pipeline: findings
    Pipeline->>AuditLog: log_step("inference", model_versions)
    
    Pipeline->>MedGemma: reason()
    MedGemma-->>Pipeline: report
    Pipeline->>AuditLog: log_step("reasoning")
    
    Pipeline->>AuditLog: log_complete(result)
    Pipeline-->>CLI: PipelineResult
```

---

## Error Handling

```mermaid
flowchart TD
    A[Pipeline Step] --> B{Success?}
    B -->|Yes| C[Continue]
    B -->|No| D{Critical?}
    D -->|Yes| E[Raise PipelineError]
    D -->|No| F[Log Warning]
    F --> G[Graceful Degradation]
    G --> C
    E --> H[Log Failure]
    H --> I[Audit Entry]
```

---

## Performance Modes

| Mode | Description | Use Case |
|------|-------------|----------|
| Standard | Full pipeline with all XAI | Production reporting |
| Fast | Skip optional XAI, reduced resolution | Triage, screening |
| Detailed | Maximum XAI, highest resolution | Second reads, complex cases |
| Debug | Full logging, intermediate outputs | Development, troubleshooting |

---

**Copyright (c) 2025 Skolyn LLC. All rights reserved.**

