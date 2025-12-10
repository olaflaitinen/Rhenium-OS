# XAI Architecture

**
---

## Overview

This document describes the explainability (XAI) architecture in Rhenium OS, including evidence dossier structure, generation flow, and integration with MedGemma.

---

## Evidence Dossier Structure

```mermaid
classDiagram
    class EvidenceDossier {
        +str dossier_id
        +Finding finding
        +list~VisualEvidence~ visual
        +list~QuantitativeEvidence~ quantitative
        +list~NarrativeEvidence~ narrative
        +datetime generated_at
        +to_dict()
        +to_json()
    }
    
    class Finding {
        +str finding_id
        +str finding_type
        +str description
        +float confidence
        +BoundingBox bbox
        +ndarray mask
        +dict measurements
    }
    
    class VisualEvidence {
        +str evidence_type
        +ndarray image
        +str description
        +ColorMap colormap
    }
    
    class QuantitativeEvidence {
        +str metric_name
        +float value
        +str unit
        +UncertaintyInterval uncertainty
    }
    
    class NarrativeEvidence {
        +str explanation
        +list~str~ limitations
        +str confidence_statement
        +list~str~ references
    }
    
    EvidenceDossier --> Finding
    EvidenceDossier --> "0..*" VisualEvidence
    EvidenceDossier --> "0..*" QuantitativeEvidence
    EvidenceDossier --> "0..*" NarrativeEvidence
```

---

## XAI Generation Flow

```mermaid
flowchart TD
    subgraph Input
        A[Finding from Perception] --> B[Image Volume]
    end
    
    subgraph Visual[Visual Evidence Generation]
        B --> C[Saliency Maps]
        B --> D[Attention Overlays]
        B --> E[Segmentation Contours]
        B --> F[Bounding Boxes]
    end
    
    subgraph Quantitative[Quantitative Evidence]
        A --> G[Measurements]
        A --> H[Radiomics Features]
        A --> I[Uncertainty Estimation]
        A --> J[Confidence Calibration]
    end
    
    subgraph Narrative[Narrative Evidence]
        A --> K[MedGemma Adapter]
        K --> L[Finding Explanation]
        K --> M[Differential Diagnosis]
        K --> N[Limitations Statement]
    end
    
    C & D & E & F --> O[Visual Evidence List]
    G & H & I & J --> P[Quantitative Evidence List]
    L & M & N --> Q[Narrative Evidence List]
    
    O & P & Q --> R[Evidence Dossier]
```

---

## Visual Evidence Types

| Type | Description | Generation Method |
|------|-------------|-------------------|
| Saliency Map | Pixel-level importance | Grad-CAM, Integrated Gradients |
| Attention Overlay | Model attention regions | Transformer attention weights |
| Segmentation Contour | Predicted structure boundary | Thresholded probability map |
| Bounding Box | Localized detection region | Detection model output |
| Difference Map | Comparison to reference | Subtraction, reconstruction error |

### Saliency Generation

```mermaid
flowchart LR
    A[Input Image] --> B[Model Forward Pass]
    B --> C[Target Class Selection]
    C --> D[Gradient Computation]
    D --> E[Gradient Weighting]
    E --> F[Saliency Map]
    F --> G[Overlay on Image]
```

---

## Quantitative Evidence

### Measurements

| Measurement | Applicable Findings | Unit |
|-------------|---------------------|------|
| Maximum diameter | Lesions, tumors | mm |
| Volume | Segmented structures | mm³, mL |
| Mean intensity | ROI analysis | HU, signal intensity |
| Texture features | Radiomics | Various |
| ADC value | DWI lesions | × 10⁻³ mm²/s |

### Uncertainty Quantification

$$\text{Uncertainty} = \sqrt{\text{Var}(\hat{y})}$$

Methods:
- Monte Carlo Dropout
- Deep Ensembles
- Bayesian Neural Networks

### Calibration

$$P(\hat{y} = y | \text{conf}(\hat{y}) = p) = p$$

Measured via Expected Calibration Error (ECE).

---

## MedGemma Integration for XAI

```mermaid
sequenceDiagram
    participant Dossier as Evidence Dossier
    participant Adapter as MedGemma Adapter
    participant Validator as Validators
    participant Template as Prompt Templates
    participant Model as MedGemma
    
    Dossier->>Adapter: request_explanation(finding, evidence)
    Adapter->>Template: select_template(finding_type, modality)
    Template-->>Adapter: prompt_template
    Adapter->>Adapter: format_prompt(finding, evidence)
    Adapter->>Model: generate(formatted_prompt)
    Model-->>Adapter: raw_explanation
    Adapter->>Validator: validate(explanation)
    Validator-->>Adapter: validation_result
    
    alt Validation Failed
        Adapter->>Adapter: flag_for_review()
    end
    
    Adapter-->>Dossier: NarrativeEvidence
```

---

## Narrative Templates

### Structure

```yaml
template:
  name: mri_brain_lesion_explanation
  modality: MRI
  finding_type: brain_lesion
  
  system_prompt: |
    You are a neuroradiology assistant explaining findings...
    
  user_prompt: |
    FINDING: {finding_description}
    LOCATION: {anatomical_location}
    SIZE: {measurements}
    SIGNAL CHARACTERISTICS: {signal_info}
    CONFIDENCE: {confidence}
    
    Provide a clinical explanation including:
    1. Description of the finding
    2. Differential diagnosis
    3. Recommended follow-up
    4. Limitations of AI assessment
    
  required_sections:
    - description
    - differential
    - recommendation
    - limitations
```

---

## Fairness-Aware XAI

```mermaid
flowchart TD
    A[Finding with Demographics] --> B{Demographic Data Available?}
    B -->|Yes| C[Stratified Confidence Analysis]
    B -->|No| D[Standard Explanation]
    
    C --> E{Confidence Disparity?}
    E -->|Yes| F[Include Disparity Warning]
    E -->|No| G[Include Calibration Statement]
    
    F --> H[Narrative Evidence]
    G --> H
    D --> H
```

### Disparity Disclosure

When performance disparities exist across subgroups, narrative evidence includes:

> "Note: Model confidence for this demographic subgroup may differ from overall calibration. Clinical correlation is especially important."

---

## Evidence Export Formats

| Format | Use Case | Contents |
|--------|----------|----------|
| JSON | API integration | Full structured data |
| HTML | Human review | Rendered visuals + text |
| DICOM-SR | PACS integration | Structured report |
| PDF | Clinical documentation | Formatted report |

---

**Copyright (c) 2025 Skolyn LLC. All rights reserved.**

****
