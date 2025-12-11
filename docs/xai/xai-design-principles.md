# XAI Design Principles: Skolyn Rhenium OS

---

## Introduction

Explainable AI (XAI) is not optional in medical imaging, it is a fundamental requirement for clinical adoption, regulatory approval, and patient safety. This document describes the design principles governing explainability in Rhenium OS.

---

## Core Principles

### 1. Every Finding Requires Evidence

No finding is reported without accompanying evidence. The Evidence Dossier is a mandatory output that provides:

- **Visual evidence**: What the model "saw" that led to the finding
- **Quantitative evidence**: Objective measurements supporting the finding
- **Narrative evidence**: Human-readable explanation of reasoning

### 2. Transparency Over Opacity

When faced with a choice between a more accurate but opaque model and a slightly less accurate but interpretable model, Rhenium OS favors interpretability, unless the accuracy difference is clinically significant.

### 3. Uncertainty is Information

Uncertainty estimates are first-class outputs:

- Confidence scores for all predictions
- Calibrated probabilities where possible
- Explicit flagging of low-confidence findings
- Uncertainty visualization in Evidence Dossiers

### 4. Fairness-Aware Explanation

Explanations must not reinforce biases:

- Demographic factors should not appear as causal in explanations
- Scanner or institution effects should be disclosed, not hidden
- Subgroup performance disparities should be acknowledged

### 5. Limitations are Acknowledged

Every AI output includes appropriate caveats:

- Known limitations of the model
- Conditions where performance may degrade
- Recommendation for human verification

---

## Evidence Dossier Structure

```
EvidenceDossier
  |
  +-- Finding
  |     +-- finding_id: unique identifier
  |     +-- finding_type: lesion, tear, hemorrhage, etc.
  |     +-- location: anatomical location
  |     +-- severity: critical/high/moderate/low/normal
  |     +-- confidence: 0.0 - 1.0
  |     +-- model_name, model_version
  |
  +-- VisualEvidence[]
  |     +-- artifact_type: saliency, overlay, attention, contour
  |     +-- data: image array
  |     +-- colormap, slice_indices
  |     +-- description
  |
  +-- QuantitativeEvidence[]
  |     +-- measurements: {name: value}
  |     +-- units: {name: unit}
  |     +-- confidence_intervals
  |     +-- uncertainty
  |     +-- radiomics_features
  |
  +-- NarrativeEvidence[]
        +-- explanation: natural language
        +-- reasoning_steps: []
        +-- guideline_references: []
        +-- limitations: []
        +-- recommendations: []
        +-- confidence_statement
```

---

## Visual Explanation Techniques

### Saliency Maps

- **Grad-CAM**: Gradient-weighted class activation mapping
- **Integrated Gradients**: Attribution via path integration
- **Occlusion Sensitivity**: Perturbation-based attribution

### Overlays and Contours

- Segmentation mask overlays with class-specific colors
- Bounding box visualization for detections
- Uncertainty heatmaps showing prediction confidence

### Attention Visualization

- Transformer attention maps
- Feature importance across spatial locations

---

## Quantitative Evidence

### Measurements

- Volume, diameter, surface area
- RECIST-compliant measurements where applicable
- Comparison to prior studies (if available)

### Radiomics

- Shape descriptors
- Texture features (GLCM, GLRLM)
- Intensity statistics

### Uncertainty Metrics

- Prediction standard deviation (ensemble/MC dropout)
- Entropy of class probabilities
- Out-of-distribution scores

---

## Narrative Evidence (MedGemma)

### Content Requirements

1. **Finding Description**: Objective characterization
2. **Clinical Significance**: Why this matters
3. **Differential Considerations**: Alternative interpretations
4. **Supporting Evidence**: Reference to visual/quantitative
5. **Limitations**: What the AI cannot determine
6. **Recommendations**: Suggested follow-up (if appropriate)

### Tone and Style

- Professional, clinical language
- Avoidance of overconfident statements
- Clear distinction between observation and interpretation
- No patient-identifiable information

---

## Integration with MedGemma

MedGemma generates narrative evidence using:

1. **Structured Prompts**: Templates ensuring consistent output format
2. **Context Injection**: Quantitative measurements and visual descriptions
3. **Uncertainty Awareness**: Prompts designed to elicit uncertainty acknowledgment
4. **Validation Hooks**: Post-generation checks for consistency

---

## Regulatory Alignment

| Requirement | XAI Response |
|-------------|--------------|
| AI Act Article 13 (Transparency) | Evidence Dossier provides full transparency |
| AI Act Article 14 (Human Oversight) | Findings require radiologist review |
| MDR Annex I Section 23.4 | Information for safe use via model cards |
| GDPR Article 22 (Automated Decisions) | Meaningful information via explanations |

---

## Conclusion

Explainability in Rhenium OS is not an afterthought but a core architectural principle. The Evidence Dossier framework ensures that every AI finding is accompanied by comprehensive, multi-modal evidence that enables radiologists to verify, trust, and appropriately act upon AI outputs.

---

**Copyright (c) 2025 Skolyn LLC. All rights reserved.**

---

## Documentation Update Notes

- Last updated: December 2025.
- Aligned with Rhenium OS disease reasoning layer.
- Revised markdown structure for consistency.
