# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
XAI Module - Explainable AI
===========================

Three-axis explainability framework:
- Visual: Heatmaps, saliency, overlays
- Quantitative: Measurements, radiomics, uncertainty
- Narrative: MedGemma-generated explanations
"""

from rhenium.xai.explanation_schema import (
    Finding,
    VisualEvidence,
    QuantitativeEvidence,
    NarrativeEvidence,
)
from rhenium.xai.evidence_dossier import EvidenceDossier
from rhenium.xai.visual_explanations import (
    generate_saliency_map,
    generate_segmentation_overlay,
)
from rhenium.xai.quantitative_explanations import (
    compute_measurements,
    compute_uncertainty_metrics,
)

__all__ = [
    "Finding",
    "VisualEvidence",
    "QuantitativeEvidence",
    "NarrativeEvidence",
    "EvidenceDossier",
    "generate_saliency_map",
    "generate_segmentation_overlay",
    "compute_measurements",
    "compute_uncertainty_metrics",
]
