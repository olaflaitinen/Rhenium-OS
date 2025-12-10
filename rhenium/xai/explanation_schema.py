# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Explanation Schema
==================

Core data structures for findings and evidence in Rhenium OS XAI.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4

import numpy as np


class FindingSeverity(str, Enum):
    """Severity levels for clinical findings."""
    CRITICAL = "critical"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"
    INCIDENTAL = "incidental"
    NORMAL = "normal"


class EvidenceType(str, Enum):
    """Types of evidence."""
    VISUAL = "visual"
    QUANTITATIVE = "quantitative"
    NARRATIVE = "narrative"


@dataclass
class VisualEvidence:
    """Visual explanation artifacts."""
    evidence_id: str = field(default_factory=lambda: uuid4().hex[:12])
    evidence_type: EvidenceType = EvidenceType.VISUAL
    artifact_type: str = "heatmap"  # heatmap, saliency, overlay, contour
    data: np.ndarray | None = None
    image_reference: str = ""  # Reference to source image
    slice_indices: list[int] = field(default_factory=list)
    colormap: str = "jet"
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize (excluding numpy arrays)."""
        return {
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type.value,
            "artifact_type": self.artifact_type,
            "image_reference": self.image_reference,
            "slice_indices": self.slice_indices,
            "colormap": self.colormap,
            "description": self.description,
            "metadata": self.metadata,
        }


@dataclass
class QuantitativeEvidence:
    """Quantitative measurements and metrics."""
    evidence_id: str = field(default_factory=lambda: uuid4().hex[:12])
    evidence_type: EvidenceType = EvidenceType.QUANTITATIVE
    measurements: dict[str, float] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)
    confidence_intervals: dict[str, tuple[float, float]] = field(default_factory=dict)
    uncertainty: float | None = None
    radiomics_features: dict[str, float] = field(default_factory=dict)
    clinical_scores: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type.value,
            "measurements": self.measurements,
            "units": self.units,
            "confidence_intervals": {
                k: list(v) for k, v in self.confidence_intervals.items()
            },
            "uncertainty": self.uncertainty,
            "radiomics_features": self.radiomics_features,
            "clinical_scores": self.clinical_scores,
        }


@dataclass
class NarrativeEvidence:
    """Natural language explanations from MedGemma."""
    evidence_id: str = field(default_factory=lambda: uuid4().hex[:12])
    evidence_type: EvidenceType = EvidenceType.NARRATIVE
    explanation: str = ""
    reasoning_steps: list[str] = field(default_factory=list)
    guideline_references: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)
    confidence_statement: str = ""
    generated_by: str = "medgemma"
    generation_timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "evidence_type": self.evidence_type.value,
            "explanation": self.explanation,
            "reasoning_steps": self.reasoning_steps,
            "guideline_references": self.guideline_references,
            "limitations": self.limitations,
            "recommendations": self.recommendations,
            "confidence_statement": self.confidence_statement,
            "generated_by": self.generated_by,
            "generation_timestamp": self.generation_timestamp.isoformat(),
        }


@dataclass
class Finding:
    """
    A clinical finding with associated evidence.

    Every finding in Rhenium OS must have an associated Evidence Dossier
    containing visual, quantitative, and narrative evidence.
    """
    finding_id: str = field(default_factory=lambda: uuid4().hex[:12])
    finding_type: str = ""  # e.g., "lesion", "tear", "hemorrhage"
    description: str = ""
    location: str = ""
    laterality: str = ""  # left, right, bilateral
    severity: FindingSeverity = FindingSeverity.NORMAL
    confidence: float = 0.0
    model_name: str = ""
    model_version: str = ""
    visual_evidence: list[VisualEvidence] = field(default_factory=list)
    quantitative_evidence: list[QuantitativeEvidence] = field(default_factory=list)
    narrative_evidence: list[NarrativeEvidence] = field(default_factory=list)
    requires_review: bool = False
    review_reason: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize finding for JSON export."""
        return {
            "finding_id": self.finding_id,
            "finding_type": self.finding_type,
            "description": self.description,
            "location": self.location,
            "laterality": self.laterality,
            "severity": self.severity.value,
            "confidence": self.confidence,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "visual_evidence": [e.to_dict() for e in self.visual_evidence],
            "quantitative_evidence": [e.to_dict() for e in self.quantitative_evidence],
            "narrative_evidence": [e.to_dict() for e in self.narrative_evidence],
            "requires_review": self.requires_review,
            "review_reason": self.review_reason,
            "created_at": self.created_at.isoformat(),
        }
