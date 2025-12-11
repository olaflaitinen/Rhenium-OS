# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Disease Presence Assessment
===========================

Logic for determining disease presence or absence from imaging evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rhenium.core.disease_types import (
    CaseEvidenceBundle,
    DiseasePresenceAssessment,
    DiseasePresenceStatus,
)
from rhenium.core.logging import get_pipeline_logger

logger = get_pipeline_logger()


@dataclass
class PresenceThresholds:
    """Thresholds for presence determination."""
    lesion_count_threshold: int = 1
    abnormality_score_threshold: float = 0.5
    uncertainty_high_threshold: float = 0.7
    uncertainty_low_threshold: float = 0.3


class PresenceAssessor:
    """
    Assess overall disease presence from perception outputs.
    
    Combines lesion counts, abnormality scores, and global features
    to determine presence/absence/uncertainty.
    """
    
    def __init__(self, thresholds: PresenceThresholds | None = None):
        """Initialize with thresholds."""
        self.thresholds = thresholds or PresenceThresholds()
    
    def assess(
        self,
        evidence: CaseEvidenceBundle,
        modality: str = "",
        organ_systems: list[str] | None = None,
    ) -> DiseasePresenceAssessment:
        """
        Assess disease presence from evidence bundle.
        
        Args:
            evidence: Aggregated perception evidence.
            modality: Imaging modality.
            organ_systems: List of organ systems evaluated.
        
        Returns:
            Disease presence assessment.
        """
        assessment = DiseasePresenceAssessment(
            study_id=evidence.study_id,
            modality=modality,
            organ_systems_involved=organ_systems or [],
        )
        
        # Count lesions
        lesion_count = len(evidence.lesion_features)
        
        # Get global abnormality score
        global_abnormality = evidence.global_features.get(
            "abnormality_score", 0.0
        )
        
        # Collect all evidence IDs
        evidence_ids = [
            lesion.get("id", "") for lesion in evidence.lesion_features
        ]
        assessment.evidence_ids = [eid for eid in evidence_ids if eid]
        
        # Determine presence status
        if lesion_count >= self.thresholds.lesion_count_threshold:
            assessment.disease_present = DiseasePresenceStatus.PRESENT
            assessment.overall_abnormality_score = max(global_abnormality, 0.5)
            assessment.rationale = (
                f"Detected {lesion_count} lesion(s) meeting threshold criteria."
            )
        elif global_abnormality >= self.thresholds.abnormality_score_threshold:
            assessment.disease_present = DiseasePresenceStatus.PRESENT
            assessment.overall_abnormality_score = global_abnormality
            assessment.rationale = (
                f"Global abnormality score {global_abnormality:.2f} exceeds threshold."
            )
        elif global_abnormality >= self.thresholds.uncertainty_low_threshold:
            assessment.disease_present = DiseasePresenceStatus.UNCERTAIN
            assessment.overall_abnormality_score = global_abnormality
            assessment.uncertainty_score = 0.5
            assessment.rationale = (
                "Borderline findings; disease cannot be excluded."
            )
        else:
            assessment.disease_present = DiseasePresenceStatus.ABSENT
            assessment.overall_abnormality_score = global_abnormality
            assessment.rationale = self._generate_negative_rationale(
                evidence, organ_systems
            )
        
        # Calculate uncertainty
        assessment.uncertainty_score = self._calculate_uncertainty(
            evidence, assessment
        )
        
        # Add limitations
        assessment.limitations = self._identify_limitations(evidence)
        
        return assessment
    
    def _generate_negative_rationale(
        self,
        evidence: CaseEvidenceBundle,
        organ_systems: list[str] | None,
    ) -> str:
        """Generate explanation for negative finding."""
        parts = []
        
        if not evidence.lesion_features:
            parts.append("No focal lesions detected above threshold.")
        
        if organ_systems:
            parts.append(
                f"Evaluated organ systems: {', '.join(organ_systems)}."
            )
        
        return " ".join(parts) if parts else "No significant abnormality detected."
    
    def _calculate_uncertainty(
        self,
        evidence: CaseEvidenceBundle,
        assessment: DiseasePresenceAssessment,
    ) -> float:
        """Calculate uncertainty score based on evidence quality."""
        uncertainty = 0.0
        
        # Check image quality indicators
        quality = evidence.metadata_features.get("image_quality", 1.0)
        if quality < 0.7:
            uncertainty += 0.2
        
        # Check for conflicting evidence
        conflicting = evidence.global_features.get("conflicting_findings", False)
        if conflicting:
            uncertainty += 0.3
        
        # Check for severe motion artifacts
        motion = evidence.metadata_features.get("motion_artifact_severity", 0.0)
        if motion > 0.5:
            uncertainty += 0.2
        
        return min(uncertainty, 1.0)
    
    def _identify_limitations(
        self,
        evidence: CaseEvidenceBundle,
    ) -> list[str]:
        """Identify limitations affecting the assessment."""
        limitations = []
        
        # Check image quality
        quality = evidence.metadata_features.get("image_quality", 1.0)
        if quality < 0.7:
            limitations.append("Limited by suboptimal image quality.")
        
        # Check coverage
        incomplete = evidence.metadata_features.get("incomplete_coverage", False)
        if incomplete:
            limitations.append("Incomplete anatomical coverage.")
        
        # Check contrast
        contrast = evidence.metadata_features.get("contrast_phase", "")
        if not contrast:
            limitations.append("Non-contrast study; some findings may be occult.")
        
        return limitations
