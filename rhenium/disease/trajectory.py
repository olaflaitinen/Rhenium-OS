# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Disease Trajectory Assessment
=============================

Longitudinal comparison for tracking disease course over time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from rhenium.core.disease_types import (
    CaseEvidenceBundle,
    DiseaseTrajectoryAssessment,
    TrajectoryLabel,
)
from rhenium.core.logging import get_pipeline_logger

logger = get_pipeline_logger()


@dataclass
class TrajectoryThresholds:
    """Thresholds for trajectory determination."""
    improvement_threshold: float = -0.20  # 20% decrease
    progression_threshold: float = 0.20   # 20% increase
    volume_change_threshold: float = 0.30  # 30% volume change
    new_lesion_threshold: int = 1


class TrajectoryAssessor:
    """
    Assess disease trajectory by comparing current and prior studies.
    
    Determines whether disease is improving, stable, or worsening
    based on lesion counts, sizes, and quantitative metrics.
    """
    
    def __init__(self, thresholds: TrajectoryThresholds | None = None):
        """Initialize with thresholds."""
        self.thresholds = thresholds or TrajectoryThresholds()
    
    def assess(
        self,
        evidence_current: CaseEvidenceBundle,
        evidence_prior: CaseEvidenceBundle,
        disease_code: str = "",
    ) -> DiseaseTrajectoryAssessment:
        """
        Assess disease trajectory between two studies.
        
        Args:
            evidence_current: Current study evidence.
            evidence_prior: Prior study evidence.
            disease_code: Disease being tracked.
        
        Returns:
            Trajectory assessment.
        """
        assessment = DiseaseTrajectoryAssessment(
            study_id_current=evidence_current.study_id,
            study_id_previous=evidence_prior.study_id,
            disease_code=disease_code,
        )
        
        # Calculate lesion changes
        current_lesions = evidence_current.lesion_features
        prior_lesions = evidence_prior.lesion_features
        
        # Count new and resolved lesions
        assessment.new_lesion_count = self._count_new_lesions(
            current_lesions, prior_lesions
        )
        assessment.resolved_lesion_count = self._count_resolved_lesions(
            current_lesions, prior_lesions
        )
        
        # Calculate quantitative deltas
        assessment.quantitative_deltas = self._calculate_deltas(
            evidence_current, evidence_prior
        )
        
        # Generate lesion change summaries
        assessment.lesion_change_summaries = self._summarize_lesion_changes(
            current_lesions, prior_lesions
        )
        
        # Determine trajectory label
        assessment.trajectory_label = self._determine_trajectory(
            assessment
        )
        
        # Generate response category
        assessment.response_category = self._determine_response_category(
            assessment
        )
        
        # Calculate timeline
        assessment.timeline_days = self._calculate_timeline(
            evidence_current, evidence_prior
        )
        
        # Generate narrative comment
        assessment.timeline_comment = self._generate_comment(assessment)
        
        return assessment
    
    def _count_new_lesions(
        self,
        current: list[dict[str, Any]],
        prior: list[dict[str, Any]],
    ) -> int:
        """Count lesions present in current but not in prior."""
        prior_ids = {l.get("id", "") for l in prior}
        new_count = 0
        
        for lesion in current:
            lesion_id = lesion.get("id", "")
            matched_prior = lesion.get("matched_prior_id", "")
            
            if not matched_prior and lesion_id not in prior_ids:
                new_count += 1
        
        return new_count
    
    def _count_resolved_lesions(
        self,
        current: list[dict[str, Any]],
        prior: list[dict[str, Any]],
    ) -> int:
        """Count lesions present in prior but not in current."""
        current_ids = {l.get("id", "") for l in current}
        current_matched = {l.get("matched_prior_id", "") for l in current}
        resolved_count = 0
        
        for lesion in prior:
            prior_id = lesion.get("id", "")
            if prior_id not in current_ids and prior_id not in current_matched:
                resolved_count += 1
        
        return resolved_count
    
    def _calculate_deltas(
        self,
        current: CaseEvidenceBundle,
        prior: CaseEvidenceBundle,
    ) -> dict[str, float]:
        """Calculate quantitative metric changes."""
        deltas = {}
        
        # Total lesion volume change
        current_vol = sum(
            l.get("volume_mm3", 0) for l in current.lesion_features
        )
        prior_vol = sum(
            l.get("volume_mm3", 0) for l in prior.lesion_features
        )
        
        if prior_vol > 0:
            deltas["total_volume_change_percent"] = (
                (current_vol - prior_vol) / prior_vol
            ) * 100
        
        # Lesion count change
        deltas["lesion_count_change"] = float(
            len(current.lesion_features) - len(prior.lesion_features)
        )
        
        # Sum of longest diameters
        current_sld = sum(
            l.get("longest_diameter_mm", 0) for l in current.lesion_features
        )
        prior_sld = sum(
            l.get("longest_diameter_mm", 0) for l in prior.lesion_features
        )
        
        if prior_sld > 0:
            deltas["sum_longest_diameters_change_percent"] = (
                (current_sld - prior_sld) / prior_sld
            ) * 100
        
        return deltas
    
    def _summarize_lesion_changes(
        self,
        current: list[dict[str, Any]],
        prior: list[dict[str, Any]],
    ) -> list[str]:
        """Generate per-lesion change summaries."""
        summaries = []
        
        for lesion in current:
            matched_id = lesion.get("matched_prior_id", "")
            location = lesion.get("location", "unknown location")
            current_size = lesion.get("longest_diameter_mm", 0)
            
            if matched_id:
                # Find matched prior lesion
                prior_lesion = next(
                    (l for l in prior if l.get("id") == matched_id), None
                )
                if prior_lesion:
                    prior_size = prior_lesion.get("longest_diameter_mm", 0)
                    change = current_size - prior_size
                    
                    if abs(change) < 1:
                        summaries.append(f"Lesion at {location}: stable")
                    elif change > 0:
                        summaries.append(
                            f"Lesion at {location}: increased {change:.1f}mm"
                        )
                    else:
                        summaries.append(
                            f"Lesion at {location}: decreased {abs(change):.1f}mm"
                        )
            else:
                summaries.append(f"New lesion at {location}: {current_size:.1f}mm")
        
        return summaries
    
    def _determine_trajectory(
        self,
        assessment: DiseaseTrajectoryAssessment,
    ) -> TrajectoryLabel:
        """Determine overall trajectory label."""
        volume_change = assessment.quantitative_deltas.get(
            "total_volume_change_percent", 0
        )
        
        # Check for new lesions first
        if assessment.new_lesion_count >= self.thresholds.new_lesion_threshold:
            return TrajectoryLabel.NEW_FINDINGS
        
        # Check volume change
        if volume_change <= self.thresholds.improvement_threshold * 100:
            return TrajectoryLabel.IMPROVED
        elif volume_change >= self.thresholds.progression_threshold * 100:
            return TrajectoryLabel.WORSENED
        else:
            return TrajectoryLabel.STABLE
    
    def _determine_response_category(
        self,
        assessment: DiseaseTrajectoryAssessment,
    ) -> str:
        """Determine treatment response category (imaging surrogate)."""
        trajectory = assessment.trajectory_label
        
        if trajectory == TrajectoryLabel.IMPROVED:
            volume_change = assessment.quantitative_deltas.get(
                "total_volume_change_percent", 0
            )
            if volume_change <= -30:
                return "Partial Response (imaging surrogate)"
            return "Minor Response (imaging surrogate)"
        elif trajectory == TrajectoryLabel.STABLE:
            return "Stable Disease (imaging surrogate)"
        elif trajectory in [TrajectoryLabel.WORSENED, TrajectoryLabel.NEW_FINDINGS]:
            return "Progressive Disease (imaging surrogate)"
        else:
            return "Indeterminate"
    
    def _calculate_timeline(
        self,
        current: CaseEvidenceBundle,
        prior: CaseEvidenceBundle,
    ) -> Optional[int]:
        """Calculate days between studies."""
        current_date = current.metadata_features.get("study_date")
        prior_date = prior.metadata_features.get("study_date")
        
        if current_date and prior_date:
            # Would parse dates and calculate difference
            return None  # Placeholder
        return None
    
    def _generate_comment(
        self,
        assessment: DiseaseTrajectoryAssessment,
    ) -> str:
        """Generate narrative comment about trajectory."""
        label = assessment.trajectory_label.value
        
        parts = [f"Overall trajectory: {label.replace('_', ' ')}."]
        
        if assessment.new_lesion_count > 0:
            parts.append(f"{assessment.new_lesion_count} new lesion(s) detected.")
        
        if assessment.resolved_lesion_count > 0:
            parts.append(f"{assessment.resolved_lesion_count} lesion(s) resolved.")
        
        vol_change = assessment.quantitative_deltas.get(
            "total_volume_change_percent"
        )
        if vol_change is not None:
            parts.append(f"Total volume change: {vol_change:+.1f}%.")
        
        return " ".join(parts)
