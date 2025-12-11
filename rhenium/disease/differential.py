# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Differential Diagnosis Generator
================================

Logic for generating ranked differential diagnoses from disease hypotheses.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rhenium.core.disease_types import (
    CaseEvidenceBundle,
    DiseaseHypothesis,
    DifferentialDiagnosisEntry,
)
from rhenium.core.logging import get_pipeline_logger

logger = get_pipeline_logger()


@dataclass
class DifferentialConfig:
    """Configuration for differential diagnosis generation."""
    max_entries: int = 5
    include_cannot_exclude: bool = True
    min_probability_threshold: float = 0.05
    include_suggested_tests: bool = True


class DifferentialGenerator:
    """
    Generate differential diagnosis lists from disease hypotheses.
    
    Ranks alternative diagnoses and identifies "cannot exclude" cases.
    """
    
    # Knowledge base of diseases requiring "cannot exclude" handling
    CANNOT_EXCLUDE_PATTERNS: dict[str, list[str]] = {
        "lung_nodule": ["malignancy", "metastasis"],
        "liver_lesion": ["hcc", "metastasis"],
        "brain_lesion": ["malignancy", "abscess"],
        "breast_lesion": ["malignancy"],
    }
    
    # Suggested additional tests by disease pattern
    SUGGESTED_TESTS: dict[str, list[str]] = {
        "lung_nodule": ["PET-CT", "CT-guided biopsy", "Follow-up CT in 3 months"],
        "liver_lesion": ["MRI with hepatobiliary contrast", "CT arterial phase"],
        "brain_lesion": ["MRI with contrast", "MR spectroscopy", "Perfusion MRI"],
        "pe": ["CT pulmonary angiography", "D-dimer", "Lower extremity ultrasound"],
    }
    
    def __init__(self, config: DifferentialConfig | None = None):
        """Initialize with configuration."""
        self.config = config or DifferentialConfig()
    
    def generate(
        self,
        hypotheses: list[DiseaseHypothesis],
        evidence: CaseEvidenceBundle,
        primary_pattern: str = "",
    ) -> list[DifferentialDiagnosisEntry]:
        """
        Generate differential diagnosis list.
        
        Args:
            hypotheses: Ranked disease hypotheses.
            evidence: Supporting evidence bundle.
            primary_pattern: Primary disease pattern for context.
        
        Returns:
            Ranked differential diagnosis entries.
        """
        entries = []
        
        # Filter hypotheses above threshold
        filtered = [
            h for h in hypotheses
            if h.probability >= self.config.min_probability_threshold
        ]
        
        # Take top entries
        for i, hyp in enumerate(filtered[:self.config.max_entries]):
            entry = DifferentialDiagnosisEntry(
                rank=i + 1,
                disease_code=hyp.disease_code,
                disease_name=hyp.disease_name,
                estimated_probability=hyp.probability,
                supporting_features=hyp.supporting_features.copy(),
                contradicting_features=hyp.contradicting_features.copy(),
            )
            
            # Check for "cannot exclude" status
            if self.config.include_cannot_exclude:
                entry.cannot_exclude_flag = self._should_cannot_exclude(
                    hyp, primary_pattern
                )
            
            # Add suggested tests
            if self.config.include_suggested_tests:
                entry.suggested_additional_tests = self._get_suggested_tests(
                    hyp, primary_pattern
                )
            
            entries.append(entry)
        
        return entries
    
    def _should_cannot_exclude(
        self,
        hypothesis: DiseaseHypothesis,
        primary_pattern: str,
    ) -> bool:
        """Determine if hypothesis requires 'cannot exclude' flag."""
        patterns = self.CANNOT_EXCLUDE_PATTERNS.get(primary_pattern, [])
        
        # Check if disease matches concerning patterns
        disease_lower = hypothesis.disease_name.lower()
        for pattern in patterns:
            if pattern in disease_lower:
                # If probability is moderate but not definitive
                if 0.1 <= hypothesis.probability <= 0.7:
                    return True
        
        return False
    
    def _get_suggested_tests(
        self,
        hypothesis: DiseaseHypothesis,
        primary_pattern: str,
    ) -> list[str]:
        """Get suggested additional tests for clarification."""
        tests = self.SUGGESTED_TESTS.get(primary_pattern, [])
        return tests[:3]  # Limit to 3 suggestions
    
    def add_cannot_exclude_entry(
        self,
        differential: list[DifferentialDiagnosisEntry],
        disease_code: str,
        disease_name: str,
        reason: str,
    ) -> list[DifferentialDiagnosisEntry]:
        """
        Add explicit 'cannot exclude' entry to differential.
        
        Used when clinical context suggests a diagnosis cannot be
        excluded despite low imaging probability.
        """
        entry = DifferentialDiagnosisEntry(
            rank=len(differential) + 1,
            disease_code=disease_code,
            disease_name=disease_name,
            estimated_probability=0.0,  # Unknown probability
            cannot_exclude_flag=True,
            notes=f"Cannot exclude based on: {reason}",
        )
        differential.append(entry)
        return differential
