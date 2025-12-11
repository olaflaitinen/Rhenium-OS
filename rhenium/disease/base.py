# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Base Disease Assessor
=====================

Abstract base classes for disease reasoning components.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from rhenium.core.disease_types import (
    CaseEvidenceBundle,
    DiseasePresenceAssessment,
    DiseaseHypothesis,
    DiseaseSubtypeHypothesis,
    DiseaseStageAssessment,
    DifferentialDiagnosisEntry,
    DiseaseTrajectoryAssessment,
    ClinicalSafetyFlag,
    DiseaseReasoningOutput,
)
from rhenium.core.logging import get_pipeline_logger

logger = get_pipeline_logger()


@dataclass
class DiseaseAssessorConfig:
    """
    Configuration for disease assessors.
    
    Attributes:
        enabled: Whether disease assessment is enabled.
        diseases: List of disease types to assess.
        generate_differential: Whether to generate differential diagnoses.
        max_differential_entries: Maximum differential diagnosis entries.
        detect_safety_flags: Whether to detect clinical safety flags.
        uncertainty_threshold: Threshold below which to mark as uncertain.
        confidence_threshold: Minimum confidence to report a finding.
        include_staging: Whether to include staging assessments.
        include_trajectory: Whether to include trajectory assessment.
    """
    enabled: bool = True
    diseases: list[str] = field(default_factory=list)
    generate_differential: bool = True
    max_differential_entries: int = 5
    detect_safety_flags: bool = True
    uncertainty_threshold: float = 0.3
    confidence_threshold: float = 0.5
    include_staging: bool = True
    include_trajectory: bool = True


class BaseDiseaseAssessor(ABC):
    """
    Abstract base class for disease assessment.
    
    Disease assessors take perception outputs (lesions, segmentations,
    measurements) and generate structured disease-level reasoning.
    
    Subclasses implement organ-specific or disease-specific logic.
    """
    
    # Assessor metadata
    name: str = "base_assessor"
    version: str = "1.0.0"
    supported_diseases: list[str] = []
    supported_modalities: list[str] = []
    
    def __init__(self, config: DiseaseAssessorConfig | None = None):
        """Initialize assessor with configuration."""
        self.config = config or DiseaseAssessorConfig()
    
    @abstractmethod
    def assess_presence(
        self,
        evidence: CaseEvidenceBundle,
    ) -> DiseasePresenceAssessment:
        """
        Assess whether disease is present.
        
        Args:
            evidence: Aggregated perception evidence.
        
        Returns:
            Disease presence assessment.
        """
        pass
    
    @abstractmethod
    def generate_hypotheses(
        self,
        evidence: CaseEvidenceBundle,
    ) -> list[DiseaseHypothesis]:
        """
        Generate disease hypotheses from evidence.
        
        Args:
            evidence: Aggregated perception evidence.
        
        Returns:
            Ranked list of disease hypotheses.
        """
        pass
    
    def generate_subtype_hypotheses(
        self,
        disease: DiseaseHypothesis,
        evidence: CaseEvidenceBundle,
    ) -> list[DiseaseSubtypeHypothesis]:
        """
        Generate subtype hypotheses for a disease.
        
        Default implementation returns empty list.
        Override in subclasses for disease-specific subtypes.
        """
        return []
    
    def assess_stage(
        self,
        disease: DiseaseHypothesis,
        evidence: CaseEvidenceBundle,
    ) -> Optional[DiseaseStageAssessment]:
        """
        Assess disease stage or severity.
        
        Default implementation returns None.
        Override in subclasses for disease-specific staging.
        """
        return None
    
    def generate_differential(
        self,
        hypotheses: list[DiseaseHypothesis],
        evidence: CaseEvidenceBundle,
    ) -> list[DifferentialDiagnosisEntry]:
        """
        Generate differential diagnosis list.
        
        Default implementation converts hypotheses to differential entries.
        """
        entries = []
        for i, hyp in enumerate(hypotheses[:self.config.max_differential_entries]):
            entry = DifferentialDiagnosisEntry(
                rank=i + 1,
                disease_code=hyp.disease_code,
                disease_name=hyp.disease_name,
                estimated_probability=hyp.probability,
                supporting_features=hyp.supporting_features,
                contradicting_features=hyp.contradicting_features,
            )
            entries.append(entry)
        return entries
    
    def detect_safety_flags(
        self,
        evidence: CaseEvidenceBundle,
        hypotheses: list[DiseaseHypothesis],
    ) -> list[ClinicalSafetyFlag]:
        """
        Detect clinical safety flags requiring escalation.
        
        Default implementation returns empty list.
        Override in subclasses for specific safety patterns.
        """
        return []
    
    def assess_trajectory(
        self,
        evidence_current: CaseEvidenceBundle,
        evidence_prior: Optional[CaseEvidenceBundle],
    ) -> Optional[DiseaseTrajectoryAssessment]:
        """
        Assess disease trajectory over time.
        
        Default implementation returns None if no prior evidence.
        Override in subclasses for specific trajectory logic.
        """
        if evidence_prior is None:
            return None
        return None
    
    def run(
        self,
        evidence: CaseEvidenceBundle,
        evidence_prior: Optional[CaseEvidenceBundle] = None,
    ) -> DiseaseReasoningOutput:
        """
        Run complete disease reasoning pipeline.
        
        Args:
            evidence: Current study evidence.
            evidence_prior: Prior study evidence for trajectory.
        
        Returns:
            Complete disease reasoning output.
        """
        logger.info(
            "Running disease assessment",
            assessor=self.name,
            study_id=evidence.study_id,
        )
        
        output = DiseaseReasoningOutput(
            study_id=evidence.study_id,
            pipeline_name=self.name,
            pipeline_version=self.version,
            evidence_bundle_id=evidence.bundle_id,
        )
        
        # Assess presence
        output.presence_assessment = self.assess_presence(evidence)
        
        # Generate hypotheses
        output.primary_hypotheses = self.generate_hypotheses(evidence)
        
        # Generate subtypes and staging for each hypothesis
        for hypothesis in output.primary_hypotheses:
            subtypes = self.generate_subtype_hypotheses(hypothesis, evidence)
            output.subtype_hypotheses.extend(subtypes)
            
            if self.config.include_staging:
                stage = self.assess_stage(hypothesis, evidence)
                if stage:
                    output.stage_assessments.append(stage)
        
        # Generate differential
        if self.config.generate_differential:
            output.differential_diagnoses = self.generate_differential(
                output.primary_hypotheses, evidence
            )
        
        # Detect safety flags
        if self.config.detect_safety_flags:
            output.safety_flags = self.detect_safety_flags(
                evidence, output.primary_hypotheses
            )
        
        # Assess trajectory
        if self.config.include_trajectory and evidence_prior:
            output.trajectory_assessment = self.assess_trajectory(
                evidence, evidence_prior
            )
        
        logger.info(
            "Disease assessment complete",
            assessor=self.name,
            num_hypotheses=len(output.primary_hypotheses),
            num_safety_flags=len(output.safety_flags),
        )
        
        return output


class OrganSpecificAssessor(BaseDiseaseAssessor):
    """
    Base class for organ-specific disease assessors.
    
    Provides common patterns for organ-based assessment.
    """
    
    organ_name: str = ""
    
    def get_organ_features(
        self,
        evidence: CaseEvidenceBundle,
    ) -> dict[str, Any]:
        """Extract organ-specific features from evidence."""
        return evidence.organ_features.get(self.organ_name, {})
    
    def get_lesions_in_organ(
        self,
        evidence: CaseEvidenceBundle,
    ) -> list[dict[str, Any]]:
        """Extract lesions belonging to this organ."""
        return [
            lesion for lesion in evidence.lesion_features
            if lesion.get("organ") == self.organ_name
        ]
