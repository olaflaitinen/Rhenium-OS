# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Brain Lesion Disease Assessment
===============================

Disease-level reasoning for intracranial lesions on CT/MRI imaging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from rhenium.core.disease_types import (
    CaseEvidenceBundle,
    DiseasePresenceAssessment,
    DiseasePresenceStatus,
    DiseaseHypothesis,
    DiseaseStageAssessment,
    ClinicalSafetyFlag,
    ConfidenceLevel,
    SafetyFlagType,
    SafetyFlagSeverity,
)
from rhenium.disease.base import OrganSpecificAssessor, DiseaseAssessorConfig
from rhenium.core.logging import get_pipeline_logger

logger = get_pipeline_logger()


class BrainLesionAssessor(OrganSpecificAssessor):
    """
    Disease assessment for intracranial lesions.
    
    Generates disease hypotheses for brain tumors, hemorrhage,
    demyelinating lesions, and ischemic changes.
    """
    
    name = "brain_lesion_assessor"
    version = "1.0.0"
    organ_name = "brain"
    supported_diseases = [
        "intracranial_hemorrhage",
        "brain_tumor",
        "brain_metastasis",
        "ischemic_stroke",
        "white_matter_disease",
    ]
    supported_modalities = ["CT", "MR"]
    
    def assess_presence(
        self,
        evidence: CaseEvidenceBundle,
    ) -> DiseasePresenceAssessment:
        """Assess presence of brain lesions."""
        assessment = DiseasePresenceAssessment(
            study_id=evidence.study_id,
            modality=evidence.metadata_features.get("modality", "CT"),
            organ_systems_involved=["brain"],
        )
        
        lesions = self._get_brain_lesions(evidence)
        
        if lesions:
            assessment.disease_present = DiseasePresenceStatus.PRESENT
            assessment.evidence_ids = [l.get("id", "") for l in lesions]
            assessment.rationale = f"Detected {len(lesions)} intracranial lesion(s)."
        else:
            assessment.disease_present = DiseasePresenceStatus.ABSENT
            assessment.rationale = "No focal intracranial lesions detected."
        
        return assessment
    
    def generate_hypotheses(
        self,
        evidence: CaseEvidenceBundle,
    ) -> list[DiseaseHypothesis]:
        """Generate disease hypotheses for brain lesions."""
        hypotheses = []
        lesions = self._get_brain_lesions(evidence)
        
        for lesion in lesions:
            lesion_type = lesion.get("type", "unknown")
            
            if lesion_type == "hemorrhage" or lesion.get("is_hemorrhage", False):
                hyp = self._create_hemorrhage_hypothesis(lesion)
            elif lesion_type == "mass" or lesion.get("mass_effect", False):
                hyp = self._create_tumor_hypothesis(lesion)
            elif lesion_type == "ischemic":
                hyp = self._create_stroke_hypothesis(lesion)
            elif lesion_type == "white_matter":
                hyp = self._create_white_matter_hypothesis(lesion)
            else:
                hyp = self._create_generic_hypothesis(lesion)
            
            hypotheses.append(hyp)
        
        hypotheses.sort(key=lambda h: h.probability, reverse=True)
        for i, hyp in enumerate(hypotheses):
            hyp.rank = i + 1
        
        return hypotheses
    
    def detect_safety_flags(
        self,
        evidence: CaseEvidenceBundle,
        hypotheses: list[DiseaseHypothesis],
    ) -> list[ClinicalSafetyFlag]:
        """Detect urgent neurological findings."""
        flags = []
        
        for hyp in hypotheses:
            if hyp.disease_code == "intracranial_hemorrhage":
                flags.append(ClinicalSafetyFlag(
                    flag_type=SafetyFlagType.URGENT_FINDING,
                    severity=SafetyFlagSeverity.CRITICAL,
                    description="Intracranial hemorrhage detected",
                    evidence_ids=hyp.evidence_ids,
                    recommended_action="Immediate neurosurgical evaluation",
                    escalation_required=True,
                    time_sensitivity="immediate",
                ))
            
            if hyp.disease_code == "ischemic_stroke":
                flags.append(ClinicalSafetyFlag(
                    flag_type=SafetyFlagType.URGENT_FINDING,
                    severity=SafetyFlagSeverity.CRITICAL,
                    description="Acute ischemic changes detected",
                    evidence_ids=hyp.evidence_ids,
                    recommended_action="Stroke team activation if clinically appropriate",
                    escalation_required=True,
                    time_sensitivity="immediate",
                ))
        
        # Check for mass effect
        for lesion in evidence.lesion_features:
            midline_shift = lesion.get("midline_shift_mm", 0)
            if midline_shift >= 5:
                flags.append(ClinicalSafetyFlag(
                    flag_type=SafetyFlagType.URGENT_FINDING,
                    severity=SafetyFlagSeverity.CRITICAL,
                    description=f"Significant midline shift ({midline_shift:.1f}mm)",
                    evidence_ids=[lesion.get("id", "")],
                    recommended_action="Urgent neurosurgical consultation",
                    escalation_required=True,
                ))
        
        return flags
    
    def _get_brain_lesions(self, evidence: CaseEvidenceBundle) -> list[dict[str, Any]]:
        """Extract brain lesion features."""
        return [l for l in evidence.lesion_features if l.get("organ") == "brain"]
    
    def _create_hemorrhage_hypothesis(self, lesion: dict) -> DiseaseHypothesis:
        """Create hemorrhage hypothesis."""
        hemorrhage_type = lesion.get("hemorrhage_type", "parenchymal")
        volume = lesion.get("volume_ml", 0)
        
        return DiseaseHypothesis(
            disease_code="intracranial_hemorrhage",
            disease_name=f"Intracranial Hemorrhage ({hemorrhage_type})",
            probability=0.95,
            evidence_ids=[lesion.get("id", "")],
            supporting_features=[
                f"Hemorrhage type: {hemorrhage_type}",
                f"Volume: {volume:.1f}mL" if volume else "Volume not calculated",
                f"Location: {lesion.get('location', 'unspecified')}",
            ],
            confidence_level=ConfidenceLevel.HIGH,
        )
    
    def _create_tumor_hypothesis(self, lesion: dict) -> DiseaseHypothesis:
        """Create brain tumor hypothesis."""
        enhancing = lesion.get("enhancing", False)
        
        if lesion.get("multiple", False):
            name = "Brain Metastases"
            code = "brain_metastasis"
            prob = 0.85
        else:
            name = "Primary Brain Tumor"
            code = "brain_tumor"
            prob = 0.75
        
        return DiseaseHypothesis(
            disease_code=code,
            disease_name=name,
            probability=prob,
            evidence_ids=[lesion.get("id", "")],
            supporting_features=[
                f"Size: {lesion.get('diameter_mm', 0):.1f}mm",
                "Enhancement present" if enhancing else "Non-enhancing",
                "Mass effect" if lesion.get("mass_effect") else "No mass effect",
            ],
            confidence_level=ConfidenceLevel.MEDIUM,
        )
    
    def _create_stroke_hypothesis(self, lesion: dict) -> DiseaseHypothesis:
        """Create ischemic stroke hypothesis."""
        territory = lesion.get("vascular_territory", "unspecified")
        acuity = lesion.get("acuity", "acute")
        
        return DiseaseHypothesis(
            disease_code="ischemic_stroke",
            disease_name=f"Ischemic Infarct ({acuity})",
            probability=0.90,
            evidence_ids=[lesion.get("id", "")],
            supporting_features=[
                f"Vascular territory: {territory}",
                f"Acuity: {acuity}",
            ],
            confidence_level=ConfidenceLevel.HIGH,
        )
    
    def _create_white_matter_hypothesis(self, lesion: dict) -> DiseaseHypothesis:
        """Create white matter disease hypothesis."""
        burden = lesion.get("fazekas_grade", 1)
        
        return DiseaseHypothesis(
            disease_code="white_matter_disease",
            disease_name="White Matter Hyperintensities",
            probability=0.85,
            evidence_ids=[lesion.get("id", "")],
            supporting_features=[
                f"Fazekas grade: {burden}" if burden else "Grade not assessed",
                "Distribution: periventricular and deep white matter",
            ],
            confidence_level=ConfidenceLevel.HIGH,
            explanation_summary="Nonspecific white matter changes, may represent small vessel ischemic disease.",
        )
    
    def _create_generic_hypothesis(self, lesion: dict) -> DiseaseHypothesis:
        """Create generic intracranial lesion hypothesis."""
        return DiseaseHypothesis(
            disease_code="intracranial_lesion_nos",
            disease_name="Intracranial Lesion (Not Otherwise Specified)",
            probability=0.50,
            evidence_ids=[lesion.get("id", "")],
            confidence_level=ConfidenceLevel.LOW,
        )
