# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Liver Lesion Disease Assessment
===============================

Disease-level reasoning for focal liver lesions on CT/MRI imaging.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

from rhenium.core.disease_types import (
    CaseEvidenceBundle,
    DiseasePresenceAssessment,
    DiseasePresenceStatus,
    DiseaseHypothesis,
    DiseaseSubtypeHypothesis,
    DiseaseStageAssessment,
    ClinicalSafetyFlag,
    ConfidenceLevel,
    SafetyFlagType,
    SafetyFlagSeverity,
)
from rhenium.disease.base import OrganSpecificAssessor, DiseaseAssessorConfig
from rhenium.core.logging import get_pipeline_logger

logger = get_pipeline_logger()


@dataclass
class LiverLesionConfig:
    """Configuration for liver lesion assessment."""
    min_lesion_size_mm: float = 5.0
    hcc_arterial_enhancement_required: bool = True
    hcc_venous_washout_required: bool = True


class LiverLesionAssessor(OrganSpecificAssessor):
    """
    Disease assessment for focal liver lesions.
    
    Generates disease hypotheses based on imaging characteristics
    (enhancement patterns, size, location) for common liver lesions.
    
    Note: All assessments are imaging-based surrogates and do not
    replace histopathologic diagnosis.
    """
    
    name = "liver_lesion_assessor"
    version = "1.0.0"
    organ_name = "liver"
    supported_diseases = ["hcc", "liver_metastasis", "hemangioma", "focal_nodular_hyperplasia", "hepatic_cyst"]
    supported_modalities = ["CT", "MR"]
    
    # LI-RADS-like category mapping (simplified)
    LIRADS_CATEGORIES = {
        "lr1": {"name": "Definitely Benign", "malignancy_prob": 0.0},
        "lr2": {"name": "Probably Benign", "malignancy_prob": 0.05},
        "lr3": {"name": "Intermediate Probability", "malignancy_prob": 0.30},
        "lr4": {"name": "Probably HCC", "malignancy_prob": 0.70},
        "lr5": {"name": "Definitely HCC", "malignancy_prob": 0.95},
        "lrm": {"name": "Probably Malignant, Not HCC Specific", "malignancy_prob": 0.85},
    }
    
    def __init__(
        self,
        config: DiseaseAssessorConfig | None = None,
        lesion_config: LiverLesionConfig | None = None,
    ):
        """Initialize with configuration."""
        super().__init__(config)
        self.lesion_config = lesion_config or LiverLesionConfig()
    
    def assess_presence(
        self,
        evidence: CaseEvidenceBundle,
    ) -> DiseasePresenceAssessment:
        """Assess presence of focal liver lesions."""
        assessment = DiseasePresenceAssessment(
            study_id=evidence.study_id,
            modality=evidence.metadata_features.get("modality", "CT"),
            organ_systems_involved=["liver"],
        )
        
        lesions = self._get_liver_lesions(evidence)
        significant = [
            l for l in lesions
            if l.get("diameter_mm", 0) >= self.lesion_config.min_lesion_size_mm
        ]
        
        if significant:
            assessment.disease_present = DiseasePresenceStatus.PRESENT
            assessment.evidence_ids = [l.get("id", "") for l in significant]
            assessment.rationale = f"Detected {len(significant)} focal liver lesion(s)."
            assessment.overall_abnormality_score = self._calculate_score(significant)
        else:
            assessment.disease_present = DiseasePresenceStatus.ABSENT
            assessment.rationale = "No significant focal liver lesions detected."
        
        return assessment
    
    def generate_hypotheses(
        self,
        evidence: CaseEvidenceBundle,
    ) -> list[DiseaseHypothesis]:
        """Generate disease hypotheses for liver lesions."""
        hypotheses = []
        lesions = self._get_liver_lesions(evidence)
        
        for lesion in lesions:
            if lesion.get("diameter_mm", 0) < self.lesion_config.min_lesion_size_mm:
                continue
            
            # Assess enhancement pattern
            lirads = self._assess_lirads_category(lesion)
            
            # Generate appropriate hypothesis
            if lirads in ["lr4", "lr5"]:
                hyp = self._create_hcc_hypothesis(lesion, lirads)
            elif lirads == "lrm":
                hyp = self._create_malignancy_hypothesis(lesion)
            elif self._is_typical_hemangioma(lesion):
                hyp = self._create_hemangioma_hypothesis(lesion)
            elif self._is_typical_cyst(lesion):
                hyp = self._create_cyst_hypothesis(lesion)
            else:
                hyp = self._create_indeterminate_hypothesis(lesion, lirads)
            
            hypotheses.append(hyp)
        
        hypotheses.sort(key=lambda h: h.probability, reverse=True)
        for i, hyp in enumerate(hypotheses):
            hyp.rank = i + 1
        
        return hypotheses
    
    def generate_subtype_hypotheses(
        self,
        disease: DiseaseHypothesis,
        evidence: CaseEvidenceBundle,
    ) -> list[DiseaseSubtypeHypothesis]:
        """Generate subtype hypotheses for liver lesions."""
        subtypes = []
        
        if disease.disease_code == "hcc":
            subtypes.append(DiseaseSubtypeHypothesis(
                disease_code="hcc",
                subtype_code="typical_hcc",
                subtype_name="Typical HCC Pattern (arterial enhancement with washout)",
                probability=0.9,
                imaging_features=[
                    "Arterial phase hyperenhancement",
                    "Venous phase washout",
                    "Capsule appearance"
                ],
            ))
        
        return subtypes
    
    def assess_stage(
        self,
        disease: DiseaseHypothesis,
        evidence: CaseEvidenceBundle,
    ) -> Optional[DiseaseStageAssessment]:
        """Assess staging for HCC using size criteria."""
        if disease.disease_code != "hcc":
            return None
        
        lesion = self._find_lesion_for_hypothesis(disease, evidence)
        if not lesion:
            return None
        
        diameter = lesion.get("diameter_mm", 0)
        
        # BCLC-like staging based on size (simplified surrogate)
        if diameter <= 20:
            stage = "Very Early (0) surrogate"
        elif diameter <= 50:
            stage = "Early (A) surrogate"
        else:
            stage = "Intermediate (B) or Advanced surrogate"
        
        return DiseaseStageAssessment(
            disease_code="hcc",
            staging_system="BCLC-like imaging surrogate",
            stage_label=stage,
            evidence_ids=disease.evidence_ids,
            assumptions=[
                "Size-based staging only",
                "Performance status and liver function not assessed",
            ],
            limitations=[
                "Portal vein invasion not assessed systematically",
                "Extrahepatic spread requires additional evaluation",
            ],
        )
    
    def detect_safety_flags(
        self,
        evidence: CaseEvidenceBundle,
        hypotheses: list[DiseaseHypothesis],
    ) -> list[ClinicalSafetyFlag]:
        """Detect safety flags for liver lesions."""
        flags = []
        
        for hyp in hypotheses:
            if hyp.disease_code == "hcc" and hyp.probability >= 0.7:
                flags.append(ClinicalSafetyFlag(
                    flag_type=SafetyFlagType.URGENT_FINDING,
                    severity=SafetyFlagSeverity.HIGH,
                    description="High probability of hepatocellular carcinoma",
                    finding_pattern="probable_hcc",
                    evidence_ids=hyp.evidence_ids,
                    recommended_action="Multidisciplinary liver tumor board review recommended",
                ))
        
        return flags
    
    def _get_liver_lesions(self, evidence: CaseEvidenceBundle) -> list[dict[str, Any]]:
        """Extract liver lesion features."""
        return [l for l in evidence.lesion_features if l.get("organ") == "liver"]
    
    def _assess_lirads_category(self, lesion: dict[str, Any]) -> str:
        """Assess LI-RADS-like category based on enhancement."""
        arterial = lesion.get("arterial_enhancement", False)
        washout = lesion.get("venous_washout", False)
        capsule = lesion.get("capsule", False)
        diameter = lesion.get("diameter_mm", 0)
        
        if arterial and washout:
            if diameter >= 20 or capsule:
                return "lr5"
            elif diameter >= 10:
                return "lr4"
        
        if arterial and not washout:
            return "lr3"
        
        return "lr2"
    
    def _is_typical_hemangioma(self, lesion: dict[str, Any]) -> bool:
        """Check for typical hemangioma pattern."""
        return (
            lesion.get("peripheral_nodular_enhancement", False) and
            lesion.get("progressive_fill_in", False)
        )
    
    def _is_typical_cyst(self, lesion: dict[str, Any]) -> bool:
        """Check for simple cyst criteria."""
        return (
            lesion.get("attenuation_hu", 100) < 20 and
            not lesion.get("enhancement", True)
        )
    
    def _calculate_score(self, lesions: list[dict[str, Any]]) -> float:
        """Calculate overall abnormality score."""
        if not lesions:
            return 0.0
        scores = []
        for l in lesions:
            cat = self._assess_lirads_category(l)
            prob = self.LIRADS_CATEGORIES.get(cat, {}).get("malignancy_prob", 0.3)
            scores.append(prob)
        return max(scores)
    
    def _create_hcc_hypothesis(self, lesion: dict, lirads: str) -> DiseaseHypothesis:
        """Create HCC hypothesis."""
        prob = self.LIRADS_CATEGORIES[lirads]["malignancy_prob"]
        return DiseaseHypothesis(
            disease_code="hcc",
            disease_name="Hepatocellular Carcinoma",
            probability=prob,
            evidence_ids=[lesion.get("id", "")],
            supporting_features=[
                "Arterial phase hyperenhancement",
                "Venous phase washout",
                f"LI-RADS category: {lirads.upper()}",
            ],
            confidence_level=ConfidenceLevel.HIGH if prob >= 0.9 else ConfidenceLevel.MEDIUM,
        )
    
    def _create_malignancy_hypothesis(self, lesion: dict) -> DiseaseHypothesis:
        """Create non-HCC malignancy hypothesis."""
        return DiseaseHypothesis(
            disease_code="liver_metastasis",
            disease_name="Probable Hepatic Malignancy (non-HCC specific)",
            probability=0.85,
            evidence_ids=[lesion.get("id", "")],
            supporting_features=["Atypical enhancement pattern for HCC"],
            confidence_level=ConfidenceLevel.MEDIUM,
        )
    
    def _create_hemangioma_hypothesis(self, lesion: dict) -> DiseaseHypothesis:
        """Create hemangioma hypothesis."""
        return DiseaseHypothesis(
            disease_code="hemangioma",
            disease_name="Hepatic Hemangioma",
            probability=0.90,
            evidence_ids=[lesion.get("id", "")],
            supporting_features=[
                "Peripheral nodular enhancement",
                "Progressive centripetal fill-in",
            ],
            confidence_level=ConfidenceLevel.HIGH,
        )
    
    def _create_cyst_hypothesis(self, lesion: dict) -> DiseaseHypothesis:
        """Create simple cyst hypothesis."""
        return DiseaseHypothesis(
            disease_code="hepatic_cyst",
            disease_name="Simple Hepatic Cyst",
            probability=0.95,
            evidence_ids=[lesion.get("id", "")],
            supporting_features=[
                "Water attenuation",
                "No enhancement",
                "Smooth margins",
            ],
            confidence_level=ConfidenceLevel.HIGH,
        )
    
    def _create_indeterminate_hypothesis(self, lesion: dict, lirads: str) -> DiseaseHypothesis:
        """Create indeterminate lesion hypothesis."""
        prob = self.LIRADS_CATEGORIES.get(lirads, {}).get("malignancy_prob", 0.3)
        return DiseaseHypothesis(
            disease_code="indeterminate_liver_lesion",
            disease_name="Indeterminate Focal Liver Lesion",
            probability=prob,
            evidence_ids=[lesion.get("id", "")],
            confidence_level=ConfidenceLevel.LOW,
        )
    
    def _find_lesion_for_hypothesis(self, hyp: DiseaseHypothesis, evidence: CaseEvidenceBundle):
        """Find lesion for hypothesis."""
        if not hyp.evidence_ids:
            return None
        target = hyp.evidence_ids[0]
        for l in self._get_liver_lesions(evidence):
            if l.get("id") == target:
                return l
        return None
