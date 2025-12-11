# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Lung Nodule Disease Assessment
==============================

Disease-level reasoning for pulmonary nodules detected on CT imaging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
class LungNoduleAssessmentConfig:
    """Configuration for lung nodule assessment."""
    # Size thresholds (mm)
    min_nodule_size_mm: float = 4.0
    high_risk_size_mm: float = 8.0
    large_nodule_size_mm: float = 30.0
    
    # Malignancy risk thresholds
    low_risk_threshold: float = 0.05
    intermediate_risk_threshold: float = 0.10
    high_risk_threshold: float = 0.30
    
    # Enable features
    assess_morphology_subtypes: bool = True
    assess_lung_rads: bool = True


class LungNoduleAssessor(OrganSpecificAssessor):
    """
    Disease assessment for pulmonary nodules.
    
    Generates disease hypotheses, subtypes (solid, part-solid, GGN),
    and risk-based staging for lung nodules.
    
    Note: All assessments are imaging-based surrogates and do not
    replace histopathologic diagnosis.
    """
    
    name = "lung_nodule_assessor"
    version = "1.0.0"
    organ_name = "lung"
    supported_diseases = ["lung_nodule", "primary_lung_malignancy", "pulmonary_metastasis"]
    supported_modalities = ["CT"]
    
    # Malignancy risk by size (simplified Brock/McWilliams-like approximation)
    SIZE_RISK_TABLE: list[tuple[float, float, float]] = [
        (0, 4, 0.005),      # <4mm: ~0.5%
        (4, 6, 0.01),       # 4-6mm: ~1%
        (6, 8, 0.02),       # 6-8mm: ~2%
        (8, 15, 0.10),      # 8-15mm: ~10%
        (15, 30, 0.30),     # 15-30mm: ~30%
        (30, float('inf'), 0.60),  # >30mm: ~60%
    ]
    
    # Morphology risk modifiers
    MORPHOLOGY_MODIFIERS: dict[str, float] = {
        "solid": 1.0,
        "part_solid": 1.5,
        "ground_glass": 0.5,
        "spiculated": 2.0,
        "calcified": 0.1,
        "smooth": 0.7,
    }
    
    def __init__(
        self,
        config: DiseaseAssessorConfig | None = None,
        nodule_config: LungNoduleAssessmentConfig | None = None,
    ):
        """Initialize with configuration."""
        super().__init__(config)
        self.nodule_config = nodule_config or LungNoduleAssessmentConfig()
    
    def assess_presence(
        self,
        evidence: CaseEvidenceBundle,
    ) -> DiseasePresenceAssessment:
        """Assess presence of pulmonary nodules."""
        assessment = DiseasePresenceAssessment(
            study_id=evidence.study_id,
            modality="CT",
            organ_systems_involved=["lung"],
        )
        
        # Get lung-related lesions
        nodules = self._get_lung_nodules(evidence)
        
        if not nodules:
            assessment.disease_present = DiseasePresenceStatus.ABSENT
            assessment.rationale = "No pulmonary nodules detected above size threshold."
            return assessment
        
        # Filter by size threshold
        significant_nodules = [
            n for n in nodules
            if n.get("diameter_mm", 0) >= self.nodule_config.min_nodule_size_mm
        ]
        
        if significant_nodules:
            assessment.disease_present = DiseasePresenceStatus.PRESENT
            assessment.overall_abnormality_score = self._calculate_overall_score(
                significant_nodules
            )
            assessment.evidence_ids = [n.get("id", "") for n in significant_nodules]
            assessment.rationale = (
                f"Detected {len(significant_nodules)} pulmonary nodule(s) "
                f"measuring >= {self.nodule_config.min_nodule_size_mm}mm."
            )
        else:
            assessment.disease_present = DiseasePresenceStatus.ABSENT
            assessment.rationale = (
                f"All detected nodules < {self.nodule_config.min_nodule_size_mm}mm "
                "threshold."
            )
        
        return assessment
    
    def generate_hypotheses(
        self,
        evidence: CaseEvidenceBundle,
    ) -> list[DiseaseHypothesis]:
        """Generate disease hypotheses for each significant nodule."""
        hypotheses = []
        nodules = self._get_lung_nodules(evidence)
        
        for nodule in nodules:
            diameter = nodule.get("diameter_mm", 0)
            if diameter < self.nodule_config.min_nodule_size_mm:
                continue
            
            # Calculate malignancy probability
            malignancy_prob = self._calculate_malignancy_risk(nodule)
            
            # Generate primary hypothesis
            if malignancy_prob >= self.nodule_config.high_risk_threshold:
                hyp = self._create_malignancy_hypothesis(nodule, malignancy_prob)
            else:
                hyp = self._create_indeterminate_hypothesis(nodule, malignancy_prob)
            
            hypotheses.append(hyp)
        
        # Sort by probability
        hypotheses.sort(key=lambda h: h.probability, reverse=True)
        
        # Assign ranks
        for i, hyp in enumerate(hypotheses):
            hyp.rank = i + 1
        
        return hypotheses
    
    def generate_subtype_hypotheses(
        self,
        disease: DiseaseHypothesis,
        evidence: CaseEvidenceBundle,
    ) -> list[DiseaseSubtypeHypothesis]:
        """Generate morphology-based subtype hypotheses."""
        if not self.nodule_config.assess_morphology_subtypes:
            return []
        
        subtypes = []
        
        # Find associated nodule
        nodule = self._find_nodule_for_hypothesis(disease, evidence)
        if not nodule:
            return []
        
        morphology = nodule.get("morphology", "solid")
        
        # Solid nodule subtype
        if morphology == "solid":
            subtypes.append(DiseaseSubtypeHypothesis(
                disease_code=disease.disease_code,
                subtype_code="solid_nodule",
                subtype_name="Solid Pulmonary Nodule",
                probability=0.95,
                imaging_features=["Homogeneous soft tissue attenuation"],
            ))
        
        # Part-solid nodule subtype
        elif morphology == "part_solid":
            subtypes.append(DiseaseSubtypeHypothesis(
                disease_code=disease.disease_code,
                subtype_code="part_solid_nodule",
                subtype_name="Part-Solid (Subsolid) Pulmonary Nodule",
                probability=0.95,
                imaging_features=[
                    "Mixed ground-glass and solid components",
                    "Higher malignancy risk if solid component >5mm"
                ],
            ))
        
        # Ground-glass nodule subtype
        elif morphology == "ground_glass":
            subtypes.append(DiseaseSubtypeHypothesis(
                disease_code=disease.disease_code,
                subtype_code="pure_ggn",
                subtype_name="Pure Ground-Glass Nodule",
                probability=0.95,
                imaging_features=[
                    "Ground-glass opacity without solid component",
                    "May represent atypical adenomatous hyperplasia or adenocarcinoma in situ"
                ],
            ))
        
        return subtypes
    
    def assess_stage(
        self,
        disease: DiseaseHypothesis,
        evidence: CaseEvidenceBundle,
    ) -> Optional[DiseaseStageAssessment]:
        """Assess T-stage surrogate based on nodule size."""
        nodule = self._find_nodule_for_hypothesis(disease, evidence)
        if not nodule:
            return None
        
        diameter = nodule.get("diameter_mm", 0)
        
        # Size-based T-stage surrogate (per TNM 8th edition approximation)
        if diameter <= 10:
            t_stage = "T1a (surrogate)"
            stage_label = "Stage IA1-IA2 (surrogate)"
        elif diameter <= 20:
            t_stage = "T1b (surrogate)"
            stage_label = "Stage IA3 (surrogate)"
        elif diameter <= 30:
            t_stage = "T1c (surrogate)"
            stage_label = "Stage IB (surrogate)"
        elif diameter <= 40:
            t_stage = "T2a (surrogate)"
            stage_label = "Stage IB-IIA (surrogate)"
        elif diameter <= 50:
            t_stage = "T2b (surrogate)"
            stage_label = "Stage IIA (surrogate)"
        else:
            t_stage = "T3+ (surrogate)"
            stage_label = "Stage IIB+ (surrogate)"
        
        return DiseaseStageAssessment(
            disease_code=disease.disease_code,
            staging_system="TNM 8th edition (imaging surrogate)",
            stage_label=stage_label,
            t_component=t_stage,
            n_component="Nx (nodal status not assessed)",
            m_component="Mx (metastatic status not assessed)",
            evidence_ids=disease.evidence_ids,
            assumptions=[
                "Size-based T-staging using imaging measurements only",
                "Histologic confirmation required for definitive staging",
            ],
            limitations=[
                "N and M components not assessed on this imaging alone",
                "Does not account for local invasion or pleural involvement",
            ],
        )
    
    def detect_safety_flags(
        self,
        evidence: CaseEvidenceBundle,
        hypotheses: list[DiseaseHypothesis],
    ) -> list[ClinicalSafetyFlag]:
        """Detect safety flags for lung nodules."""
        flags = []
        
        for hyp in hypotheses:
            # High-probability malignancy flag
            if hyp.probability >= 0.5:
                flags.append(ClinicalSafetyFlag(
                    flag_type=SafetyFlagType.URGENT_FINDING,
                    severity=SafetyFlagSeverity.HIGH,
                    description=f"Pulmonary nodule with estimated malignancy probability {hyp.probability:.0%}",
                    finding_pattern="high_risk_nodule",
                    evidence_ids=hyp.evidence_ids,
                    recommended_action="Recommend multidisciplinary review and consideration for tissue diagnosis",
                ))
        
        # Check for large nodule
        nodules = self._get_lung_nodules(evidence)
        for nodule in nodules:
            if nodule.get("diameter_mm", 0) >= self.nodule_config.large_nodule_size_mm:
                flags.append(ClinicalSafetyFlag(
                    flag_type=SafetyFlagType.URGENT_FINDING,
                    severity=SafetyFlagSeverity.HIGH,
                    description=f"Large pulmonary nodule/mass (>={self.nodule_config.large_nodule_size_mm}mm)",
                    finding_pattern="large_pulmonary_mass",
                    evidence_ids=[nodule.get("id", "")],
                    recommended_action="Expedited workup recommended",
                ))
        
        return flags
    
    def _get_lung_nodules(
        self,
        evidence: CaseEvidenceBundle,
    ) -> list[dict[str, Any]]:
        """Extract lung nodule features from evidence."""
        return [
            l for l in evidence.lesion_features
            if l.get("organ") == "lung" and l.get("type") in ["nodule", "mass"]
        ]
    
    def _calculate_malignancy_risk(
        self,
        nodule: dict[str, Any],
    ) -> float:
        """Calculate malignancy risk based on size and morphology."""
        diameter = nodule.get("diameter_mm", 0)
        
        # Base risk from size
        base_risk = 0.01
        for min_size, max_size, risk in self.SIZE_RISK_TABLE:
            if min_size <= diameter < max_size:
                base_risk = risk
                break
        
        # Apply morphology modifier
        morphology = nodule.get("morphology", "solid")
        modifier = self.MORPHOLOGY_MODIFIERS.get(morphology, 1.0)
        
        return min(base_risk * modifier, 0.99)
    
    def _calculate_overall_score(
        self,
        nodules: list[dict[str, Any]],
    ) -> float:
        """Calculate overall abnormality score from nodules."""
        if not nodules:
            return 0.0
        
        max_risk = max(self._calculate_malignancy_risk(n) for n in nodules)
        return max_risk
    
    def _create_malignancy_hypothesis(
        self,
        nodule: dict[str, Any],
        probability: float,
    ) -> DiseaseHypothesis:
        """Create hypothesis for suspected malignancy."""
        features = self._extract_nodule_features(nodule)
        
        return DiseaseHypothesis(
            disease_code="primary_lung_malignancy",
            disease_name="Suspected Primary Lung Malignancy",
            probability=probability,
            evidence_ids=[nodule.get("id", "")],
            supporting_features=features["supporting"],
            contradicting_features=features["contradicting"],
            confidence_level=(
                ConfidenceLevel.HIGH if probability >= 0.5
                else ConfidenceLevel.MEDIUM
            ),
            explanation_summary=(
                f"Pulmonary nodule {nodule.get('diameter_mm', 0):.1f}mm with "
                f"estimated malignancy probability {probability:.0%} based on "
                "size and morphology features."
            ),
        )
    
    def _create_indeterminate_hypothesis(
        self,
        nodule: dict[str, Any],
        probability: float,
    ) -> DiseaseHypothesis:
        """Create hypothesis for indeterminate nodule."""
        return DiseaseHypothesis(
            disease_code="indeterminate_pulmonary_nodule",
            disease_name="Indeterminate Pulmonary Nodule",
            probability=probability,
            evidence_ids=[nodule.get("id", "")],
            supporting_features=[
                f"Size: {nodule.get('diameter_mm', 0):.1f}mm",
                f"Morphology: {nodule.get('morphology', 'solid')}",
            ],
            confidence_level=ConfidenceLevel.MEDIUM,
            explanation_summary=(
                f"Pulmonary nodule {nodule.get('diameter_mm', 0):.1f}mm with "
                f"indeterminate features. Malignancy probability estimated at {probability:.0%}."
            ),
        )
    
    def _extract_nodule_features(
        self,
        nodule: dict[str, Any],
    ) -> dict[str, list[str]]:
        """Extract supporting and contradicting features."""
        supporting = []
        contradicting = []
        
        diameter = nodule.get("diameter_mm", 0)
        morphology = nodule.get("morphology", "solid")
        
        # Size-based features
        if diameter >= 15:
            supporting.append(f"Size {diameter:.1f}mm (>15mm concerning)")
        elif diameter < 6:
            contradicting.append(f"Small size ({diameter:.1f}mm)")
        
        # Morphology features
        if morphology == "spiculated":
            supporting.append("Spiculated margins (concerning)")
        elif morphology == "calcified":
            contradicting.append("Calcification (favors benign)")
        elif morphology == "smooth":
            contradicting.append("Smooth margins (favors benign)")
        
        return {"supporting": supporting, "contradicting": contradicting}
    
    def _find_nodule_for_hypothesis(
        self,
        hypothesis: DiseaseHypothesis,
        evidence: CaseEvidenceBundle,
    ) -> Optional[dict[str, Any]]:
        """Find the nodule associated with a hypothesis."""
        if not hypothesis.evidence_ids:
            return None
        
        target_id = hypothesis.evidence_ids[0]
        for nodule in self._get_lung_nodules(evidence):
            if nodule.get("id") == target_id:
                return nodule
        
        return None
