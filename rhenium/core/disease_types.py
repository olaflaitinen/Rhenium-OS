# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Rhenium OS Disease Reasoning Data Models
=========================================

Structured definitions for disease-level clinical reasoning outputs.

This module provides Pydantic-based data models for representing:
- Disease presence/absence assessments
- Disease hypotheses with probability scores
- Disease subtype classification
- Staging and severity assessments
- Differential diagnoses
- Longitudinal trajectory assessments
- Clinical safety flags

All disease assessments are image-based surrogates and do not replace
clinical diagnosis by qualified medical professionals.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional
from uuid import uuid4


# =============================================================================
# Enumerations
# =============================================================================

class DiseasePresenceStatus(str, Enum):
    """Disease presence determination."""
    PRESENT = "present"
    ABSENT = "absent"
    UNCERTAIN = "uncertain"
    NOT_EVALUATED = "not_evaluated"


class ConfidenceLevel(str, Enum):
    """Confidence level for assessments."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    NOT_APPLICABLE = "not_applicable"


class TrajectoryLabel(str, Enum):
    """Disease trajectory over time."""
    IMPROVED = "improved"
    STABLE = "stable"
    WORSENED = "worsened"
    NEW_FINDINGS = "new_findings"
    INDETERMINATE = "indeterminate"
    NOT_APPLICABLE = "not_applicable"


class SafetyFlagSeverity(str, Enum):
    """Severity of clinical safety flags."""
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"


class SafetyFlagType(str, Enum):
    """Types of clinical safety flags."""
    URGENT_FINDING = "urgent_finding"
    LIMITED_IMAGE_QUALITY = "limited_image_quality"
    POTENTIALLY_UNSAFE_DECISION = "potentially_unsafe_decision"
    REQUIRES_ESCALATION = "requires_escalation"
    CONTRAINDICATION = "contraindication"
    INCIDENTAL_CRITICAL = "incidental_critical"


class PrognosisCategory(str, Enum):
    """Prognosis risk categories based on imaging surrogates."""
    EXCELLENT = "excellent"
    GOOD = "good"
    INTERMEDIATE = "intermediate"
    POOR = "poor"
    VERY_POOR = "very_poor"
    INDETERMINATE = "indeterminate"


class TreatmentResponseCategory(str, Enum):
    """Treatment response categories (RECIST-like)."""
    COMPLETE_RESPONSE = "complete_response"
    PARTIAL_RESPONSE = "partial_response"
    STABLE_DISEASE = "stable_disease"
    PROGRESSIVE_DISEASE = "progressive_disease"
    NOT_EVALUABLE = "not_evaluable"


class RiskCategory(str, Enum):
    """Multi-factor risk stratification categories."""
    VERY_LOW = "very_low"
    LOW = "low"
    INTERMEDIATE = "intermediate"
    HIGH = "high"
    VERY_HIGH = "very_high"


class ActionUrgency(str, Enum):
    """Urgency levels for clinical actions."""
    ROUTINE = "routine"
    SOON = "soon"
    URGENT = "urgent"
    EMERGENT = "emergent"


class ActionType(str, Enum):
    """Types of recommended clinical actions."""
    FOLLOW_UP_IMAGING = "follow_up_imaging"
    TISSUE_SAMPLING = "tissue_sampling"
    SPECIALIST_REFERRAL = "specialist_referral"
    INTERVENTION = "intervention"
    MEDICAL_THERAPY = "medical_therapy"
    SURVEILLANCE = "surveillance"
    NO_ACTION = "no_action"


class QualityGrade(str, Enum):
    """Image/analysis quality grades."""
    EXCELLENT = "excellent"
    GOOD = "good"
    ADEQUATE = "adequate"
    LIMITED = "limited"
    NON_DIAGNOSTIC = "non_diagnostic"


# =============================================================================
# Core Disease Assessment Models
# =============================================================================

@dataclass
class DiseasePresenceAssessment:
    """
    Assessment of whether any disease is present in a study.
    
    This is the first-level question: "Is there evidence of disease?"
    
    Attributes:
        assessment_id: Unique identifier for this assessment.
        study_id: Reference to the imaging study.
        modality: Imaging modality (MRI, CT, X-ray, Ultrasound).
        organ_systems_involved: List of organ systems evaluated.
        disease_present: Overall disease presence status.
        overall_abnormality_score: Aggregate abnormality score (0.0-1.0).
        evidence_ids: References to supporting evidence (lesions, regions).
        uncertainty_score: Uncertainty in the assessment (0.0-1.0).
        rationale: Brief explanation of the assessment.
        limitations: List of assessment limitations.
        created_at: Assessment timestamp.
    """
    assessment_id: str = field(default_factory=lambda: uuid4().hex[:12])
    study_id: str = ""
    modality: str = ""
    organ_systems_involved: list[str] = field(default_factory=list)
    disease_present: DiseasePresenceStatus = DiseasePresenceStatus.NOT_EVALUATED
    overall_abnormality_score: float = 0.0
    evidence_ids: list[str] = field(default_factory=list)
    uncertainty_score: float = 0.0
    rationale: str = ""
    limitations: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "assessment_id": self.assessment_id,
            "study_id": self.study_id,
            "modality": self.modality,
            "organ_systems_involved": self.organ_systems_involved,
            "disease_present": self.disease_present.value,
            "overall_abnormality_score": self.overall_abnormality_score,
            "evidence_ids": self.evidence_ids,
            "uncertainty_score": self.uncertainty_score,
            "rationale": self.rationale,
            "limitations": self.limitations,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class DiseaseHypothesis:
    """
    A hypothesis about a specific disease based on imaging findings.
    
    Represents the answer to: "What disease is most likely?"
    
    Attributes:
        hypothesis_id: Unique identifier.
        disease_code: Standardized disease code (internal or SNOMED-like).
        disease_name: Human-readable disease name.
        probability: Estimated probability (0.0-1.0).
        rank: Rank among competing hypotheses (1 = most likely).
        evidence_ids: References to supporting imaging evidence.
        supporting_features: Features that support this diagnosis.
        contradicting_features: Features that argue against this diagnosis.
        confidence_level: Overall confidence in this hypothesis.
        explanation_summary: Brief explanation of reasoning.
        metadata: Additional metadata.
    """
    hypothesis_id: str = field(default_factory=lambda: uuid4().hex[:12])
    disease_code: str = ""
    disease_name: str = ""
    probability: float = 0.0
    rank: int = 0
    evidence_ids: list[str] = field(default_factory=list)
    supporting_features: list[str] = field(default_factory=list)
    contradicting_features: list[str] = field(default_factory=list)
    confidence_level: ConfidenceLevel = ConfidenceLevel.NOT_APPLICABLE
    explanation_summary: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "hypothesis_id": self.hypothesis_id,
            "disease_code": self.disease_code,
            "disease_name": self.disease_name,
            "probability": self.probability,
            "rank": self.rank,
            "evidence_ids": self.evidence_ids,
            "supporting_features": self.supporting_features,
            "contradicting_features": self.contradicting_features,
            "confidence_level": self.confidence_level.value,
            "explanation_summary": self.explanation_summary,
            "metadata": self.metadata,
        }


@dataclass
class DiseaseSubtypeHypothesis:
    """
    Hypothesis about disease subtype or variant.
    
    For a given disease, answers: "Which subtype is this?"
    
    Note: Subtypes are imaging-based phenotype surrogates, not
    histopathologic ground truth.
    
    Attributes:
        subtype_id: Unique identifier.
        disease_code: Parent disease code.
        subtype_code: Subtype identifier.
        subtype_name: Human-readable subtype name.
        probability: Probability of this subtype (0.0-1.0).
        evidence_ids: Supporting evidence references.
        imaging_features: Key imaging features suggesting this subtype.
        notes: Additional clinical notes.
    """
    subtype_id: str = field(default_factory=lambda: uuid4().hex[:12])
    disease_code: str = ""
    subtype_code: str = ""
    subtype_name: str = ""
    probability: float = 0.0
    evidence_ids: list[str] = field(default_factory=list)
    imaging_features: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "subtype_id": self.subtype_id,
            "disease_code": self.disease_code,
            "subtype_code": self.subtype_code,
            "subtype_name": self.subtype_name,
            "probability": self.probability,
            "evidence_ids": self.evidence_ids,
            "imaging_features": self.imaging_features,
            "notes": self.notes,
        }


@dataclass
class DiseaseStageAssessment:
    """
    Assessment of disease stage, grade, or severity.
    
    Provides image-based staging surrogates, not definitive clinical staging.
    
    Attributes:
        stage_id: Unique identifier.
        disease_code: Disease being staged.
        staging_system: Name of staging system used (e.g., "TNM-like surrogate").
        stage_label: Textual stage label (e.g., "Stage II", "Moderate").
        numeric_stage: Numeric stage value if applicable.
        t_component: Tumor (T) component for TNM-like systems.
        n_component: Node (N) component for TNM-like systems.
        m_component: Metastasis (M) component for TNM-like systems.
        severity_score: Continuous severity score (0.0-1.0).
        evidence_ids: Supporting evidence references.
        assumptions: Assumptions made in staging.
        limitations: Limitations of this staging assessment.
    """
    stage_id: str = field(default_factory=lambda: uuid4().hex[:12])
    disease_code: str = ""
    staging_system: str = ""
    stage_label: str = ""
    numeric_stage: Optional[int] = None
    t_component: str = ""
    n_component: str = ""
    m_component: str = ""
    severity_score: float = 0.0
    evidence_ids: list[str] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    limitations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "stage_id": self.stage_id,
            "disease_code": self.disease_code,
            "staging_system": self.staging_system,
            "stage_label": self.stage_label,
            "numeric_stage": self.numeric_stage,
            "t_component": self.t_component,
            "n_component": self.n_component,
            "m_component": self.m_component,
            "severity_score": self.severity_score,
            "evidence_ids": self.evidence_ids,
            "assumptions": self.assumptions,
            "limitations": self.limitations,
        }


@dataclass
class DifferentialDiagnosisEntry:
    """
    Entry in a differential diagnosis list.
    
    Provides alternative diagnoses with supporting and contradicting features.
    
    Attributes:
        entry_id: Unique identifier.
        rank: Position in differential (1 = most likely alternative).
        disease_code: Disease code for this differential.
        disease_name: Human-readable disease name.
        estimated_probability: Probability estimate (0.0-1.0).
        supporting_features: Features supporting this diagnosis.
        contradicting_features: Features arguing against this diagnosis.
        suggested_additional_tests: Recommended tests to clarify.
        cannot_exclude_flag: If True, indicates "cannot exclude" status.
        notes: Additional notes.
    """
    entry_id: str = field(default_factory=lambda: uuid4().hex[:12])
    rank: int = 0
    disease_code: str = ""
    disease_name: str = ""
    estimated_probability: float = 0.0
    supporting_features: list[str] = field(default_factory=list)
    contradicting_features: list[str] = field(default_factory=list)
    suggested_additional_tests: list[str] = field(default_factory=list)
    cannot_exclude_flag: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "entry_id": self.entry_id,
            "rank": self.rank,
            "disease_code": self.disease_code,
            "disease_name": self.disease_name,
            "estimated_probability": self.estimated_probability,
            "supporting_features": self.supporting_features,
            "contradicting_features": self.contradicting_features,
            "suggested_additional_tests": self.suggested_additional_tests,
            "cannot_exclude_flag": self.cannot_exclude_flag,
            "notes": self.notes,
        }


@dataclass
class DiseaseTrajectoryAssessment:
    """
    Assessment of disease trajectory over time (longitudinal comparison).
    
    Compares current study with prior imaging to assess disease course.
    
    Attributes:
        trajectory_id: Unique identifier.
        study_id_current: Current study reference.
        study_id_previous: Prior study reference for comparison.
        disease_code: Disease being tracked.
        trajectory_label: Overall trajectory determination.
        lesion_change_summaries: Per-lesion change descriptions.
        quantitative_deltas: Numeric changes (e.g., volume change percentage).
        new_lesion_count: Number of new lesions detected.
        resolved_lesion_count: Number of lesions that resolved.
        response_category: Treatment response category if applicable.
        timeline_days: Days between studies.
        timeline_comment: Narrative about temporal changes.
    """
    trajectory_id: str = field(default_factory=lambda: uuid4().hex[:12])
    study_id_current: str = ""
    study_id_previous: str = ""
    disease_code: str = ""
    trajectory_label: TrajectoryLabel = TrajectoryLabel.NOT_APPLICABLE
    lesion_change_summaries: list[str] = field(default_factory=list)
    quantitative_deltas: dict[str, float] = field(default_factory=dict)
    new_lesion_count: int = 0
    resolved_lesion_count: int = 0
    response_category: str = ""
    timeline_days: Optional[int] = None
    timeline_comment: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "trajectory_id": self.trajectory_id,
            "study_id_current": self.study_id_current,
            "study_id_previous": self.study_id_previous,
            "disease_code": self.disease_code,
            "trajectory_label": self.trajectory_label.value,
            "lesion_change_summaries": self.lesion_change_summaries,
            "quantitative_deltas": self.quantitative_deltas,
            "new_lesion_count": self.new_lesion_count,
            "resolved_lesion_count": self.resolved_lesion_count,
            "response_category": self.response_category,
            "timeline_days": self.timeline_days,
            "timeline_comment": self.timeline_comment,
        }


@dataclass
class ClinicalSafetyFlag:
    """
    Clinical safety flag for urgent or critical findings.
    
    Identifies patterns requiring immediate clinical attention or escalation.
    
    Attributes:
        flag_id: Unique identifier.
        flag_type: Category of safety concern.
        severity: Severity level (low to critical).
        description: Description of the safety concern.
        finding_pattern: Pattern that triggered this flag.
        evidence_ids: Supporting evidence references.
        recommended_action: Suggested clinical action.
        escalation_required: Whether immediate escalation is recommended.
        time_sensitivity: Time frame for action (e.g., "immediate", "24h").
    """
    flag_id: str = field(default_factory=lambda: uuid4().hex[:12])
    flag_type: SafetyFlagType = SafetyFlagType.URGENT_FINDING
    severity: SafetyFlagSeverity = SafetyFlagSeverity.MODERATE
    description: str = ""
    finding_pattern: str = ""
    evidence_ids: list[str] = field(default_factory=list)
    recommended_action: str = ""
    escalation_required: bool = False
    time_sensitivity: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "flag_id": self.flag_id,
            "flag_type": self.flag_type.value,
            "severity": self.severity.value,
            "description": self.description,
            "finding_pattern": self.finding_pattern,
            "evidence_ids": self.evidence_ids,
            "recommended_action": self.recommended_action,
            "escalation_required": self.escalation_required,
            "time_sensitivity": self.time_sensitivity,
        }


# =============================================================================
# Extended Clinical Assessment Models
# =============================================================================

@dataclass
class PrognosisAssessment:
    """
    Imaging-based prognosis assessment.
    
    Provides survival/outcome prediction surrogates based on imaging features.
    Formula: Prognosis Score = w1*tumor_burden + w2*invasion_score + w3*necrosis_ratio
    
    Attributes:
        assessment_id: Unique identifier.
        disease_code: Disease being prognosticated.
        prognosis_category: Risk category (excellent to very poor).
        prognosis_score: Continuous score (0.0-1.0, higher = worse).
        survival_estimate_months: Estimated survival in months (imaging surrogate).
        contributing_factors: Factors contributing to prognosis.
        confidence_interval: 95% CI for survival estimate.
        limitations: Assessment limitations.
    """
    assessment_id: str = field(default_factory=lambda: uuid4().hex[:12])
    disease_code: str = ""
    prognosis_category: PrognosisCategory = PrognosisCategory.INDETERMINATE
    prognosis_score: float = 0.0
    survival_estimate_months: Optional[float] = None
    contributing_factors: list[str] = field(default_factory=list)
    confidence_interval: tuple[float, float] = (0.0, 0.0)
    limitations: list[str] = field(default_factory=lambda: [
        "Imaging-based prognosis is a surrogate and requires clinical correlation."
    ])

    def to_dict(self) -> dict[str, Any]:
        return {
            "assessment_id": self.assessment_id,
            "disease_code": self.disease_code,
            "prognosis_category": self.prognosis_category.value,
            "prognosis_score": self.prognosis_score,
            "survival_estimate_months": self.survival_estimate_months,
            "contributing_factors": self.contributing_factors,
            "confidence_interval": list(self.confidence_interval),
            "limitations": self.limitations,
        }


@dataclass
class TreatmentResponseAssessment:
    """
    Treatment response evaluation (RECIST/mRECIST-like).
    
    Formula: Sum of Target Lesions Change = (Sum_current - Sum_baseline) / Sum_baseline * 100
    
    CR: Complete disappearance
    PR: >= 30% decrease
    PD: >= 20% increase or new lesions
    SD: Neither PR nor PD
    
    Attributes:
        assessment_id: Unique identifier.
        disease_code: Disease being evaluated.
        response_category: RECIST-like response category.
        target_lesion_sum_baseline_mm: Baseline sum of target lesions.
        target_lesion_sum_current_mm: Current sum of target lesions.
        percent_change: Percentage change from baseline.
        new_lesions_detected: Number of new lesions.
        non_target_lesion_status: Status of non-target lesions.
        overall_response: Final response determination.
    """
    assessment_id: str = field(default_factory=lambda: uuid4().hex[:12])
    disease_code: str = ""
    response_category: TreatmentResponseCategory = TreatmentResponseCategory.NOT_EVALUABLE
    target_lesion_sum_baseline_mm: float = 0.0
    target_lesion_sum_current_mm: float = 0.0
    percent_change: float = 0.0
    new_lesions_detected: int = 0
    non_target_lesion_status: str = ""
    overall_response: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "assessment_id": self.assessment_id,
            "disease_code": self.disease_code,
            "response_category": self.response_category.value,
            "target_lesion_sum_baseline_mm": self.target_lesion_sum_baseline_mm,
            "target_lesion_sum_current_mm": self.target_lesion_sum_current_mm,
            "percent_change": self.percent_change,
            "new_lesions_detected": self.new_lesions_detected,
            "non_target_lesion_status": self.non_target_lesion_status,
            "overall_response": self.overall_response,
        }


@dataclass
class ComorbidityAssessment:
    """
    Incidental comorbidity detection from imaging.
    
    Identifies conditions beyond the primary indication.
    
    Attributes:
        assessment_id: Unique identifier.
        comorbidities: List of detected comorbidities with severity.
        organ_systems_affected: Affected organ systems.
        clinical_significance: Overall clinical significance.
        recommended_actions: Recommended follow-up for each comorbidity.
    """
    assessment_id: str = field(default_factory=lambda: uuid4().hex[:12])
    comorbidities: list[dict[str, Any]] = field(default_factory=list)
    organ_systems_affected: list[str] = field(default_factory=list)
    clinical_significance: str = ""
    recommended_actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "assessment_id": self.assessment_id,
            "comorbidities": self.comorbidities,
            "organ_systems_affected": self.organ_systems_affected,
            "clinical_significance": self.clinical_significance,
            "recommended_actions": self.recommended_actions,
        }


@dataclass
class BiomarkerCorrelate:
    """
    Imaging-derived biomarker surrogates.
    
    Estimates biomarker values from imaging features using validated correlations.
    
    Example biomarkers: ADC for cellularity, SUV for metabolism, 
    enhancement patterns for vascularity.
    
    Attributes:
        correlate_id: Unique identifier.
        biomarker_name: Name of the biomarker.
        biomarker_code: Standardized biomarker code.
        estimated_value: Estimated biomarker value.
        unit: Measurement unit.
        confidence: Confidence in the estimate (0.0-1.0).
        imaging_features_used: Features used for estimation.
        correlation_method: Method used for correlation.
    """
    correlate_id: str = field(default_factory=lambda: uuid4().hex[:12])
    biomarker_name: str = ""
    biomarker_code: str = ""
    estimated_value: float = 0.0
    unit: str = ""
    confidence: float = 0.0
    imaging_features_used: list[str] = field(default_factory=list)
    correlation_method: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "correlate_id": self.correlate_id,
            "biomarker_name": self.biomarker_name,
            "biomarker_code": self.biomarker_code,
            "estimated_value": self.estimated_value,
            "unit": self.unit,
            "confidence": self.confidence,
            "imaging_features_used": self.imaging_features_used,
            "correlation_method": self.correlation_method,
        }


@dataclass
class GeneticCorrelateHypothesis:
    """
    Imaging-genomic correlation hypothesis.
    
    Estimates likelihood of genetic mutations based on imaging phenotypes.
    
    Example: Spiculated margins in lung nodules correlating with EGFR mutations.
    
    Attributes:
        hypothesis_id: Unique identifier.
        mutation_name: Name of the mutation.
        gene: Gene name.
        probability: Probability of mutation presence (0.0-1.0).
        imaging_phenotype: Imaging phenotype suggesting this mutation.
        supporting_features: Features supporting this hypothesis.
        literature_reference: Reference to supporting literature.
    """
    hypothesis_id: str = field(default_factory=lambda: uuid4().hex[:12])
    mutation_name: str = ""
    gene: str = ""
    probability: float = 0.0
    imaging_phenotype: str = ""
    supporting_features: list[str] = field(default_factory=list)
    literature_reference: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "mutation_name": self.mutation_name,
            "gene": self.gene,
            "probability": self.probability,
            "imaging_phenotype": self.imaging_phenotype,
            "supporting_features": self.supporting_features,
            "literature_reference": self.literature_reference,
        }


@dataclass
class RiskStratification:
    """
    Multi-factor risk stratification based on imaging.
    
    Formula: Risk Score = Σ(wi * fi) where wi = weight, fi = feature value
    
    Combines multiple imaging features for comprehensive risk assessment.
    
    Attributes:
        stratification_id: Unique identifier.
        disease_code: Disease for risk assessment.
        risk_category: Final risk category.
        risk_score: Continuous risk score (0.0-1.0).
        contributing_factors: Factors with their contributions.
        protective_factors: Factors reducing risk.
        risk_model_name: Name of the risk model used.
        recommendations: Risk-based recommendations.
    """
    stratification_id: str = field(default_factory=lambda: uuid4().hex[:12])
    disease_code: str = ""
    risk_category: RiskCategory = RiskCategory.INTERMEDIATE
    risk_score: float = 0.0
    contributing_factors: dict[str, float] = field(default_factory=dict)
    protective_factors: dict[str, float] = field(default_factory=dict)
    risk_model_name: str = ""
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stratification_id": self.stratification_id,
            "disease_code": self.disease_code,
            "risk_category": self.risk_category.value,
            "risk_score": self.risk_score,
            "contributing_factors": self.contributing_factors,
            "protective_factors": self.protective_factors,
            "risk_model_name": self.risk_model_name,
            "recommendations": self.recommendations,
        }


@dataclass
class ClinicalActionability:
    """
    Recommended clinical actions based on imaging findings.
    
    Provides actionable recommendations with urgency levels.
    
    Attributes:
        action_id: Unique identifier.
        action_type: Category of action.
        urgency: Urgency level.
        description: Description of the recommended action.
        rationale: Clinical rationale for this action.
        finding_ids: Findings triggering this action.
        contraindications: Any contraindications to consider.
        alternative_actions: Alternative actions if primary is not feasible.
    """
    action_id: str = field(default_factory=lambda: uuid4().hex[:12])
    action_type: ActionType = ActionType.NO_ACTION
    urgency: ActionUrgency = ActionUrgency.ROUTINE
    description: str = ""
    rationale: str = ""
    finding_ids: list[str] = field(default_factory=list)
    contraindications: list[str] = field(default_factory=list)
    alternative_actions: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type.value,
            "urgency": self.urgency.value,
            "description": self.description,
            "rationale": self.rationale,
            "finding_ids": self.finding_ids,
            "contraindications": self.contraindications,
            "alternative_actions": self.alternative_actions,
        }


@dataclass
class QualityMetrics:
    """
    Image and analysis quality metrics.
    
    Assesses quality of input images and analysis outputs.
    
    Formula: Quality Score = (1 - artifact_severity) * completeness * resolution_factor
    
    Attributes:
        metrics_id: Unique identifier.
        quality_grade: Overall quality grade.
        quality_score: Continuous quality score (0.0-1.0).
        artifacts_detected: List of detected artifacts.
        artifact_severity: Severity of artifacts (0.0-1.0).
        anatomical_coverage: Completeness of anatomical coverage.
        motion_severity: Motion artifact severity (0.0-1.0).
        contrast_timing: Contrast timing assessment.
        usability_for_diagnosis: Whether usable for diagnosis.
        recommendations: Recommendations for improvement.
    """
    metrics_id: str = field(default_factory=lambda: uuid4().hex[:12])
    quality_grade: QualityGrade = QualityGrade.ADEQUATE
    quality_score: float = 0.5
    artifacts_detected: list[str] = field(default_factory=list)
    artifact_severity: float = 0.0
    anatomical_coverage: float = 1.0
    motion_severity: float = 0.0
    contrast_timing: str = ""
    usability_for_diagnosis: bool = True
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metrics_id": self.metrics_id,
            "quality_grade": self.quality_grade.value,
            "quality_score": self.quality_score,
            "artifacts_detected": self.artifacts_detected,
            "artifact_severity": self.artifact_severity,
            "anatomical_coverage": self.anatomical_coverage,
            "motion_severity": self.motion_severity,
            "contrast_timing": self.contrast_timing,
            "usability_for_diagnosis": self.usability_for_diagnosis,
            "recommendations": self.recommendations,
        }


# =============================================================================
# Aggregate Structures
# =============================================================================

@dataclass
class CaseEvidenceBundle:
    """
    Aggregated evidence from perception outputs for disease reasoning.
    
    Collects lesion features, organ measurements, and global imaging
    characteristics as inputs to disease inference engines.
    
    Attributes:
        bundle_id: Unique identifier.
        study_id: Reference to imaging study.
        lesion_features: Per-lesion feature dictionaries.
        organ_features: Per-organ measurement dictionaries.
        global_features: Study-level global features.
        longitudinal_features: Features from prior study comparison.
        metadata_features: Demographic and acquisition metadata.
    """
    bundle_id: str = field(default_factory=lambda: uuid4().hex[:12])
    study_id: str = ""
    lesion_features: list[dict[str, Any]] = field(default_factory=list)
    organ_features: dict[str, dict[str, Any]] = field(default_factory=dict)
    global_features: dict[str, Any] = field(default_factory=dict)
    longitudinal_features: dict[str, Any] = field(default_factory=dict)
    metadata_features: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "bundle_id": self.bundle_id,
            "study_id": self.study_id,
            "lesion_features": self.lesion_features,
            "organ_features": self.organ_features,
            "global_features": self.global_features,
            "longitudinal_features": self.longitudinal_features,
            "metadata_features": self.metadata_features,
        }


@dataclass
class DiseaseReasoningOutput:
    """
    Complete disease reasoning output for a case.
    
    Aggregates all disease-level assessments into a single structure
    for integration with Evidence Dossiers and reporting.
    
    Attributes:
        output_id: Unique identifier.
        study_id: Reference to imaging study.
        pipeline_name: Name of the pipeline that generated this output.
        pipeline_version: Version of the pipeline.
        presence_assessment: Overall disease presence assessment.
        primary_hypotheses: Primary disease hypotheses (ranked).
        subtype_hypotheses: Subtype hypotheses for each disease.
        stage_assessments: Staging assessments per disease.
        differential_diagnoses: Differential diagnosis list.
        trajectory_assessment: Longitudinal trajectory if applicable.
        safety_flags: Clinical safety flags.
        evidence_bundle_id: Reference to input evidence bundle.
        disclaimers: Mandatory disclaimers for this output.
        created_at: Output generation timestamp.
    """
    output_id: str = field(default_factory=lambda: uuid4().hex[:12])
    study_id: str = ""
    pipeline_name: str = ""
    pipeline_version: str = ""
    presence_assessment: Optional[DiseasePresenceAssessment] = None
    primary_hypotheses: list[DiseaseHypothesis] = field(default_factory=list)
    subtype_hypotheses: list[DiseaseSubtypeHypothesis] = field(default_factory=list)
    stage_assessments: list[DiseaseStageAssessment] = field(default_factory=list)
    differential_diagnoses: list[DifferentialDiagnosisEntry] = field(default_factory=list)
    trajectory_assessment: Optional[DiseaseTrajectoryAssessment] = None
    safety_flags: list[ClinicalSafetyFlag] = field(default_factory=list)
    evidence_bundle_id: str = ""
    disclaimers: list[str] = field(default_factory=lambda: [
        "These assessments are image-based surrogates and do not replace clinical diagnosis.",
        "All findings require verification by qualified medical professionals.",
        "Staging assessments are approximations based on imaging features only.",
    ])
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "output_id": self.output_id,
            "study_id": self.study_id,
            "pipeline_name": self.pipeline_name,
            "pipeline_version": self.pipeline_version,
            "presence_assessment": (
                self.presence_assessment.to_dict() if self.presence_assessment else None
            ),
            "primary_hypotheses": [h.to_dict() for h in self.primary_hypotheses],
            "subtype_hypotheses": [s.to_dict() for s in self.subtype_hypotheses],
            "stage_assessments": [s.to_dict() for s in self.stage_assessments],
            "differential_diagnoses": [d.to_dict() for d in self.differential_diagnoses],
            "trajectory_assessment": (
                self.trajectory_assessment.to_dict() if self.trajectory_assessment else None
            ),
            "safety_flags": [f.to_dict() for f in self.safety_flags],
            "evidence_bundle_id": self.evidence_bundle_id,
            "disclaimers": self.disclaimers,
            "created_at": self.created_at.isoformat(),
        }

    @property
    def has_disease(self) -> bool:
        """Check if disease was detected."""
        if self.presence_assessment:
            return self.presence_assessment.disease_present == DiseasePresenceStatus.PRESENT
        return len(self.primary_hypotheses) > 0

    @property
    def has_critical_flags(self) -> bool:
        """Check for critical safety flags requiring escalation."""
        return any(
            f.severity == SafetyFlagSeverity.CRITICAL or f.escalation_required
            for f in self.safety_flags
        )

    @property
    def primary_diagnosis(self) -> Optional[DiseaseHypothesis]:
        """Get the primary (most likely) diagnosis."""
        if self.primary_hypotheses:
            return min(self.primary_hypotheses, key=lambda h: h.rank)
        return None
