# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""Tests for disease data models and reasoning."""

import pytest
from datetime import datetime, timezone


class TestDiseaseTypes:
    """Test core disease data models."""

    def test_disease_presence_assessment_creation(self):
        """Test DiseasePresenceAssessment creation."""
        from rhenium.core.disease_types import (
            DiseasePresenceAssessment,
            DiseasePresenceStatus,
        )

        assessment = DiseasePresenceAssessment(
            study_id="study_001",
            modality="CT",
            organ_systems_involved=["lung"],
            disease_present=DiseasePresenceStatus.PRESENT,
            overall_abnormality_score=0.75,
            evidence_ids=["lesion_001", "lesion_002"],
            rationale="Detected 2 pulmonary nodules.",
        )

        assert assessment.study_id == "study_001"
        assert assessment.disease_present == DiseasePresenceStatus.PRESENT
        assert len(assessment.evidence_ids) == 2
        
        # Test serialization
        data = assessment.to_dict()
        assert data["disease_present"] == "present"
        assert "created_at" in data

    def test_disease_hypothesis_creation(self):
        """Test DiseaseHypothesis creation."""
        from rhenium.core.disease_types import DiseaseHypothesis, ConfidenceLevel

        hypothesis = DiseaseHypothesis(
            disease_code="primary_lung_malignancy",
            disease_name="Suspected Primary Lung Malignancy",
            probability=0.65,
            rank=1,
            evidence_ids=["nodule_001"],
            supporting_features=["Size > 15mm", "Spiculated margins"],
            contradicting_features=[],
            confidence_level=ConfidenceLevel.MEDIUM,
        )

        assert hypothesis.probability == 0.65
        assert hypothesis.rank == 1
        assert len(hypothesis.supporting_features) == 2

        data = hypothesis.to_dict()
        assert data["confidence_level"] == "medium"

    def test_disease_subtype_hypothesis(self):
        """Test DiseaseSubtypeHypothesis creation."""
        from rhenium.core.disease_types import DiseaseSubtypeHypothesis

        subtype = DiseaseSubtypeHypothesis(
            disease_code="lung_nodule",
            subtype_code="solid_nodule",
            subtype_name="Solid Pulmonary Nodule",
            probability=0.95,
            imaging_features=["Homogeneous attenuation"],
        )

        assert subtype.subtype_name == "Solid Pulmonary Nodule"
        assert subtype.probability == 0.95

    def test_disease_stage_assessment(self):
        """Test DiseaseStageAssessment creation."""
        from rhenium.core.disease_types import DiseaseStageAssessment

        stage = DiseaseStageAssessment(
            disease_code="hcc",
            staging_system="BCLC-like imaging surrogate",
            stage_label="Early (A) surrogate",
            t_component="T1",
            assumptions=["Size-based staging only"],
            limitations=["Liver function not assessed"],
        )

        assert stage.staging_system == "BCLC-like imaging surrogate"
        assert len(stage.assumptions) == 1

    def test_differential_diagnosis_entry(self):
        """Test DifferentialDiagnosisEntry creation."""
        from rhenium.core.disease_types import DifferentialDiagnosisEntry

        entry = DifferentialDiagnosisEntry(
            rank=1,
            disease_code="lung_adenocarcinoma",
            disease_name="Lung Adenocarcinoma",
            estimated_probability=0.5,
            supporting_features=["Size", "Location"],
            cannot_exclude_flag=True,
        )

        assert entry.cannot_exclude_flag is True
        assert entry.rank == 1

    def test_disease_trajectory_assessment(self):
        """Test DiseaseTrajectoryAssessment creation."""
        from rhenium.core.disease_types import (
            DiseaseTrajectoryAssessment,
            TrajectoryLabel,
        )

        trajectory = DiseaseTrajectoryAssessment(
            study_id_current="study_002",
            study_id_previous="study_001",
            disease_code="lung_nodule",
            trajectory_label=TrajectoryLabel.WORSENED,
            new_lesion_count=1,
            quantitative_deltas={"total_volume_change_percent": 35.5},
        )

        assert trajectory.trajectory_label == TrajectoryLabel.WORSENED
        assert trajectory.new_lesion_count == 1

    def test_clinical_safety_flag(self):
        """Test ClinicalSafetyFlag creation."""
        from rhenium.core.disease_types import (
            ClinicalSafetyFlag,
            SafetyFlagType,
            SafetyFlagSeverity,
        )

        flag = ClinicalSafetyFlag(
            flag_type=SafetyFlagType.URGENT_FINDING,
            severity=SafetyFlagSeverity.CRITICAL,
            description="Large intracranial hemorrhage",
            recommended_action="Immediate neurosurgical evaluation",
            escalation_required=True,
            time_sensitivity="immediate",
        )

        assert flag.escalation_required is True
        assert flag.severity == SafetyFlagSeverity.CRITICAL

    def test_case_evidence_bundle(self):
        """Test CaseEvidenceBundle creation."""
        from rhenium.core.disease_types import CaseEvidenceBundle

        bundle = CaseEvidenceBundle(
            study_id="study_001",
            lesion_features=[
                {"id": "nodule_001", "organ": "lung", "diameter_mm": 12.5},
            ],
            organ_features={"lung": {"volume_ml": 5000}},
            global_features={"abnormality_score": 0.6},
        )

        assert len(bundle.lesion_features) == 1
        assert bundle.organ_features["lung"]["volume_ml"] == 5000

    def test_disease_reasoning_output(self):
        """Test DiseaseReasoningOutput creation and properties."""
        from rhenium.core.disease_types import (
            DiseaseReasoningOutput,
            DiseasePresenceAssessment,
            DiseasePresenceStatus,
            DiseaseHypothesis,
            ClinicalSafetyFlag,
            SafetyFlagSeverity,
            ConfidenceLevel,
        )

        output = DiseaseReasoningOutput(
            study_id="study_001",
            pipeline_name="lung_nodule_pipeline",
            presence_assessment=DiseasePresenceAssessment(
                disease_present=DiseasePresenceStatus.PRESENT,
            ),
            primary_hypotheses=[
                DiseaseHypothesis(
                    disease_code="lung_nodule",
                    disease_name="Pulmonary Nodule",
                    probability=0.8,
                    rank=1,
                    confidence_level=ConfidenceLevel.HIGH,
                ),
            ],
            safety_flags=[
                ClinicalSafetyFlag(
                    severity=SafetyFlagSeverity.CRITICAL,
                    escalation_required=True,
                ),
            ],
        )

        assert output.has_disease is True
        assert output.has_critical_flags is True
        assert output.primary_diagnosis is not None
        assert output.primary_diagnosis.disease_code == "lung_nodule"

        # Test serialization
        data = output.to_dict()
        assert data["study_id"] == "study_001"
        assert len(data["disclaimers"]) == 3

    def test_disease_reasoning_output_no_disease(self):
        """Test DiseaseReasoningOutput with no disease."""
        from rhenium.core.disease_types import (
            DiseaseReasoningOutput,
            DiseasePresenceAssessment,
            DiseasePresenceStatus,
        )

        output = DiseaseReasoningOutput(
            study_id="study_002",
            presence_assessment=DiseasePresenceAssessment(
                disease_present=DiseasePresenceStatus.ABSENT,
            ),
        )

        assert output.has_disease is False
        assert output.has_critical_flags is False
        assert output.primary_diagnosis is None


class TestPresenceAssessor:
    """Test disease presence assessment."""

    def test_presence_assessment_positive(self):
        """Test presence assessor with lesions present."""
        from rhenium.disease.presence_assessment import PresenceAssessor
        from rhenium.core.disease_types import (
            CaseEvidenceBundle,
            DiseasePresenceStatus,
        )

        assessor = PresenceAssessor()
        evidence = CaseEvidenceBundle(
            study_id="study_001",
            lesion_features=[
                {"id": "nodule_001", "organ": "lung", "diameter_mm": 10},
            ],
            global_features={"abnormality_score": 0.6},
        )

        assessment = assessor.assess(evidence, modality="CT", organ_systems=["lung"])
        
        assert assessment.disease_present == DiseasePresenceStatus.PRESENT
        assert len(assessment.evidence_ids) == 1

    def test_presence_assessment_negative(self):
        """Test presence assessor with no lesions."""
        from rhenium.disease.presence_assessment import PresenceAssessor
        from rhenium.core.disease_types import (
            CaseEvidenceBundle,
            DiseasePresenceStatus,
        )

        assessor = PresenceAssessor()
        evidence = CaseEvidenceBundle(
            study_id="study_002",
            lesion_features=[],
            global_features={"abnormality_score": 0.1},
        )

        assessment = assessor.assess(evidence, modality="CT", organ_systems=["lung"])
        
        assert assessment.disease_present == DiseasePresenceStatus.ABSENT


class TestSafetyFlagDetector:
    """Test clinical safety flag detection."""

    def test_quality_flag_detection(self):
        """Test image quality safety flag detection."""
        from rhenium.disease.safety_flags import SafetyFlagDetector
        from rhenium.core.disease_types import CaseEvidenceBundle

        detector = SafetyFlagDetector()
        evidence = CaseEvidenceBundle(
            study_id="study_001",
            metadata_features={
                "image_quality": 0.3,
                "motion_artifact_severity": 0.8,
            },
        )

        flags = detector.detect(evidence, [], context="chest_ct")
        
        assert len(flags) >= 1
        assert any("motion" in f.finding_pattern or "quality" in f.description.lower() for f in flags)


class TestDifferentialGenerator:
    """Test differential diagnosis generation."""

    def test_differential_generation(self):
        """Test generating differential from hypotheses."""
        from rhenium.disease.differential import DifferentialGenerator
        from rhenium.core.disease_types import (
            CaseEvidenceBundle,
            DiseaseHypothesis,
            ConfidenceLevel,
        )

        generator = DifferentialGenerator()
        evidence = CaseEvidenceBundle(study_id="study_001")
        hypotheses = [
            DiseaseHypothesis(
                disease_code="malignancy",
                disease_name="Primary Malignancy",
                probability=0.6,
                confidence_level=ConfidenceLevel.MEDIUM,
            ),
            DiseaseHypothesis(
                disease_code="benign",
                disease_name="Benign Lesion",
                probability=0.3,
                confidence_level=ConfidenceLevel.LOW,
            ),
        ]

        differential = generator.generate(hypotheses, evidence, "lung_nodule")
        
        assert len(differential) == 2
        assert differential[0].rank == 1
        assert differential[0].disease_name == "Primary Malignancy"
