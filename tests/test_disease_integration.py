# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""Integration tests for disease reasoning pipelines."""

import pytest
from unittest.mock import Mock, patch


class TestDiseaseReasoningIntegration:
    """Integration tests for disease reasoning with pipeline."""

    def test_lung_nodule_pipeline_with_disease_reasoning(self):
        """Test lung nodule pipeline produces disease output."""
        from rhenium.core.disease_types import (
            CaseEvidenceBundle,
            DiseasePresenceStatus,
        )
        from rhenium.disease.lung.nodule_assessment import LungNoduleAssessor

        # Create mock evidence
        evidence = CaseEvidenceBundle(
            study_id="test_study_001",
            lesion_features=[
                {
                    "id": "nodule_001",
                    "organ": "lung",
                    "type": "nodule",
                    "diameter_mm": 15.0,
                    "morphology": "spiculated",
                    "location_mm": (100, 200, 50),
                    "lobe": "RUL",
                },
            ],
            organ_features={"lung": {"volume_ml": 5500}},
            global_features={"abnormality_score": 0.7},
        )

        # Run disease assessment
        assessor = LungNoduleAssessor()
        output = assessor.run(evidence)

        # Verify presence assessment
        assert output.presence_assessment is not None
        assert output.presence_assessment.disease_present == DiseasePresenceStatus.PRESENT

        # Verify hypotheses generated
        assert len(output.primary_hypotheses) >= 1
        assert output.primary_hypotheses[0].probability > 0.1

        # Verify differential generated
        assert len(output.differential_diagnoses) >= 1

        # Verify serialization works
        output_dict = output.to_dict()
        assert "primary_hypotheses" in output_dict
        assert "presence_assessment" in output_dict

    def test_liver_lesion_pipeline_with_disease_reasoning(self):
        """Test liver lesion pipeline produces disease output."""
        from rhenium.core.disease_types import (
            CaseEvidenceBundle,
            DiseasePresenceStatus,
        )
        from rhenium.disease.liver.lesion_assessment import LiverLesionAssessor

        evidence = CaseEvidenceBundle(
            study_id="test_study_002",
            lesion_features=[
                {
                    "id": "lesion_001",
                    "organ": "liver",
                    "diameter_mm": 25.0,
                    "arterial_enhancement": True,
                    "venous_washout": True,
                    "location_segment": 6,
                },
            ],
            metadata_features={"modality": "CT"},
        )

        assessor = LiverLesionAssessor()
        output = assessor.run(evidence)

        assert output.presence_assessment is not None
        assert output.presence_assessment.disease_present == DiseasePresenceStatus.PRESENT
        assert len(output.primary_hypotheses) >= 1

        # Should detect HCC pattern
        hcc_hypothesis = next(
            (h for h in output.primary_hypotheses if h.disease_code == "hcc"),
            None
        )
        assert hcc_hypothesis is not None

    def test_brain_lesion_pipeline_with_safety_flags(self):
        """Test brain lesion pipeline detects safety flags."""
        from rhenium.core.disease_types import (
            CaseEvidenceBundle,
            SafetyFlagSeverity,
        )
        from rhenium.disease.brain.lesion_assessment import BrainLesionAssessor

        evidence = CaseEvidenceBundle(
            study_id="test_study_003",
            lesion_features=[
                {
                    "id": "hemorrhage_001",
                    "organ": "brain",
                    "type": "hemorrhage",
                    "is_hemorrhage": True,
                    "hemorrhage_type": "intraparenchymal",
                    "volume_ml": 45.0,
                    "location": "right basal ganglia",
                    "midline_shift_mm": 8.0,
                },
            ],
            metadata_features={"modality": "CT"},
        )

        assessor = BrainLesionAssessor()
        output = assessor.run(evidence)

        # Should detect hemorrhage
        assert output.presence_assessment is not None

        # Should generate safety flags
        assert len(output.safety_flags) >= 1
        
        # Check for critical safety flag
        critical_flags = [
            f for f in output.safety_flags
            if f.severity == SafetyFlagSeverity.CRITICAL
        ]
        assert len(critical_flags) >= 1

    def test_disease_output_in_pipeline_result(self):
        """Test disease output integrates with PipelineResult."""
        from rhenium.pipelines.base_pipeline import PipelineResult
        from rhenium.core.disease_types import (
            DiseaseReasoningOutput,
            DiseasePresenceAssessment,
            DiseasePresenceStatus,
            DiseaseHypothesis,
            ClinicalSafetyFlag,
            SafetyFlagSeverity,
            ConfidenceLevel,
        )

        disease_output = DiseaseReasoningOutput(
            study_id="test_study",
            presence_assessment=DiseasePresenceAssessment(
                disease_present=DiseasePresenceStatus.PRESENT,
            ),
            primary_hypotheses=[
                DiseaseHypothesis(
                    disease_code="test_disease",
                    disease_name="Test Disease",
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

        result = PipelineResult(
            pipeline_name="test_pipeline",
            disease_output=disease_output,
        )

        assert result.has_disease is True
        assert result.has_critical_findings is True

        result_dict = result.to_dict()
        assert result_dict["disease_output"] is not None
        assert result_dict["disease_output"]["study_id"] == "test_study"

    def test_trajectory_assessment_with_prior_study(self):
        """Test trajectory assessment with comparison to prior."""
        from rhenium.core.disease_types import (
            CaseEvidenceBundle,
            TrajectoryLabel,
        )
        from rhenium.disease.trajectory import TrajectoryAssessor

        current = CaseEvidenceBundle(
            study_id="study_002",
            lesion_features=[
                {"id": "nodule_001", "matched_prior_id": "prior_001", 
                 "volume_mm3": 1500, "longest_diameter_mm": 14.0, "location": "RUL"},
            ],
            metadata_features={"study_date": "2025-12-01"},
        )

        prior = CaseEvidenceBundle(
            study_id="study_001",
            lesion_features=[
                {"id": "prior_001", "volume_mm3": 1000, 
                 "longest_diameter_mm": 12.0, "location": "RUL"},
            ],
            metadata_features={"study_date": "2025-06-01"},
        )

        assessor = TrajectoryAssessor()
        trajectory = assessor.assess(current, prior, disease_code="lung_nodule")

        assert trajectory.trajectory_label == TrajectoryLabel.WORSENED
        assert trajectory.new_lesion_count == 0
        assert "total_volume_change_percent" in trajectory.quantitative_deltas
        assert trajectory.quantitative_deltas["total_volume_change_percent"] > 0


class TestDifferentialGeneratorIntegration:
    """Integration tests for differential diagnosis generation."""

    def test_differential_with_cannot_exclude(self):
        """Test differential generation with cannot-exclude logic."""
        from rhenium.disease.differential import DifferentialGenerator, DifferentialConfig
        from rhenium.core.disease_types import (
            CaseEvidenceBundle,
            DiseaseHypothesis,
            ConfidenceLevel,
        )

        config = DifferentialConfig(
            max_entries=5,
            include_cannot_exclude=True,
            include_suggested_tests=True,
        )
        generator = DifferentialGenerator(config)

        hypotheses = [
            DiseaseHypothesis(
                disease_code="primary_lung_malignancy",
                disease_name="Primary Lung Malignancy",
                probability=0.4,
                confidence_level=ConfidenceLevel.MEDIUM,
            ),
            DiseaseHypothesis(
                disease_code="granuloma",
                disease_name="Pulmonary Granuloma",
                probability=0.3,
                confidence_level=ConfidenceLevel.MEDIUM,
            ),
        ]

        evidence = CaseEvidenceBundle(study_id="test")
        differential = generator.generate(hypotheses, evidence, "lung_nodule")

        assert len(differential) == 2
        assert differential[0].rank == 1
        
        # First entry should be malignancy
        assert differential[0].disease_code == "primary_lung_malignancy"
        
        # Should have cannot-exclude flag for malignancy in moderate probability range
        assert differential[0].cannot_exclude_flag is True


class TestSafetyFlagIntegration:
    """Integration tests for safety flag detection."""

    def test_safety_flag_detection_from_evidence(self):
        """Test safety flag detection from evidence bundle."""
        from rhenium.disease.safety_flags import SafetyFlagDetector
        from rhenium.core.disease_types import CaseEvidenceBundle, SafetyFlagSeverity

        detector = SafetyFlagDetector()
        
        evidence = CaseEvidenceBundle(
            study_id="urgent_001",
            lesion_features=[
                {"id": "hemorrhage_001", "safety_pattern": "large_ich"},
            ],
            global_features={
                "safety_indicators": {"large_ich": True},
            },
            metadata_features={"image_quality": 0.9},
        )

        flags = detector.detect(evidence, [], context="head_ct")

        assert len(flags) >= 1
        critical_count = sum(1 for f in flags if f.severity == SafetyFlagSeverity.CRITICAL)
        assert critical_count >= 1
