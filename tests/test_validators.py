# Copyright (c) 2025 Skolyn LLC. All rights reserved.
# SPDX-License-Identifier: EUPL-1.1

"""Tests for MedGemma validators."""

import pytest
from unittest.mock import MagicMock

from rhenium.medgemma.validators import (
    NarrativeQuantitativeConsistencyValidator,
    LateralityConsistencyValidator,
    HighRiskFindingValidator,
    OverconfidenceValidator,
    ValidationSuite,
    get_default_validation_suite,
)
from rhenium.xai.explanation_schema import Finding


@pytest.fixture
def sample_finding():
    """Create a sample finding."""
    return Finding(
        finding_id="test_001",
        finding_type="lesion",
        description="Test lesion",
        confidence=0.85,
        laterality="left",
        measurements={"diameter_mm": 25.0},
    )


class TestNarrativeQuantitativeConsistencyValidator:
    """Test narrative-quantitative consistency validation."""
    
    def test_consistent_narrative(self, sample_finding):
        """Test validation passes for consistent narrative."""
        validator = NarrativeQuantitativeConsistencyValidator()
        result = validator.validate(
            narrative="Moderate-sized lesion noted in the left region.",
            findings=[sample_finding],
            measurements={"size_mm": 25.0},
        )
        assert result.passed
        assert len(result.issues) == 0
    
    def test_inconsistent_small_descriptor(self):
        """Test detection of inconsistent size descriptor."""
        validator = NarrativeQuantitativeConsistencyValidator()
        result = validator.validate(
            narrative="A tiny lesion was identified.",
            measurements={"size_mm": 50.0},
        )
        assert not result.passed
        assert result.requires_human_review
        assert len(result.issues) > 0


class TestLateralityConsistencyValidator:
    """Test laterality consistency validation."""
    
    def test_unilateral_finding(self, sample_finding):
        """Test validation for unilateral finding."""
        validator = LateralityConsistencyValidator()
        result = validator.validate(
            narrative="Finding in the left knee.",
            findings=[sample_finding],
        )
        assert result.passed
    
    def test_bilateral_warning(self):
        """Test warning for bilateral findings without designation."""
        validator = LateralityConsistencyValidator()
        left_finding = Finding(
            finding_id="l1", finding_type="lesion",
            description="Left", confidence=0.9, laterality="left",
        )
        right_finding = Finding(
            finding_id="r1", finding_type="lesion",
            description="Right", confidence=0.9, laterality="right",
        )
        result = validator.validate(
            narrative="Multiple findings.",
            findings=[left_finding, right_finding],
        )
        assert len(result.warnings) > 0


class TestHighRiskFindingValidator:
    """Test high-risk finding detection."""
    
    def test_normal_finding(self):
        """Test no flag for normal findings."""
        validator = HighRiskFindingValidator()
        result = validator.validate(
            narrative="Unremarkable examination. No significant abnormalities.",
        )
        assert result.passed
        assert not result.requires_human_review
    
    def test_malignancy_detection(self):
        """Test detection of malignancy terminology."""
        validator = HighRiskFindingValidator()
        result = validator.validate(
            narrative="Findings suspicious for malignancy requiring biopsy.",
        )
        assert result.requires_human_review
        assert any("malign" in w.lower() for w in result.warnings)
    
    def test_stroke_detection(self):
        """Test detection of stroke terminology."""
        validator = HighRiskFindingValidator()
        result = validator.validate(
            narrative="Acute infarct in the left MCA territory.",
        )
        assert result.requires_human_review


class TestOverconfidenceValidator:
    """Test overconfidence detection."""
    
    def test_appropriate_uncertainty(self):
        """Test acceptance of appropriate uncertainty language."""
        validator = OverconfidenceValidator()
        result = validator.validate(
            narrative="This finding likely represents a benign lesion. "
                     "Differential diagnosis may include cyst or adenoma. "
                     "Recommend follow-up imaging.",
        )
        assert len(result.warnings) == 0
    
    def test_overconfident_language(self):
        """Test detection of overconfident language."""
        validator = OverconfidenceValidator()
        result = validator.validate(
            narrative="This is definitely a malignant tumor. "
                     "There is absolutely no doubt about the diagnosis.",
        )
        assert len(result.warnings) > 0


class TestValidationSuite:
    """Test the validation suite."""
    
    def test_default_suite(self):
        """Test default validation suite runs all validators."""
        suite = get_default_validation_suite()
        assert len(suite.validators) == 4
    
    def test_validate_all(self, sample_finding):
        """Test running all validators."""
        suite = ValidationSuite()
        results = suite.validate_all(
            narrative="Moderate lesion in the left knee.",
            findings=[sample_finding],
        )
        assert len(results) == 4
        assert all(r.validator_name for r in results)
    
    def test_requires_human_review(self):
        """Test human review detection."""
        suite = ValidationSuite()
        results = suite.validate_all(
            narrative="Suspected malignant tumor identified.",
        )
        assert suite.requires_human_review(results)
