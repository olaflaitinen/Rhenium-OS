# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""Tests for fairness metrics and bias mitigation."""

import pytest
import numpy as np

from rhenium.governance.fairness_metrics import (
    compute_stratified_auc,
    compute_disparity_metrics,
    evaluate_fairness,
    SubgroupMetrics,
)
from rhenium.governance.fairness_reports import (
    generate_fairness_report,
    assess_fairness_thresholds,
)
from rhenium.governance.bias_mitigation import (
    ReweightingStrategy,
    OversamplingStrategy,
    MitigationConfig,
    get_mitigation_strategy,
)


class TestFairnessMetrics:
    """Test fairness metric computations."""
    
    def test_stratified_auc_computation(self):
        """Test stratified AUC across groups."""
        np.random.seed(42)
        n = 200
        
        y_true = np.random.randint(0, 2, n)
        y_scores = y_true * 0.6 + np.random.rand(n) * 0.4
        groups = np.array(["A"] * 100 + ["B"] * 100)
        
        result = compute_stratified_auc(y_true, y_scores, groups)
        
        assert "A" in result
        assert "B" in result
        assert 0.5 <= result["A"] <= 1.0
        assert 0.5 <= result["B"] <= 1.0
    
    def test_disparity_metrics(self):
        """Test disparity metric computation."""
        subgroups = [
            SubgroupMetrics("sex", "male", 100, auc=0.92),
            SubgroupMetrics("sex", "female", 100, auc=0.88),
        ]
        
        disparities = compute_disparity_metrics(subgroups, "auc")
        
        assert "max_disparity" in disparities
        assert disparities["max_disparity"] == pytest.approx(0.04)
        assert "disparity_ratio" in disparities
        
    def test_evaluate_fairness(self):
        """Test comprehensive fairness evaluation."""
        np.random.seed(42)
        n = 300
        
        y_true = np.random.randint(0, 2, n)
        y_scores = y_true * 0.7 + np.random.rand(n) * 0.3
        
        demographics = {
            "institution": np.array(["A"] * 150 + ["B"] * 150),
        }
        
        result = evaluate_fairness(y_true, y_scores, demographics)
        
        assert result.overall_metrics["auc"] > 0.5
        assert len(result.subgroup_metrics) >= 2


class TestFairnessReports:
    """Test fairness report generation."""
    
    def test_markdown_report_generation(self):
        """Test markdown report generation."""
        from rhenium.governance.fairness_metrics import FairnessMetrics
        
        metrics = FairnessMetrics(
            overall_metrics={"auc": 0.9},
            subgroup_metrics=[
                SubgroupMetrics("sex", "male", 100, auc=0.92),
                SubgroupMetrics("sex", "female", 100, auc=0.88),
            ],
            disparity_metrics={"sex_max_disparity": 0.04},
        )
        
        report = generate_fairness_report(
            metrics,
            pipeline_name="test_pipeline",
            dataset_name="test_dataset",
            format="markdown",
        )
        
        assert "Fairness Evaluation Report" in report
        assert "test_pipeline" in report
        assert "male" in report
        assert "female" in report
    
    def test_threshold_assessment_pass(self):
        """Test fairness threshold assessment - passing."""
        from rhenium.governance.fairness_metrics import FairnessMetrics
        
        metrics = FairnessMetrics(
            overall_metrics={},
            subgroup_metrics=[],
            disparity_metrics={"sex_max_disparity": 0.02, "sex_disparity_ratio": 0.98},
        )
        
        result = assess_fairness_thresholds(metrics)
        assert result["passed"] is True
        
    def test_threshold_assessment_fail(self):
        """Test fairness threshold assessment - failing."""
        from rhenium.governance.fairness_metrics import FairnessMetrics
        
        metrics = FairnessMetrics(
            overall_metrics={},
            subgroup_metrics=[],
            disparity_metrics={"sex_max_disparity": 0.15},
        )
        
        result = assess_fairness_thresholds(metrics)
        assert result["passed"] is False
        assert len(result["issues"]) > 0


class TestBiasMitigation:
    """Test bias mitigation strategies."""
    
    def test_reweighting_strategy(self):
        """Test reweighting strategy."""
        strategy = ReweightingStrategy(target_distribution="uniform")
        
        n = 200
        groups = np.array(["majority"] * 150 + ["minority"] * 50)
        labels = np.zeros(n)
        
        _, weights = strategy.apply(None, labels, groups)
        
        assert len(weights) == n
        # Minority group should have higher weights
        minority_weights = weights[groups == "minority"]
        majority_weights = weights[groups == "majority"]
        assert minority_weights.mean() > majority_weights.mean()
    
    def test_oversampling_strategy(self):
        """Test oversampling strategy."""
        strategy = OversamplingStrategy()
        
        n = 200
        groups = np.array(["majority"] * 150 + ["minority"] * 50)
        labels = np.zeros(n)
        
        indices, _ = strategy.apply(None, labels, groups)
        
        # Should have more samples after oversampling
        assert len(indices) >= n
    
    def test_mitigation_factory(self):
        """Test mitigation strategy factory."""
        config = MitigationConfig(
            strategy="reweighting",
            parameters={"target_distribution": "balanced"},
        )
        
        strategy = get_mitigation_strategy(config)
        assert isinstance(strategy, ReweightingStrategy)
