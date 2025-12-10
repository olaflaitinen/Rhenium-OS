# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Fairness Metrics Module
=======================

Metrics for evaluating algorithmic fairness across demographic subgroups,
institutions, and scanner vendors. Supports stratified performance analysis
and bias detection as required by EU AI Act and clinical best practices.

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rhenium.core.logging import get_governance_logger

logger = get_governance_logger()


@dataclass
class SubgroupMetrics:
    """Performance metrics for a specific subgroup."""
    subgroup_name: str
    subgroup_value: str
    sample_count: int
    auc: float | None = None
    sensitivity: float | None = None
    specificity: float | None = None
    ppv: float | None = None
    npv: float | None = None
    calibration_error: float | None = None
    dice: float | None = None


@dataclass
class FairnessMetrics:
    """Aggregated fairness metrics across subgroups."""
    overall_metrics: dict[str, float]
    subgroup_metrics: list[SubgroupMetrics]
    disparity_metrics: dict[str, float] = field(default_factory=dict)


def compute_stratified_auc(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    groups: np.ndarray,
) -> dict[str, float]:
    """
    Compute AUC stratified by group membership.
    
    Args:
        y_true: Binary ground truth labels
        y_scores: Prediction scores
        groups: Group identifiers for each sample
        
    Returns:
        Dictionary mapping group name to AUC
    """
    from sklearn.metrics import roc_auc_score
    
    results = {}
    unique_groups = np.unique(groups)
    
    for group in unique_groups:
        mask = groups == group
        if mask.sum() < 10 or len(np.unique(y_true[mask])) < 2:
            logger.warning(f"Insufficient samples for group {group}")
            continue
            
        try:
            auc = roc_auc_score(y_true[mask], y_scores[mask])
            results[str(group)] = float(auc)
        except Exception as e:
            logger.warning(f"AUC computation failed for group {group}: {e}")
            
    return results


def compute_calibration_by_group(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    groups: np.ndarray,
    n_bins: int = 10,
) -> dict[str, float]:
    """
    Compute Expected Calibration Error (ECE) per group.
    
    Args:
        y_true: Binary ground truth
        y_prob: Predicted probabilities
        groups: Group identifiers
        n_bins: Number of calibration bins
        
    Returns:
        ECE per group
    """
    results = {}
    unique_groups = np.unique(groups)
    
    for group in unique_groups:
        mask = groups == group
        if mask.sum() < n_bins:
            continue
            
        ece = _compute_ece(y_true[mask], y_prob[mask], n_bins)
        results[str(group)] = float(ece)
        
    return results


def _compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> float:
    """Compute Expected Calibration Error."""
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        bin_mask = (y_prob >= bin_boundaries[i]) & (y_prob < bin_boundaries[i + 1])
        if bin_mask.sum() == 0:
            continue
            
        bin_acc = y_true[bin_mask].mean()
        bin_conf = y_prob[bin_mask].mean()
        bin_weight = bin_mask.sum() / len(y_true)
        ece += bin_weight * abs(bin_acc - bin_conf)
        
    return ece


def compute_disparity_metrics(
    subgroup_metrics: list[SubgroupMetrics],
    metric_name: str = "auc",
) -> dict[str, float]:
    """
    Compute disparity metrics across subgroups.
    
    Returns:
        - max_disparity: Maximum difference between subgroups
        - disparity_ratio: Ratio of worst to best performing group
        - variance: Variance of metric across groups
    """
    values = []
    for sm in subgroup_metrics:
        value = getattr(sm, metric_name, None)
        if value is not None:
            values.append(value)
            
    if len(values) < 2:
        return {}
        
    values = np.array(values)
    return {
        "max_disparity": float(np.max(values) - np.min(values)),
        "disparity_ratio": float(np.min(values) / np.max(values)) if np.max(values) > 0 else 0.0,
        "variance": float(np.var(values)),
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
    }


def evaluate_fairness(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    demographic_groups: dict[str, np.ndarray],
) -> FairnessMetrics:
    """
    Comprehensive fairness evaluation across multiple demographic dimensions.
    
    Args:
        y_true: Ground truth labels
        y_scores: Prediction scores
        demographic_groups: Dict mapping dimension name to group assignments
        
    Returns:
        FairnessMetrics with overall, per-subgroup, and disparity metrics
    """
    from sklearn.metrics import roc_auc_score
    
    logger.info("Evaluating fairness", num_dimensions=len(demographic_groups))
    
    # Overall metrics
    overall = {
        "auc": float(roc_auc_score(y_true, y_scores)),
        "sample_count": int(len(y_true)),
    }
    
    # Per-subgroup metrics
    subgroup_results = []
    all_disparities = {}
    
    for dimension_name, groups in demographic_groups.items():
        stratified_auc = compute_stratified_auc(y_true, y_scores, groups)
        calibration = compute_calibration_by_group(y_true, y_scores, groups)
        
        for group_value, auc in stratified_auc.items():
            subgroup_results.append(SubgroupMetrics(
                subgroup_name=dimension_name,
                subgroup_value=group_value,
                sample_count=int((groups == group_value).sum()),
                auc=auc,
                calibration_error=calibration.get(group_value),
            ))
            
        # Compute disparity for this dimension
        dimension_subgroups = [s for s in subgroup_results if s.subgroup_name == dimension_name]
        disparities = compute_disparity_metrics(dimension_subgroups, "auc")
        for key, value in disparities.items():
            all_disparities[f"{dimension_name}_{key}"] = value
            
    logger.info("Fairness evaluation complete", num_subgroups=len(subgroup_results))
    
    return FairnessMetrics(
        overall_metrics=overall,
        subgroup_metrics=subgroup_results,
        disparity_metrics=all_disparities,
    )
