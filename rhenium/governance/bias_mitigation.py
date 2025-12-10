# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Bias Mitigation Module
======================

Strategies and hooks for mitigating algorithmic bias in medical imaging AI.
Includes reweighting, augmentation, and algorithmic approaches.

"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np

from rhenium.core.logging import get_governance_logger

logger = get_governance_logger()


@dataclass
class MitigationConfig:
    """Configuration for bias mitigation strategy."""
    strategy: str
    parameters: dict[str, Any]


class BiasMitigationStrategy(ABC):
    """Abstract base class for bias mitigation strategies."""
    
    @abstractmethod
    def apply(self, data: Any, labels: np.ndarray, groups: np.ndarray) -> tuple[Any, np.ndarray]:
        """Apply mitigation strategy. Returns modified data and weights."""
        pass


class ReweightingStrategy(BiasMitigationStrategy):
    """
    Reweight samples to equalize representation across groups.
    
    This strategy assigns higher weights to underrepresented groups
    during training to reduce performance disparities.
    """
    
    def __init__(self, target_distribution: str = "uniform"):
        """
        Args:
            target_distribution: Target group distribution (uniform, balanced)
        """
        self.target_distribution = target_distribution
        
    def apply(self, data: Any, labels: np.ndarray, groups: np.ndarray) -> tuple[Any, np.ndarray]:
        """Compute sample weights to balance groups."""
        unique_groups, group_counts = np.unique(groups, return_counts=True)
        
        if self.target_distribution == "uniform":
            target_count = len(groups) / len(unique_groups)
        else:
            target_count = np.max(group_counts)
            
        weights = np.ones(len(groups))
        
        for group, count in zip(unique_groups, group_counts):
            group_weight = target_count / count
            weights[groups == group] = group_weight
            
        # Normalize weights
        weights = weights / weights.sum() * len(weights)
        
        logger.info("Reweighting applied", 
                   unique_groups=len(unique_groups),
                   weight_range=(weights.min(), weights.max()))
        
        return data, weights


class OversamplingStrategy(BiasMitigationStrategy):
    """
    Oversample underrepresented groups to balance training data.
    """
    
    def __init__(self, sampling_strategy: str = "minority"):
        self.sampling_strategy = sampling_strategy
        
    def apply(self, data: Any, labels: np.ndarray, groups: np.ndarray) -> tuple[Any, np.ndarray]:
        """Generate oversampling indices."""
        unique_groups, group_counts = np.unique(groups, return_counts=True)
        max_count = np.max(group_counts)
        
        resampled_indices = []
        
        for group in unique_groups:
            group_indices = np.where(groups == group)[0]
            n_samples = max_count - len(group_indices)
            
            if n_samples > 0:
                additional = np.random.choice(group_indices, size=n_samples, replace=True)
                resampled_indices.extend(additional)
                
            resampled_indices.extend(group_indices)
            
        resampled_indices = np.array(resampled_indices)
        np.random.shuffle(resampled_indices)
        
        logger.info("Oversampling applied",
                   original_size=len(groups),
                   resampled_size=len(resampled_indices))
        
        return resampled_indices, np.ones(len(resampled_indices))


class ThresholdAdjustmentStrategy(BiasMitigationStrategy):
    """
    Post-hoc threshold adjustment to equalize metrics across groups.
    
    This strategy learns group-specific decision thresholds to achieve
    equal opportunity or equalized odds.
    """
    
    def __init__(self, criterion: str = "equal_opportunity"):
        """
        Args:
            criterion: Fairness criterion (equal_opportunity, equalized_odds)
        """
        self.criterion = criterion
        self.thresholds: dict[str, float] = {}
        
    def learn_thresholds(
        self,
        y_true: np.ndarray,
        y_scores: np.ndarray,
        groups: np.ndarray,
        target_tpr: float = 0.9,
    ) -> dict[str, float]:
        """Learn group-specific thresholds."""
        unique_groups = np.unique(groups)
        
        for group in unique_groups:
            mask = groups == group
            group_true = y_true[mask]
            group_scores = y_scores[mask]
            
            # Find threshold achieving target TPR
            positive_scores = group_scores[group_true == 1]
            if len(positive_scores) > 0:
                threshold = np.percentile(positive_scores, (1 - target_tpr) * 100)
            else:
                threshold = 0.5
                
            self.thresholds[str(group)] = float(threshold)
            
        logger.info("Thresholds learned", thresholds=self.thresholds)
        return self.thresholds
        
    def apply(self, data: Any, labels: np.ndarray, groups: np.ndarray) -> tuple[Any, np.ndarray]:
        """Return stored thresholds (data unchanged)."""
        return data, np.ones(len(labels))


def get_mitigation_strategy(config: MitigationConfig) -> BiasMitigationStrategy:
    """Factory function for bias mitigation strategies."""
    strategies = {
        "reweighting": ReweightingStrategy,
        "oversampling": OversamplingStrategy,
        "threshold_adjustment": ThresholdAdjustmentStrategy,
    }
    
    if config.strategy not in strategies:
        raise ValueError(f"Unknown strategy: {config.strategy}")
        
    return strategies[config.strategy](**config.parameters)
