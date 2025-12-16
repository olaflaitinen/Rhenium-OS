"""Fairness assessment for AI models."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass
class FairnessMetrics:
    """Fairness metrics across demographic groups."""
    demographic_parity: float    # |P(Y=1|G=0) - P(Y=1|G=1)|
    equalized_odds: float        # Max diff in TPR/FPR across groups
    calibration_diff: float      # Difference in calibration
    sample_sizes: dict[str, int]


class FairnessAnalyzer:
    """Analyze model fairness across demographic groups."""

    def __init__(self, protected_attribute: str = "sex"):
        self.protected_attribute = protected_attribute

    def compute_metrics(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        demographics: np.ndarray,
    ) -> FairnessMetrics:
        """Compute fairness metrics."""
        groups = np.unique(demographics)
        group_metrics = {}

        for g in groups:
            mask = demographics == g
            group_metrics[g] = {
                "pred_rate": predictions[mask].mean(),
                "tpr": self._tpr(predictions[mask], labels[mask]),
                "fpr": self._fpr(predictions[mask], labels[mask]),
                "n": mask.sum(),
            }

        # Demographic parity
        rates = [m["pred_rate"] for m in group_metrics.values()]
        dp = max(rates) - min(rates)

        # Equalized odds
        tprs = [m["tpr"] for m in group_metrics.values()]
        fprs = [m["fpr"] for m in group_metrics.values()]
        eo = max(max(tprs) - min(tprs), max(fprs) - min(fprs))

        return FairnessMetrics(
            demographic_parity=float(dp),
            equalized_odds=float(eo),
            calibration_diff=0.0,
            sample_sizes={str(g): m["n"] for g, m in group_metrics.items()},
        )

    def _tpr(self, pred: np.ndarray, label: np.ndarray) -> float:
        pos = label == 1
        if pos.sum() == 0:
            return 0.0
        return (pred[pos] == 1).mean()

    def _fpr(self, pred: np.ndarray, label: np.ndarray) -> float:
        neg = label == 0
        if neg.sum() == 0:
            return 0.0
        return (pred[neg] == 1).mean()

    def generate_report(
        self,
        metrics: FairnessMetrics,
        threshold: float = 0.1,
    ) -> dict[str, Any]:
        """Generate fairness report."""
        return {
            "demographic_parity": {
                "value": metrics.demographic_parity,
                "threshold": threshold,
                "passed": metrics.demographic_parity < threshold,
            },
            "equalized_odds": {
                "value": metrics.equalized_odds,
                "threshold": threshold,
                "passed": metrics.equalized_odds < threshold,
            },
            "sample_sizes": metrics.sample_sizes,
            "recommendations": self._get_recommendations(metrics, threshold),
        }

    def _get_recommendations(
        self,
        metrics: FairnessMetrics,
        threshold: float,
    ) -> list[str]:
        recs = []
        if metrics.demographic_parity > threshold:
            recs.append("Consider rebalancing training data or applying fairness constraints")
        if min(metrics.sample_sizes.values()) < 100:
            recs.append("Small sample sizes may affect reliability of fairness metrics")
        return recs
