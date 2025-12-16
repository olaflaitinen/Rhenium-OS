"""Calibration metrics and analysis."""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass


@dataclass
class CalibrationResult:
    """Calibration analysis result."""
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    bin_confidences: np.ndarray
    bin_accuracies: np.ndarray
    bin_counts: np.ndarray


def expected_calibration_error(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> CalibrationResult:
    """
    Compute Expected Calibration Error (ECE).

    ECE = Σ (|B_m| / n) * |acc(B_m) - conf(B_m)|

    Args:
        probs: Predicted probabilities
        labels: True binary labels
        n_bins: Number of calibration bins

    Returns:
        CalibrationResult with metrics
    """
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(probs, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_confidences = np.zeros(n_bins)
    bin_accuracies = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins)

    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            bin_confidences[i] = probs[mask].mean()
            bin_accuracies[i] = labels[mask].mean()
            bin_counts[i] = mask.sum()

    # ECE: weighted average of calibration errors
    weights = bin_counts / len(probs)
    ece = (weights * np.abs(bin_accuracies - bin_confidences)).sum()

    # MCE: maximum calibration error
    nonzero = bin_counts > 0
    mce = np.abs(bin_accuracies[nonzero] - bin_confidences[nonzero]).max() if nonzero.any() else 0

    return CalibrationResult(
        ece=float(ece),
        mce=float(mce),
        bin_confidences=bin_confidences,
        bin_accuracies=bin_accuracies,
        bin_counts=bin_counts,
    )


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute Brier Score.

    BS = (1/n) * Σ(p_i - y_i)²

    Lower is better. Range [0, 1].
    """
    return float(np.mean((probs - labels) ** 2))


def reliability_diagram_data(
    probs: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 10,
) -> dict[str, np.ndarray]:
    """Get data for plotting reliability diagram."""
    result = expected_calibration_error(probs, labels, n_bins)
    return {
        "confidences": result.bin_confidences,
        "accuracies": result.bin_accuracies,
        "counts": result.bin_counts,
        "perfect": np.linspace(0, 1, n_bins),
    }


class TemperatureScaling:
    """Temperature scaling for probability calibration."""

    def __init__(self, temperature: float = 1.0):
        self.temperature = temperature

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling to logits."""
        scaled = logits / self.temperature
        return self._softmax(scaled)

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """Find optimal temperature using NLL."""
        from scipy.optimize import minimize_scalar

        def nll(t: float) -> float:
            scaled = logits / t
            probs = self._softmax(scaled)
            return -np.mean(labels * np.log(probs + 1e-10))

        result = minimize_scalar(nll, bounds=(0.1, 10), method='bounded')
        self.temperature = result.x
        return self.temperature

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - x.max(axis=-1, keepdims=True))
        return exp_x / exp_x.sum(axis=-1, keepdims=True)
