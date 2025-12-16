"""Evaluation metrics for medical imaging."""

from __future__ import annotations
import numpy as np
import torch
from typing import Literal


def dice_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5) -> float:
    """Dice similarity coefficient."""
    intersection = (pred * target).sum()
    return float(2.0 * intersection / (pred.sum() + target.sum() + smooth))


def iou_score(pred: np.ndarray, target: np.ndarray, smooth: float = 1e-5) -> float:
    """Intersection over Union (Jaccard)."""
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return float(intersection / (union + smooth))


def hausdorff_distance_95(pred: np.ndarray, target: np.ndarray) -> float:
    """95th percentile Hausdorff distance."""
    from scipy.ndimage import distance_transform_edt
    if pred.sum() == 0 or target.sum() == 0:
        return float('inf')
    pred_dist = distance_transform_edt(~pred.astype(bool))
    target_dist = distance_transform_edt(~target.astype(bool))
    pred_to_target = pred_dist[target > 0]
    target_to_pred = target_dist[pred > 0]
    return float(max(np.percentile(pred_to_target, 95), np.percentile(target_to_pred, 95)))


def psnr(pred: np.ndarray, target: np.ndarray, max_val: float = 1.0) -> float:
    """Peak Signal-to-Noise Ratio."""
    mse = ((pred - target) ** 2).mean()
    if mse == 0:
        return float('inf')
    return float(10 * np.log10(max_val ** 2 / mse))


def ssim(pred: np.ndarray, target: np.ndarray) -> float:
    """Structural Similarity Index (simplified 2D)."""
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    mu_p, mu_t = pred.mean(), target.mean()
    sigma_p, sigma_t = pred.std(), target.std()
    sigma_pt = ((pred - mu_p) * (target - mu_t)).mean()
    return float((2 * mu_p * mu_t + C1) * (2 * sigma_pt + C2) /
                 ((mu_p ** 2 + mu_t ** 2 + C1) * (sigma_p ** 2 + sigma_t ** 2 + C2)))


def auroc(pred_probs: np.ndarray, targets: np.ndarray) -> float:
    """Area Under ROC Curve."""
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(targets, pred_probs))


def expected_calibration_error(
    probs: np.ndarray, targets: np.ndarray, n_bins: int = 10
) -> float:
    """Expected Calibration Error."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() > 0:
            acc = targets[mask].mean()
            conf = probs[mask].mean()
            ece += mask.sum() * abs(acc - conf)
    return float(ece / len(probs))
