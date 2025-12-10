# Copyright (c) 2025 Skolyn LLC. All rights reserved.
# SPDX-License-Identifier: EUPL-1.1

"""
Evaluation Metrics
==================

Clinical metrics for segmentation, detection, classification, and image quality.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import distance_transform_edt


def dice_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Dice coefficient."""
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = np.sum(pred & target)
    return float(2 * intersection / (np.sum(pred) + np.sum(target) + 1e-8))


def iou_score(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Intersection over Union."""
    pred = pred.astype(bool)
    target = target.astype(bool)
    intersection = np.sum(pred & target)
    union = np.sum(pred | target)
    return float(intersection / (union + 1e-8))


def hausdorff_distance(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute 95th percentile Hausdorff distance."""
    pred = pred.astype(bool)
    target = target.astype(bool)

    if not pred.any() or not target.any():
        return float('inf')

    pred_dist = distance_transform_edt(~pred)
    target_dist = distance_transform_edt(~target)

    dist_pred_to_target = pred_dist[target]
    dist_target_to_pred = target_dist[pred]

    hd95 = max(
        np.percentile(dist_pred_to_target, 95) if len(dist_pred_to_target) > 0 else 0,
        np.percentile(dist_target_to_pred, 95) if len(dist_target_to_pred) > 0 else 0,
    )
    return float(hd95)


def compute_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """Compute Area Under ROC Curve."""
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y_true, y_scores))


def compute_sensitivity_specificity(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[float, float]:
    """Compute sensitivity and specificity."""
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    sensitivity = tp / (tp + fn + 1e-8)
    specificity = tn / (tn + fp + 1e-8)
    return float(sensitivity), float(specificity)


def psnr(pred: np.ndarray, target: np.ndarray, max_val: float = 1.0) -> float:
    """Compute Peak Signal-to-Noise Ratio."""
    mse = np.mean((pred - target) ** 2)
    return float(20 * np.log10(max_val / np.sqrt(mse + 1e-8)))


def ssim(pred: np.ndarray, target: np.ndarray) -> float:
    """Compute Structural Similarity Index (simplified)."""
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    mu_x = np.mean(pred)
    mu_y = np.mean(target)
    sigma_x = np.var(pred)
    sigma_y = np.var(target)
    sigma_xy = np.cov(pred.flatten(), target.flatten())[0, 1]

    ssim_val = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / \
               ((mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2))
    return float(ssim_val)
