# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Quantitative Explanations
=========================

Measurement extraction, radiomics features, and uncertainty quantification.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

from rhenium.xai.explanation_schema import QuantitativeEvidence
from rhenium.core.logging import get_xai_logger

logger = get_xai_logger()


def compute_measurements(
    mask: np.ndarray,
    spacing: tuple[float, ...] = (1.0, 1.0, 1.0),
) -> QuantitativeEvidence:
    """
    Compute measurements from a segmentation mask.

    Args:
        mask: Binary segmentation mask.
        spacing: Voxel spacing in mm.

    Returns:
        QuantitativeEvidence with measurements.
    """
    logger.info("Computing measurements from mask")

    measurements = {}
    units = {}

    # Volume
    voxel_volume = np.prod(spacing)
    volume_voxels = np.sum(mask > 0)
    volume_mm3 = volume_voxels * voxel_volume
    measurements["volume_mm3"] = float(volume_mm3)
    measurements["volume_ml"] = float(volume_mm3 / 1000)
    units["volume_mm3"] = "mm^3"
    units["volume_ml"] = "mL"

    # Bounding box dimensions
    if mask.any():
        coords = np.argwhere(mask > 0)
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)
        dimensions = (max_coords - min_coords + 1) * np.array(spacing[:len(min_coords)])

        measurements["max_diameter_mm"] = float(np.max(dimensions))
        measurements["dimension_x_mm"] = float(dimensions[0]) if len(dimensions) > 0 else 0
        measurements["dimension_y_mm"] = float(dimensions[1]) if len(dimensions) > 1 else 0
        measurements["dimension_z_mm"] = float(dimensions[2]) if len(dimensions) > 2 else 0
        units["max_diameter_mm"] = "mm"

    # Surface area (simplified)
    if mask.any():
        from scipy.ndimage import binary_erosion
        surface = mask.astype(bool) ^ binary_erosion(mask.astype(bool))
        surface_voxels = np.sum(surface)
        measurements["surface_area_mm2"] = float(surface_voxels * (spacing[0] * spacing[1]))
        units["surface_area_mm2"] = "mm^2"

    return QuantitativeEvidence(
        measurements=measurements,
        units=units,
    )


def compute_radiomics_features(
    image: np.ndarray,
    mask: np.ndarray,
) -> QuantitativeEvidence:
    """
    Compute basic radiomics features.

    Args:
        image: Image array.
        mask: Region of interest mask.

    Returns:
        QuantitativeEvidence with radiomics features.
    """
    logger.info("Computing radiomics features")

    roi_values = image[mask > 0]

    if len(roi_values) == 0:
        return QuantitativeEvidence(radiomics_features={})

    features = {
        "mean_intensity": float(np.mean(roi_values)),
        "std_intensity": float(np.std(roi_values)),
        "min_intensity": float(np.min(roi_values)),
        "max_intensity": float(np.max(roi_values)),
        "median_intensity": float(np.median(roi_values)),
        "skewness": float(_compute_skewness(roi_values)),
        "kurtosis": float(_compute_kurtosis(roi_values)),
        "entropy": float(_compute_entropy(roi_values)),
    }

    return QuantitativeEvidence(radiomics_features=features)


def compute_uncertainty_metrics(
    predictions: list[np.ndarray],
) -> QuantitativeEvidence:
    """
    Compute uncertainty from ensemble predictions.

    Args:
        predictions: List of prediction arrays from ensemble.

    Returns:
        QuantitativeEvidence with uncertainty metrics.
    """
    logger.info("Computing uncertainty metrics")

    if not predictions:
        return QuantitativeEvidence(uncertainty=1.0)

    stacked = np.stack(predictions, axis=0)
    mean_pred = np.mean(stacked, axis=0)
    std_pred = np.std(stacked, axis=0)

    measurements = {
        "mean_uncertainty": float(np.mean(std_pred)),
        "max_uncertainty": float(np.max(std_pred)),
        "prediction_variance": float(np.var(stacked)),
    }

    # Entropy for classification
    if mean_pred.ndim == 1:
        entropy = -np.sum(mean_pred * np.log(mean_pred + 1e-10))
        measurements["prediction_entropy"] = float(entropy)

    return QuantitativeEvidence(
        uncertainty=float(np.mean(std_pred)),
        measurements=measurements,
    )


def _compute_skewness(values: np.ndarray) -> float:
    """Compute skewness of values."""
    n = len(values)
    if n < 3:
        return 0.0
    mean = np.mean(values)
    std = np.std(values)
    if std == 0:
        return 0.0
    return float(np.mean(((values - mean) / std) ** 3))


def _compute_kurtosis(values: np.ndarray) -> float:
    """Compute kurtosis of values."""
    n = len(values)
    if n < 4:
        return 0.0
    mean = np.mean(values)
    std = np.std(values)
    if std == 0:
        return 0.0
    return float(np.mean(((values - mean) / std) ** 4) - 3)


def _compute_entropy(values: np.ndarray, bins: int = 64) -> float:
    """Compute entropy of intensity distribution."""
    hist, _ = np.histogram(values, bins=bins, density=True)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist + 1e-10)))
