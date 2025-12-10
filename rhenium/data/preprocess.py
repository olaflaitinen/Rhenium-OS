# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Preprocessing Module
====================

Standard preprocessing utilities for medical imaging data: normalization,
resampling, cropping/padding, and spatial transforms.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable

import numpy as np
from scipy import ndimage

from rhenium.core.logging import get_data_logger

logger = get_data_logger()


class NormalizationMethod(str, Enum):
    """Intensity normalization methods."""
    ZSCORE = "zscore"
    MINMAX = "minmax"
    PERCENTILE = "percentile"
    HISTOGRAM = "histogram"


@dataclass
class PreprocessingStep:
    """Record of a preprocessing step for audit trail."""
    name: str
    parameters: dict
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]


@dataclass
class PreprocessingPipeline:
    """Configurable preprocessing pipeline with audit logging."""
    steps: list[Callable] = field(default_factory=list)
    history: list[PreprocessingStep] = field(default_factory=list)

    def add_step(self, step: Callable) -> "PreprocessingPipeline":
        """Add a preprocessing step."""
        self.steps.append(step)
        return self

    def run(self, data: np.ndarray) -> np.ndarray:
        """Execute all preprocessing steps."""
        result = data
        for step in self.steps:
            result = step(result)
        return result


def normalize_intensity(
    data: np.ndarray,
    method: NormalizationMethod = NormalizationMethod.ZSCORE,
    percentile_low: float = 1.0,
    percentile_high: float = 99.0,
) -> np.ndarray:
    """
    Normalize intensity values.

    Args:
        data: Input array.
        method: Normalization method.
        percentile_low: Lower percentile for clipping.
        percentile_high: Upper percentile for clipping.

    Returns:
        Normalized array.
    """
    data = data.astype(np.float32)

    if method == NormalizationMethod.ZSCORE:
        mean = np.mean(data)
        std = np.std(data)
        return (data - mean) / (std + 1e-8)

    elif method == NormalizationMethod.MINMAX:
        data_min = np.min(data)
        data_max = np.max(data)
        return (data - data_min) / (data_max - data_min + 1e-8)

    elif method == NormalizationMethod.PERCENTILE:
        p_low = np.percentile(data, percentile_low)
        p_high = np.percentile(data, percentile_high)
        data = np.clip(data, p_low, p_high)
        return (data - p_low) / (p_high - p_low + 1e-8)

    return data


def resample_volume(
    data: np.ndarray,
    current_spacing: tuple[float, ...],
    target_spacing: tuple[float, ...],
    order: int = 1,
) -> np.ndarray:
    """
    Resample volume to target voxel spacing.

    Args:
        data: Input volume.
        current_spacing: Current voxel spacing.
        target_spacing: Target voxel spacing.
        order: Interpolation order (0=nearest, 1=linear, 3=cubic).

    Returns:
        Resampled volume.
    """
    zoom_factors = tuple(c / t for c, t in zip(current_spacing, target_spacing))
    return ndimage.zoom(data, zoom_factors, order=order)


def crop_or_pad(
    data: np.ndarray,
    target_shape: tuple[int, ...],
    pad_value: float = 0.0,
) -> np.ndarray:
    """
    Crop or pad volume to target shape (centered).

    Args:
        data: Input array.
        target_shape: Target shape.
        pad_value: Value for padding.

    Returns:
        Cropped/padded array.
    """
    result = np.full(target_shape, pad_value, dtype=data.dtype)

    # Calculate slices for both arrays
    data_slices = []
    result_slices = []

    for i, (d_size, t_size) in enumerate(zip(data.shape, target_shape)):
        if d_size >= t_size:
            # Crop: center crop from data
            start = (d_size - t_size) // 2
            data_slices.append(slice(start, start + t_size))
            result_slices.append(slice(None))
        else:
            # Pad: center place into result
            start = (t_size - d_size) // 2
            data_slices.append(slice(None))
            result_slices.append(slice(start, start + d_size))

    result[tuple(result_slices)] = data[tuple(data_slices)]
    return result


def window_level(
    data: np.ndarray,
    window_center: float,
    window_width: float,
    output_range: tuple[float, float] = (0.0, 1.0),
) -> np.ndarray:
    """Apply window/level transform for CT/radiograph display."""
    low = window_center - window_width / 2
    high = window_center + window_width / 2
    data = np.clip(data, low, high)
    data = (data - low) / (high - low)
    return data * (output_range[1] - output_range[0]) + output_range[0]
