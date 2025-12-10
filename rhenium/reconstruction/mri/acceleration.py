# Copyright (c) 2025 Skolyn LLC. All rights reserved.
# SPDX-License-Identifier: EUPL-1.1

"""
MRI Acceleration Module
=======================

Undersampling masks, k-space preprocessing, and acceleration utilities.
"""

from __future__ import annotations

from enum import Enum

import numpy as np


class MaskType(str, Enum):
    """Undersampling mask types."""
    RANDOM = "random"
    EQUISPACED = "equispaced"
    POISSON_DISK = "poisson_disk"


def generate_undersampling_mask(
    shape: tuple[int, int],
    acceleration: float,
    center_fraction: float = 0.08,
    mask_type: MaskType = MaskType.EQUISPACED,
    seed: int | None = None,
) -> np.ndarray:
    """
    Generate k-space undersampling mask.

    Args:
        shape: K-space shape (kx, ky).
        acceleration: Acceleration factor (e.g., 4 for 4x).
        center_fraction: Fraction of center lines to fully sample.
        mask_type: Type of undersampling pattern.
        seed: Random seed for reproducibility.

    Returns:
        Binary mask array.
    """
    if seed is not None:
        np.random.seed(seed)

    mask = np.zeros(shape, dtype=np.float32)
    num_cols = shape[1]
    num_center = int(num_cols * center_fraction)
    center_start = (num_cols - num_center) // 2

    # Always sample center lines
    mask[:, center_start:center_start + num_center] = 1

    if mask_type == MaskType.EQUISPACED:
        step = int(acceleration)
        for i in range(0, num_cols, step):
            mask[:, i] = 1

    elif mask_type == MaskType.RANDOM:
        remaining = num_cols - num_center
        num_to_sample = int(remaining / acceleration)
        indices = np.setdiff1d(np.arange(num_cols),
                               np.arange(center_start, center_start + num_center))
        selected = np.random.choice(indices, num_to_sample, replace=False)
        mask[:, selected] = 1

    return mask


def apply_undersampling(
    kspace: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Apply undersampling mask to k-space data."""
    return kspace * mask


def estimate_acceleration(mask: np.ndarray) -> float:
    """Estimate acceleration factor from mask."""
    return 1.0 / (mask.mean() + 1e-8)
