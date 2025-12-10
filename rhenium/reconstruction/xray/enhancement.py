# Copyright (c) 2025 Skolyn LLC. All rights reserved.
# SPDX-License-Identifier: EUPL-1.1

"""
X-ray Enhancement Pipeline
==========================

Noise reduction, contrast enhancement, and artifact removal for X-ray images.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter

from rhenium.core.logging import get_reconstruction_logger
from rhenium.core.registry import registry, ComponentType

logger = get_reconstruction_logger()


@dataclass
class XRayEnhancementConfig:
    """X-ray enhancement configuration."""
    denoise: bool = True
    denoise_method: str = "gaussian"
    denoise_sigma: float = 1.0
    contrast_enhance: bool = True
    clahe_clip_limit: float = 2.0
    grid_line_removal: bool = False


@dataclass
class XRayEnhancementPipeline:
    """
    X-ray image enhancement pipeline.

    Applies:
    - Noise reduction
    - Contrast enhancement (CLAHE-like)
    - Grid-line artifact removal
    """
    config: XRayEnhancementConfig = field(default_factory=XRayEnhancementConfig)

    def run(self, image: np.ndarray) -> np.ndarray:
        """Execute enhancement pipeline."""
        logger.info("Running X-ray enhancement")

        result = image.astype(np.float32)

        if self.config.denoise:
            result = self._denoise(result)

        if self.config.contrast_enhance:
            result = self._contrast_enhance(result)

        if self.config.grid_line_removal:
            result = self._remove_grid_lines(result)

        logger.info("X-ray enhancement complete")
        return result

    def _denoise(self, image: np.ndarray) -> np.ndarray:
        """Apply denoising."""
        if self.config.denoise_method == "gaussian":
            return gaussian_filter(image, sigma=self.config.denoise_sigma)
        elif self.config.denoise_method == "median":
            return median_filter(image, size=3)
        return image

    def _contrast_enhance(self, image: np.ndarray) -> np.ndarray:
        """Apply contrast enhancement (simplified CLAHE-like)."""
        p_low, p_high = np.percentile(image, [1, 99])
        image = np.clip(image, p_low, p_high)
        image = (image - p_low) / (p_high - p_low + 1e-8)
        return image

    def _remove_grid_lines(self, image: np.ndarray) -> np.ndarray:
        """Remove anti-scatter grid lines (placeholder)."""
        # Would use FFT-based filtering in production
        return image


registry.register(
    "xray_enhancement",
    ComponentType.RECONSTRUCTION,
    XRayEnhancementPipeline,
    version="1.0.0",
    description="X-ray image enhancement pipeline",
)
