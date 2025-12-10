# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Rhenium OS X-ray Preprocessing
==============================

Standardization of X-ray images for analysis:
- Intensity Normalization
- Geometric Standardization (Orientation)
- Auto-cropping (collimate removal)
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass

@dataclass
class XRayPreprocessingConfig:
    """Configuration for X-ray preprocessing."""
    target_size: tuple[int, int] | None = (1024, 1024)
    preserve_aspect_ratio: bool = True
    normalization_method: str = "min_max"  # min_max, z_score, histogram_match
    invert_monochrome1: bool = True
    crop_borders: bool = False


class XRayPreprocessor:
    """
    X-ray specific image preprocessing pipeline.
    """
    
    def __init__(self, config: XRayPreprocessingConfig | None = None):
        self.config = config or XRayPreprocessingConfig()
        
    def process(
        self,
        image: np.ndarray,
        photometric_interpretation: str = "MONOCHROME2"
    ) -> np.ndarray:
        """
        Full preprocessing chain.
        """
        # 1. Photometric Interpretation
        if self.config.invert_monochrome1 and photometric_interpretation == "MONOCHROME1":
            image = np.max(image) - image
            
        # 2. Border Cropping (Collimation)
        if self.config.crop_borders:
            image = self._crop_borders(image)
            
        # 3. Normalization
        image = self._normalize(image)
        
        # 4. Resizing (if using DL models)
        if self.config.target_size:
            image = self._resize(image)
            
        return image
    
    def _normalize(self, image: np.ndarray) -> np.ndarray:
        """Normalize intensity."""
        if self.config.normalization_method == "min_max":
            img_min = np.min(image)
            img_max = np.max(image)
            if img_max > img_min:
                return (image - img_min) / (img_max - img_min)
            return np.zeros_like(image, dtype=np.float32)
        elif self.config.normalization_method == "z_score":
            mean = np.mean(image)
            std = np.std(image)
            if std > 0:
                return (image - mean) / std
            return image
        return image
        
    def _crop_borders(self, image: np.ndarray) -> np.ndarray:
        """
        Remove dark collimation areas.
        Simple logic: thresholding or variance detection.
        """
        # Placeholder for robust collimation detection
        return image

    def _resize(self, image: np.ndarray) -> np.ndarray:
        """Resize image."""
        # Simple resize placeholder
        # In production, use cv2.resize or similar with interpolation
        from scipy.ndimage import zoom
        if self.config.target_size:
            h, w = image.shape
            th, tw = self.config.target_size
            return zoom(image, (th/h, tw/w))
        return image
