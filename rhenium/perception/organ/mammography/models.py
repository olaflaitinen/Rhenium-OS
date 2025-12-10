# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Rhenium OS Mammography Perception
=================================

Models for Mammography (FFDM / DBT):
- Mass / Calcification Detection
- Breast Density Estimation
- BIRADS Category Prediction
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

@dataclass
class MammoFinding:
    """Mammography finding."""
    type: str  # Mass, Calcification, Architecture Distortion, Asymmetry
    location_bbox: tuple[float, float, float, float]
    view: str  # CC or MLO
    laterality: str # L or R
    confidence: float
    birads_prob: float | None = None
    
@dataclass
class MammoDensity:
    """Breast density assessment."""
    birads_density: str # A, B, C, D
    dense_area_percent: float | None = None


class MammoLesionDetector:
    """
    Lesion detection in mammography.
    """
    
    def detect(self, image: np.ndarray, view: str) -> list[MammoFinding]:
        """Detect lesions."""
        return []


class BreastDensityEstimator:
    """
    Automated breast density classification.
    """
    
    def estimate(self, image: np.ndarray) -> MammoDensity:
        """Estimate density."""
        return MammoDensity(birads_density="B")
