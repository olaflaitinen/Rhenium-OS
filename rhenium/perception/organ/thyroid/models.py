# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Rhenium OS Thyroid Ultrasound
=============================

Thyroid nodule detection and characterization (TI-RADS).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

@dataclass
class ThyroidNodule:
    """Thyroid nodule finding (TI-RADS features)."""
    composition: str  # Cystic, Spongiform, Mixed, Solid
    echogenicity: str  # Anechoic, Hyperechoic, Isoechoic, Hypoechoic
    shape: str        # Wider-than-tall, Taller-than-wide
    margins: str      # Smooth, Irregular
    echogenic_foci: str # None, Macrocalc, Microcalc
    
    tirads_score: int | None = None


class ThyroidNoduleDetector:
    """Detection and characterization of thyroid nodules."""
    def detect(self, image: np.ndarray) -> list[ThyroidNodule]:
        return []
