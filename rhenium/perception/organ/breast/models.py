# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Rhenium OS Breast Ultrasound
============================

Breast lesion detection and characterization (BI-RADS).
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

@dataclass
class BreastLesion:
    """Breast lesion finding (BI-RADS features)."""
    shape: str        # Oval, Round, Irregular
    orientation: str  # Parallel, Not parallel
    margin: str       # Circumscribed, Indistinct, Spiculated
    echo_pattern: str # Anechoic, Hyperechoic, etc.
    
    birads_category: str | None = None


class BreastLesionDetector:
    """Breast ultrasound lesion detection."""
    def detect(self, image: np.ndarray) -> list[BreastLesion]:
        return []
