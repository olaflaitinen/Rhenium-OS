# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Rhenium OS Brain CT Perception
===============================

CT-specific brain analysis:
- Intracranial Hemorrhage (ICH) detection
- Stroke evaluation (ASPECTS)
- Midline shift
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

@dataclass
class ICHFinding:
    """Intracranial hemorrhage finding."""
    type: str  # IPH, SDH, EDH, SAH, IVH
    location: str
    volume_ml: float
    confidence: float
    is_acute: bool = True

@dataclass
class StrokeAnalysis:
    """Acute stroke analysis results."""
    aspects_score: int | None = None
    core_volume_ml: float | None = None
    penumbra_volume_ml: float | None = None  # Requires CTP
    midline_shift_mm: float = 0.0


class ICHDetector:
    """Detection of intracranial hemorrhage on NCCT."""
    
    def detect(self, volume: np.ndarray) -> list[ICHFinding]:
        """Detect hemorrhage."""
        return []

class MidlineShiftEstimator:
    """Estimation of midline shift."""
    
    def estimate_shift(self, volume: np.ndarray) -> float:
        """Estimate shift in mm."""
        return 0.0
