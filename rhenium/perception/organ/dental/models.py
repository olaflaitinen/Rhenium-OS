# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Rhenium OS Dental X-ray Perception
==================================

Models for Dental Radiography:
- Tooth Detection and Numbering (FDI/Universal)
- Caries Detection
- Periapical Lesion Detection
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

@dataclass
class DentalFinding:
    """Dental pathology."""
    type: str  # Caries, Periapical Lesion, Bone Loss
    tooth_number: int  # FDI notation (e.g. 11, 26)
    location_bbox: tuple[float, float, float, float] | None = None
    confidence: float = 0.0


class ToothDetector:
    """
    Detects and numbers teeth in Panoramic or Intraoral X-rays.
    """
    
    def detect_teeth(self, image: np.ndarray) -> list[dict]:
        """Return list of tooth bounding boxes and numbers."""
        return []


class CariesDetector:
    """
    Detects carious lesions (cavities).
    """
    
    def detect(self, image: np.ndarray) -> list[DentalFinding]:
        """Detect caries."""
        return []
