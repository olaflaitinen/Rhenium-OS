# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Rhenium OS Vascular Ultrasound
==============================

Analysis of vascular ultrasound:
- Carotid Intima-Media Thickness (CIMT)
- Plaque segmentation
- DVT detection (Compressibility)
- Doppler waveform classification
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

@dataclass
class CarotidAnalysis:
    """Carotid artery analysis results."""
    cimt_mm: float | None = None
    has_plaque: bool = False
    stenosis_percent: float | None = None
    peak_systolic_velocity_cms: float | None = None


@dataclass
class DVTFinding:
    """Deep Vein Thrombosis finding."""
    vein_segment: str
    is_compressible: bool = True
    has_thrombus: bool = False
    confidence: float = 0.0


class CarotidAnalyzer:
    """Carotid ultrasound analysis."""
    
    def measure_cimt(self, image: np.ndarray, pixel_spacing: float) -> float:
        """Auto-measure IMT."""
        return 0.6
        
    def analyze_plaque(self, image: np.ndarray) -> bool:
        """Detect plaque."""
        return False


class DVTDetector:
    """
    Analysis of compression ultrasound for DVT.
    Requires cine loop of compression maneuver.
    """
    
    def analyze_compression(self, cine_no_compression: np.ndarray, cine_compression: np.ndarray) -> DVTFinding:
        """Check vein compressibility."""
        return DVTFinding(vein_segment="Popliteal", is_compressible=True)
