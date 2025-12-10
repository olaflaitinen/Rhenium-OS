# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Rhenium OS Ultrasound Beamforming
==================================

Digital beamforming and image formation pipelines.
Converts RF (Radio Frequency) or IQ (In-phase/Quadrature) data 
into B-mode images.
"""

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class BeamformingConfig:
    """Configuration for US beamforming."""
    sound_speed_m_s: float = 1540.0
    sampling_rate_hz: float = 40e6  # 40 MHz
    f_number: float = 1.0  # Transmit f-number
    apodization_window: str = "hamming"
    scan_conversion_interp: str = "linear"  # linear, cubic


class Beamformer(ABC):
    """Abstract base class for ultrasound beamformers."""
    
    @abstractmethod
    def process_rf(self, rf_data: np.ndarray, config: BeamformingConfig) -> np.ndarray:
        """
        Process RF channel data to beamformed RF lines.
        
        Args:
            rf_data: Dimensions [channels, samples, lines]
        """
        pass

class DASBeamformer(Beamformer):
    """
    Delay-and-Sum (DAS) Beamformer.
    
    Standard method for medical ultrasound.
    1. Apply time delays to align signals from focus point.
    2. Apply apodization (weighting).
    3. Sum channels.
    """
    
    def process_rf(self, rf_data: np.ndarray, config: BeamformingConfig) -> np.ndarray:
        """Process RF data using DAS."""
        # Conceptual implementation
        # In practice would use numba/cuda for speed
        channels, samples, lines = rf_data.shape
        return np.sum(rf_data, axis=0)  # Naive sum without delays for placeholder


class ImageFormation:
    """
    Post-beamforming image formation chain.
    
    RF Line -> Envelope Detection -> Log Compression -> Scan Conversion -> B-mode
    """
    
    @staticmethod
    def envelope_detection(rf_line: np.ndarray) -> np.ndarray:
        """
        Extract envelope using Hilbert transform.
        """
        from scipy.signal import hilbert
        analytic_signal = hilbert(rf_line, axis=0)
        envelope = np.abs(analytic_signal)
        return envelope
    
    @staticmethod
    def log_compression(envelope: np.ndarray, dynamic_range_db: float = 60) -> np.ndarray:
        """
        Log compression to reduce dynamic range for display.
        
        Output = 20 * log10(envelope / max) + dynamic_range
        """
        # Avoid log(0)
        env_norm = envelope / (np.max(envelope) + 1e-10)
        log_data = 20 * np.log10(env_norm + 1e-10)
        
        # Clip to dynamic range
        compressed = np.clip(log_data + dynamic_range_db, 0, dynamic_range_db)
        
        # Normalize to 0-1
        return compressed / dynamic_range_db


class ScanConverter:
    """
    Scan conversion from polar/probe geometry to Cartesian image grid.
    """
    
    def convert_sector(
        self,
        b_lines: np.ndarray,
        radius_start: float,
        radius_end: float,
        angle_start_rad: float,
        angle_end_rad: float,
        output_shape: tuple[int, int]
    ) -> np.ndarray:
        """
        Convert sector scan (phased/curved) to Cartesian.
        """
        # Conceptual: coordinate mapping + interpolation
        return np.zeros(output_shape, dtype=np.float32)

    def convert_linear(
        self,
        b_lines: np.ndarray,
        width_mm: float,
        depth_mm: float,
        output_shape: tuple[int, int]
    ) -> np.ndarray:
        """
        Convert linear scan to Cartesian (mostly resizing/anisotropic scaling).
        """
        from scipy.ndimage import zoom
        # Placeholder for aspect ratio correction
        return b_lines
