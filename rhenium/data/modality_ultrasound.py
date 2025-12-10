# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Rhenium OS Ultrasound Modality Support
======================================

Comprehensive Ultrasound modality handling, including DICOM, Cine loops,
and RF/IQ data representations.

Supported Categories:
- B-mode (2D Grayscale)
- Doppler (Color, Power, Spectral PW/CW)
- Echocardiography (2D, M-mode, Doppler, Strain)
- Elastography (Strain, Shear Wave)
- 3D/4D Volumetric Ultrasound
- Contrast-Enhanced Ultrasound (CEUS)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, List, Optional

import numpy as np


# =============================================================================
# Ultrasound Enumerations
# =============================================================================

class USProbeType(Enum):
    """Ultrasound transducer types."""
    LINEAR = "linear"
    CURVED = "curved"  # Convex
    PHASED_ARRAY = "phased_array"
    ENDOCAVITARY = "endocavitary"  # Transvaginal, Transrectal
    TRANSESOPHAGEAL = "tee"
    MATRIX_3D = "matrix_3d"
    PENCIL = "pencil"  # CW Doppler
    UNKNOWN = "unknown"


class USMode(Enum):
    """Ultrasound imaging modes."""
    B_MODE = "b_mode"
    M_MODE = "m_mode"
    COLOR_DOPPLER = "color_doppler"
    POWER_DOPPLER = "power_doppler"
    PULSED_WAVE_DOPPLER = "pw_doppler"
    CONTINUOUS_WAVE_DOPPLER = "cw_doppler"
    TISSUE_DOPPLER = "tissue_doppler"
    ELASTOGRAPHY_STRAIN = "elasto_strain"
    ELASTOGRAPHY_SHEAR_WAVE = "elasto_swe"
    CEUS = "ceus"
    VOLUMETRIC_3D = "3d"
    VOLUMETRIC_4D = "4d"  # 3D + time


class USRegion(Enum):
    """Anatomical regions for presets."""
    ABDOMEN = "abdomen"
    CARDIAC = "cardiac"
    OBSTETRICS = "obstetrics"
    GYNECOLOGY = "gynecology"
    VASCULAR = "vascular"
    BREAST = "breast"
    THYROID = "thyroid"
    MSK = "msk"
    LUNG = "lung"
    FAST = "fast"  # Trauma
    SMALL_PARTS = "small_parts"


# =============================================================================
# Ultrasound Acquisition Metadata
# =============================================================================

@dataclass
class USProbeInfo:
    """Detailed probe information."""
    name: str | None = None
    probe_type: USProbeType = USProbeType.UNKNOWN
    min_frequency_mhz: float | None = None
    max_frequency_mhz: float | None = None
    center_frequency_mhz: float | None = None
    element_count: int | None = None
    footprint_mm: float | None = None


@dataclass
class USAcquisitionParameters:
    """
    Ultrasound acquisition setting parameters.
    
    Captures scanner settings that affect image appearance and physics.
    """
    # Probe and Frequency
    probe_name: str | None = None
    frequency_mhz: float | None = None
    
    # Image Geometry
    depth_mm: float | None = None
    focus_depths_mm: List[float] = field(default_factory=list)
    sector_width_deg: float | None = None  # For phased/curved
    fov_width_mm: float | None = None  # For linear
    
    # Gain and Contrast
    gain_db: float | None = None
    dynamic_range_db: float | None = None
    mechanical_index: float | None = None
    thermal_index: float | None = None
    
    # Temporal
    frame_rate_hz: float | None = None
    cine_duration_sec: float | None = None
    
    # Doppler Specific
    prf_hz: float | None = None  # Pulse Repetition Frequency
    wall_filter_hz: float | None = None
    doppler_angle_deg: float | None = None
    
    # Processing
    speckle_reduction_on: bool = False
    compound_imaging_on: bool = False
    harmonic_imaging_on: bool = False


# =============================================================================
# Ultrasound Data Structures
# =============================================================================

@dataclass
class USFrameInfo:
    """Metadata for a single ultrasound frame or image."""
    sop_instance_uid: str
    instance_number: int
    acquisition_datetime: str | None = None
    
    # Pixel Data
    rows: int = 0
    columns: int = 0
    physical_delta_x_mm: float | None = None  # Lateral resolution
    physical_delta_y_mm: float | None = None  # Axial resolution
    
    # Region calibration (for defining ROI in mixed mode images)
    regions: List[dict] = field(default_factory=list)
    
    # Interpretation
    photometric_interpretation: str = "MONOCHROME2"  # or RGB/YBR for Doppler
    
    def get_pixel_spacing(self) -> tuple[float, float] | None:
        """Get (y, x) spacing in mm."""
        if self.physical_delta_y_mm and self.physical_delta_x_mm:
            return (self.physical_delta_y_mm, self.physical_delta_x_mm)
        return None


@dataclass
class USSeriesInfo:
    """
    Collection of US frames or cine loops.
    """
    series_uid: str
    modality: str = "US"
    series_description: str | None = None
    
    # Classification
    mode: USMode = USMode.B_MODE
    region: USRegion = USRegion.ABDOMEN
    laterality: str | None = None  # Left, Right
    
    # Data
    frames: List[USFrameInfo] = field(default_factory=list)
    is_cine: bool = False
    
    # Raw Data Reference (Optional)
    rf_data_path: str | None = None
    iq_data_path: str | None = None


# =============================================================================
# Ultrasound Preprocessing
# =============================================================================

@dataclass
class USPreprocessingConfig:
    """Configuration for ultrasound preprocessing."""
    # Intensity normalization
    normalize_intensity: bool = True
    target_mean_intensity: float = 0.5
    
    # Speckle reduction (simple filters)
    apply_speckle_filter: bool = False
    speckle_filter_type: str = "median"  # median, anisotropic
    
    # Cine processing
    temporal_smoothing: bool = False
    
    # Doppler
    extract_color_overlay: bool = True


class USPreprocessor:
    """
    Preprocessing pipeline for Ultrasound data.
    
    Handles:
    - Log compression adjustment (conceptually)
    - Scan conversion (if raw polar data)
    - Speckle noise reduction
    - Doppler overlay separation
    """
    
    def __init__(self, config: USPreprocessingConfig | None = None):
        self.config = config or USPreprocessingConfig()
        
    def preprocess_bmode(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess B-mode grayscale image.
        """
        # Ensure float32 [0, 1]
        if image.dtype == np.uint8:
            image = image.astype(np.float32) / 255.0
            
        if self.config.apply_speckle_filter:
            image = self._apply_speckle_filter(image)
            
        return image
    
    def separate_doppler(self, frame_color: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Separate B-mode background and Color Doppler overlay.
        
        Args:
            frame_color: RGB ultrasound frame
            
        Returns:
            (bmode_gray, doppler_overlay)
        """
        # Simplistic heuristic:
        # B-mode is grayscale (R~=G~=B), Doppler is colored
        
        # Convert to float
        if frame_color.dtype == np.uint8:
            frame = frame_color.astype(np.float32) / 255.0
        else:
            frame = frame_color
            
        # Variance channel-wise
        variance = np.var(frame, axis=2)
        
        # Grayscale mask (low variance)
        gray_mask = variance < 0.01
        
        bmode = np.mean(frame, axis=2)
        
        # Isolate doppler (colored pixels)
        doppler = frame.copy()
        doppler[gray_mask] = 0
        
        return bmode, doppler
        
    def _apply_speckle_filter(self, image: np.ndarray) -> np.ndarray:
        """Apply basic speckle reduction filter."""
        from scipy.ndimage import median_filter
        if self.config.speckle_filter_type == "median":
            return median_filter(image, size=3)
        return image
