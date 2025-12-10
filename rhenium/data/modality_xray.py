# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Rhenium OS X-ray Modality Support
=================================

Comprehensive X-ray modality handling for Projection Radiography.

Supported Categories:
- General Radiography: Chest (CXR), Abdomen (AXR), MSK, Skull
- Digital Radiography: CR, DX, Portable
- Mammography: FFDM, DBT (Projections)
- Dental: Intraoral, Panoramic
- Fluoroscopy: Single-frame snapshots
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, List, Optional

import numpy as np


# =============================================================================
# X-ray Enumerations
# =============================================================================

class XRayModalityType(Enum):
    """DICOM Modality types for X-ray."""
    CR = "CR"  # Computed Radiography
    DX = "DX"  # Digital Radiography
    MG = "MG"  # Mammography
    IO = "IO"  # Intra-oral Radiography
    PX = "PX"  # Panoramic X-ray
    RF = "RF"  # Radiofluoroscopy (snapshots)
    XA = "XA"  # X-ray Angiography (snapshots)
    UNKNOWN = "UNKNOWN"


class XRayBodyPart(Enum):
    """Examined body part."""
    CHEST = "CHEST"
    ABDOMEN = "ABDOMEN"
    PELVIS = "PELVIS"
    SPINE = "SPINE"
    SKULL = "SKULL"
    EXTREMITY = "EXTREMITY"  # General extremity
    HAND = "HAND"
    WRIST = "WRIST"
    KNEE = "KNEE"
    FOOT = "FOOT"
    SHOULDER = "SHOULDER"
    BREAST = "BREAST"
    HEAD_NECK = "HEAD_NECK"  # Dental/Maxillofacial
    DENTAL = "DENTAL"
    UNKNOWN = "UNKNOWN"


class XRayProjection(Enum):
    """Projection/View orientation."""
    PA = "PA"  # Postero-Anterior
    AP = "AP"  # Antero-Posterior
    LATERAL = "LATERAL"
    OBLIQUE = "OBLIQUE"
    DECUBITUS = "DECUBITUS"
    AXIAL = "AXIAL"  # e.g., for shoulders
    UNKNOWN = "UNKNOWN"
    
    # Mammography specific
    CC = "CC"   # Cranio-Caudal
    MLO = "MLO" # Medio-Lateral Oblique


# =============================================================================
# X-ray Acquisition Metadata
# =============================================================================

@dataclass
class XRayAcquisitionParameters:
    """
    X-ray acquisition technique parameters.
    """
    # Generator settings
    kvp: float | None = None
    mas: float | None = None
    exposure_time_ms: float | None = None
    exposure_index: float | None = None  # EI
    
    # Geometry
    sid_mm: float | None = None  # Source Image Distance
    sod_mm: float | None = None  # Source Object Distance
    
    # Detector
    detector_type: str | None = None
    pixel_spacing_mm: tuple[float, float] | None = None
    imager_pixel_spacing_mm: tuple[float, float] | None = None
    
    # Grid
    grid_used: bool = False
    
    # Mammography specific
    compression_force_n: float | None = None
    body_part_thickness_mm: float | None = None
    anode_target_material: str | None = None
    filter_material: str | None = None


@dataclass
class XRayImageInfo:
    """
    Metadata for a single X-ray image (instance).
    """
    sop_instance_uid: str
    series_instance_uid: str
    study_instance_uid: str
    
    modality: XRayModalityType = XRayModalityType.DX
    body_part: XRayBodyPart = XRayBodyPart.UNKNOWN
    projection: XRayProjection = XRayProjection.UNKNOWN
    view_position: str | None = None  # DICOM ViewPosition (e.g., "PA")
    laterality: str | None = None     # 'L', 'R', or None
    
    # Acquisition
    params: XRayAcquisitionParameters = field(default_factory=XRayAcquisitionParameters)
    
    # Image characteristics
    rows: int = 0
    columns: int = 0
    photometric_interpretation: str = "MONOCHROME2"
    bits_allocated: int = 16
    bits_stored: int = 12
    pixel_representation: int = 0  # 0=unsigned, 1=signed
    
    # Transforms
    window_center: float | None = None
    window_width: float | None = None
    rescale_slope: float = 1.0
    rescale_intercept: float = 0.0
    
    # Presentation state
    voi_lut_function: str = "LINEAR"


@dataclass
class XRayStudyMetadata:
    """
    Metadata for an X-ray study (exam).
    """
    study_date: str | None = None
    accession_number: str | None = None
    images: List[XRayImageInfo] = field(default_factory=list)
    
    def get_images_by_projection(self, proj: XRayProjection) -> List[XRayImageInfo]:
        """Filter images by projection."""
        return [img for img in self.images if img.projection == proj]


# =============================================================================
# X-ray Preprocessing Interfaces
# =============================================================================

class XRayPhotometricInterpreter:
    """
    Handles pixel value interpretation and inversion.
    """
    
    @staticmethod
    def ensure_monochrome(
        pixel_array: np.ndarray,
        photometric_interpretation: str
    ) -> np.ndarray:
        """
        Convert to MONOCHROME2 (White is dense/bone, Black is air) convention if needed.
        
        Note:
        - MONOCHROME1: Min value = White (standard for Mammography prints)
        - MONOCHROME2: Min value = Black (standard for CXR/CT display)
        
        If MONOCHROME1, usually needs inversion for standard specific digital viewing.
        """
        if photometric_interpretation == "MONOCHROME1":
            return np.max(pixel_array) - pixel_array
        return pixel_array
