# Copyright (c) 2025 Skolyn LLC. All rights reserved.
# SPDX-License-Identifier: EUPL-1.1

"""
Rhenium OS CT Modality Support
===============================

Comprehensive CT modality handling for all major CT subtypes and protocols.

Supported CT Categories:
- Non-contrast: Brain, Chest, Abdomen/Pelvis, Bone/Trauma
- Contrast-enhanced: Single-phase, Multi-phase, CTA
- CT Perfusion: Brain CTP, Cardiac perfusion
- Low-dose: Lung screening, Body protocols
- Dual-energy/Spectral: Virtual monoenergetic, Iodine maps, Material decomposition
- Cardiac: Coronary CTA, Calcium scoring
- Oncologic: Staging, Whole-body trauma
- Special: Colonography, Dental/Maxillofacial, Temporal bone

Skolyn: Early. Accurate. Trusted.

Last Updated: December 2025
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np


# =============================================================================
# CT Protocol and Series Type Enumeration
# =============================================================================

class CTCategory(Enum):
    """Top-level CT categories."""
    NON_CONTRAST = auto()
    CONTRAST_ENHANCED = auto()
    CT_ANGIOGRAPHY = auto()
    CT_PERFUSION = auto()
    LOW_DOSE = auto()
    DUAL_ENERGY = auto()
    CARDIAC = auto()
    COLONOGRAPHY = auto()
    UNKNOWN = auto()


class CTSeriesType(Enum):
    """
    Comprehensive enumeration of CT series types.
    
    Classification based on acquisition parameters, contrast timing,
    and clinical applications.
    """
    # Non-contrast CT
    NON_CONTRAST_HEAD = "nc_head"
    NON_CONTRAST_CHEST = "nc_chest"
    NON_CONTRAST_ABDOMEN = "nc_abdomen"
    NON_CONTRAST_PELVIS = "nc_pelvis"
    NON_CONTRAST_SPINE = "nc_spine"
    NON_CONTRAST_EXTREMITY = "nc_extremity"
    
    # Contrast-enhanced phases
    ARTERIAL_PHASE = "arterial"
    PORTAL_VENOUS_PHASE = "portal_venous"
    DELAYED_PHASE = "delayed"
    EQUILIBRIUM_PHASE = "equilibrium"
    NEPHROGENIC_PHASE = "nephrogenic"
    
    # CT Angiography
    CTA_HEAD_NECK = "cta_head_neck"
    CTA_CORONARY = "cta_coronary"
    CTA_PULMONARY = "cta_pulmonary"
    CTA_AORTA = "cta_aorta"
    CTA_RUNOFF = "cta_runoff"
    CTA_MESENTERIC = "cta_mesenteric"
    CTA_RENAL = "cta_renal"
    
    # CT Perfusion
    CTP_BRAIN = "ctp_brain"
    CTP_CARDIAC = "ctp_cardiac"
    CTP_LIVER = "ctp_liver"
    
    # Low-dose protocols
    LOW_DOSE_CHEST = "low_dose_chest"
    ULTRA_LOW_DOSE_CHEST = "ultra_low_dose_chest"
    LOW_DOSE_ABDOMEN = "low_dose_abdomen"
    
    # Dual-energy / Spectral CT
    DUAL_ENERGY_MIXED = "de_mixed"
    VIRTUAL_MONOENERGETIC = "virtual_mono"
    IODINE_MAP = "iodine_map"
    CALCIUM_MAP = "calcium_map"
    VIRTUAL_NON_CONTRAST = "vnc"
    MATERIAL_DECOMPOSITION = "material_decomp"
    
    # Cardiac CT
    CALCIUM_SCORE = "calcium_score"
    CCTA_PROSPECTIVE = "ccta_prospective"
    CCTA_RETROSPECTIVE = "ccta_retrospective"
    CARDIAC_FUNCTION = "cardiac_function"
    
    # Special protocols
    CT_COLONOGRAPHY = "colonography"
    DENTAL_CBCT = "dental_cbct"
    TEMPORAL_BONE = "temporal_bone"
    MAXILLOFACIAL = "maxillofacial"
    
    # Trauma
    WHOLE_BODY_TRAUMA = "whole_body_trauma"
    
    # Unknown
    UNKNOWN = "unknown"


class CTKernel(Enum):
    """CT reconstruction kernels."""
    SOFT_TISSUE = "soft"
    LUNG = "lung"
    BONE = "bone"
    BRAIN = "brain"
    VASCULAR = "vascular"
    STANDARD = "standard"
    SHARP = "sharp"
    SMOOTH = "smooth"
    ULTRA_SHARP = "ultra_sharp"


# =============================================================================
# CT Acquisition Parameters
# =============================================================================

@dataclass
class CTAcquisitionParameters:
    """
    CT acquisition parameters extracted from DICOM or protocol.
    
    These parameters are critical for:
    - Protocol classification
    - Dose estimation
    - Reconstruction optimization
    - Quality assessment
    """
    # Tube parameters
    kvp: float | None = None  # Peak kilovoltage
    ma: float | None = None  # Tube current (mA)
    mas: float | None = None  # Tube current-time product (mAs)
    exposure_time_ms: float | None = None
    
    # Geometry
    rotation_time_s: float | None = None
    pitch: float | None = None
    collimation_mm: float | None = None
    detector_rows: int | None = None
    
    # Reconstruction
    slice_thickness_mm: float | None = None
    slice_interval_mm: float | None = None
    reconstruction_diameter_mm: float | None = None
    kernel_name: str | None = None
    kernel_type: CTKernel = CTKernel.STANDARD
    
    # Field of view
    fov_mm: float | None = None
    matrix_size: tuple[int, int] | None = None
    pixel_spacing_mm: tuple[float, float] | None = None
    
    # Dose metrics (from DICOM Dose Report if available)
    ctdi_vol_mgy: float | None = None  # CTDIvol
    dlp_mgy_cm: float | None = None  # Dose-Length Product
    
    # Contrast
    contrast_agent: str | None = None
    contrast_volume_ml: float | None = None
    injection_rate_ml_s: float | None = None
    delay_s: float | None = None
    
    # Dual-energy specific
    is_dual_energy: bool = False
    kvp_low: float | None = None
    kvp_high: float | None = None
    
    # Cardiac specific
    is_ecg_gated: bool = False
    gating_type: str | None = None  # prospective, retrospective
    heart_rate_bpm: float | None = None
    
    # Vendor-specific
    manufacturer: str | None = None
    scanner_model: str | None = None
    protocol_name: str | None = None
    series_description: str | None = None
    
    @property
    def is_low_dose(self) -> bool:
        """Check if protocol is low-dose based on CTDIvol or mAs."""
        if self.ctdi_vol_mgy is not None:
            return self.ctdi_vol_mgy < 3.0  # Typical threshold for low-dose chest
        if self.mas is not None:
            return self.mas < 50
        return False
    
    @property
    def is_contrast_enhanced(self) -> bool:
        """Check if contrast was administered."""
        return self.contrast_agent is not None or self.delay_s is not None


# =============================================================================
# CT Volume and Study Structures
# =============================================================================

@dataclass
class CTSeriesInfo:
    """
    Information about a single CT series.
    
    Contains metadata, acquisition parameters, and derived classifications.
    """
    # Identification
    series_uid: str
    series_number: int
    series_description: str
    
    # Classification
    series_type: CTSeriesType = CTSeriesType.UNKNOWN
    category: CTCategory = CTCategory.UNKNOWN
    
    # Acquisition parameters
    acquisition_params: CTAcquisitionParameters = field(
        default_factory=CTAcquisitionParameters
    )
    
    # Geometry
    image_orientation: str = ""  # Axial, Coronal, Sagittal
    number_of_slices: int = 0
    
    # Data reference
    file_paths: list[str] = field(default_factory=list)
    volume_shape: tuple[int, ...] | None = None
    
    # Quality indicators
    estimated_noise_hu: float | None = None
    quality_flags: list[str] = field(default_factory=list)


@dataclass
class CTStudyMetadata:
    """
    Metadata for a complete CT study containing multiple series.
    
    Aggregates information across all series for multi-phase analysis.
    """
    # Study identification
    study_uid: str
    study_date: str
    study_time: str
    accession_number: str | None = None
    
    # Patient information (de-identified)
    patient_id_hash: str | None = None
    patient_age_years: int | None = None
    patient_sex: str | None = None
    
    # Series inventory
    series_list: list[CTSeriesInfo] = field(default_factory=list)
    
    # Scanner information
    institution_name: str | None = None
    manufacturer: str | None = None
    scanner_model: str | None = None
    
    # Study classification
    body_part: str | None = None
    study_type: str | None = None
    
    def get_series_by_type(self, series_type: CTSeriesType) -> list[CTSeriesInfo]:
        """Filter series by type."""
        return [s for s in self.series_list if s.series_type == series_type]
    
    def get_series_by_category(self, category: CTCategory) -> list[CTSeriesInfo]:
        """Filter series by category."""
        return [s for s in self.series_list if s.category == category]
    
    def has_multiphase_data(self) -> bool:
        """Check if study contains multi-phase contrast series."""
        phases = {s.series_type for s in self.series_list}
        contrast_phases = {
            CTSeriesType.ARTERIAL_PHASE,
            CTSeriesType.PORTAL_VENOUS_PHASE,
            CTSeriesType.DELAYED_PHASE,
        }
        return len(phases.intersection(contrast_phases)) >= 2


# =============================================================================
# CT Series Classification Engine
# =============================================================================

class CTSeriesClassifier:
    """
    Heuristic-based CT series classifier.
    
    Classifies CT series into types based on:
    - DICOM tags (SeriesDescription, ProtocolName)
    - Acquisition parameters (kVp, contrast timing, ECG gating)
    - Body part and anatomical region
    
    Note: Classification heuristics may require site-specific customization
    due to variations in naming conventions across institutions and vendors.
    """
    
    # Pattern matching for series descriptions
    _CTA_PATTERNS = ["cta", "angio", "runoff", "mra"]
    _CTP_PATTERNS = ["perfusion", "ctp", "perf"]
    _CALCIUM_PATTERNS = ["calcium", "cac", "score"]
    _CCTA_PATTERNS = ["ccta", "coronary", "cardiac_cta"]
    _COLONOGRAPHY_PATTERNS = ["colon", "colonography", "ctc", "virtual"]
    _LOW_DOSE_PATTERNS = ["low_dose", "ldct", "screening", "ultra_low"]
    _DUAL_ENERGY_PATTERNS = ["dual", "de_", "spectral", "gsi", "monoenergetic"]
    
    _ARTERIAL_PATTERNS = ["arterial", "art_", "a_phase", "early"]
    _VENOUS_PATTERNS = ["venous", "portal", "pv_", "v_phase"]
    _DELAYED_PATTERNS = ["delayed", "late", "equilibrium", "d_phase"]
    
    _HEAD_PATTERNS = ["head", "brain", "neuro"]
    _CHEST_PATTERNS = ["chest", "thorax", "lung"]
    _ABDOMEN_PATTERNS = ["abdomen", "abd", "liver", "pancreas"]
    _PELVIS_PATTERNS = ["pelvis", "pelvic"]
    _SPINE_PATTERNS = ["spine", "cervical", "thoracic", "lumbar"]
    
    @classmethod
    def classify(
        cls,
        params: CTAcquisitionParameters,
        series_description: str = "",
    ) -> tuple[CTSeriesType, CTCategory]:
        """
        Classify CT series based on parameters and description.
        
        Args:
            params: Acquisition parameters from DICOM
            series_description: Series description or protocol name
        
        Returns:
            Tuple of (series_type, category)
        """
        desc_lower = series_description.lower()
        
        # Check for dual-energy first
        if params.is_dual_energy or cls._matches_any(desc_lower, cls._DUAL_ENERGY_PATTERNS):
            if "iodine" in desc_lower:
                return CTSeriesType.IODINE_MAP, CTCategory.DUAL_ENERGY
            if "mono" in desc_lower:
                return CTSeriesType.VIRTUAL_MONOENERGETIC, CTCategory.DUAL_ENERGY
            if "vnc" in desc_lower:
                return CTSeriesType.VIRTUAL_NON_CONTRAST, CTCategory.DUAL_ENERGY
            return CTSeriesType.DUAL_ENERGY_MIXED, CTCategory.DUAL_ENERGY
        
        # Check for cardiac
        if params.is_ecg_gated or cls._matches_any(desc_lower, cls._CCTA_PATTERNS):
            if cls._matches_any(desc_lower, cls._CALCIUM_PATTERNS):
                return CTSeriesType.CALCIUM_SCORE, CTCategory.CARDIAC
            return CTSeriesType.CCTA_PROSPECTIVE, CTCategory.CARDIAC
        
        # Check for CT angiography
        if cls._matches_any(desc_lower, cls._CTA_PATTERNS):
            if "pulmonary" in desc_lower or "pe_" in desc_lower:
                return CTSeriesType.CTA_PULMONARY, CTCategory.CT_ANGIOGRAPHY
            if "aorta" in desc_lower:
                return CTSeriesType.CTA_AORTA, CTCategory.CT_ANGIOGRAPHY
            if "coronary" in desc_lower:
                return CTSeriesType.CTA_CORONARY, CTCategory.CARDIAC
            if cls._matches_any(desc_lower, cls._HEAD_PATTERNS):
                return CTSeriesType.CTA_HEAD_NECK, CTCategory.CT_ANGIOGRAPHY
            return CTSeriesType.CTA_AORTA, CTCategory.CT_ANGIOGRAPHY
        
        # Check for perfusion
        if cls._matches_any(desc_lower, cls._CTP_PATTERNS):
            if cls._matches_any(desc_lower, cls._HEAD_PATTERNS):
                return CTSeriesType.CTP_BRAIN, CTCategory.CT_PERFUSION
            return CTSeriesType.CTP_LIVER, CTCategory.CT_PERFUSION
        
        # Check for colonography
        if cls._matches_any(desc_lower, cls._COLONOGRAPHY_PATTERNS):
            return CTSeriesType.CT_COLONOGRAPHY, CTCategory.COLONOGRAPHY
        
        # Check for low-dose
        if params.is_low_dose or cls._matches_any(desc_lower, cls._LOW_DOSE_PATTERNS):
            if cls._matches_any(desc_lower, cls._CHEST_PATTERNS):
                return CTSeriesType.LOW_DOSE_CHEST, CTCategory.LOW_DOSE
            return CTSeriesType.LOW_DOSE_ABDOMEN, CTCategory.LOW_DOSE
        
        # Check for contrast phases
        if params.is_contrast_enhanced:
            if cls._matches_any(desc_lower, cls._ARTERIAL_PATTERNS):
                return CTSeriesType.ARTERIAL_PHASE, CTCategory.CONTRAST_ENHANCED
            if cls._matches_any(desc_lower, cls._VENOUS_PATTERNS):
                return CTSeriesType.PORTAL_VENOUS_PHASE, CTCategory.CONTRAST_ENHANCED
            if cls._matches_any(desc_lower, cls._DELAYED_PATTERNS):
                return CTSeriesType.DELAYED_PHASE, CTCategory.CONTRAST_ENHANCED
        
        # Non-contrast by body part
        if cls._matches_any(desc_lower, cls._HEAD_PATTERNS):
            return CTSeriesType.NON_CONTRAST_HEAD, CTCategory.NON_CONTRAST
        if cls._matches_any(desc_lower, cls._CHEST_PATTERNS):
            return CTSeriesType.NON_CONTRAST_CHEST, CTCategory.NON_CONTRAST
        if cls._matches_any(desc_lower, cls._ABDOMEN_PATTERNS):
            return CTSeriesType.NON_CONTRAST_ABDOMEN, CTCategory.NON_CONTRAST
        if cls._matches_any(desc_lower, cls._PELVIS_PATTERNS):
            return CTSeriesType.NON_CONTRAST_PELVIS, CTCategory.NON_CONTRAST
        if cls._matches_any(desc_lower, cls._SPINE_PATTERNS):
            return CTSeriesType.NON_CONTRAST_SPINE, CTCategory.NON_CONTRAST
        
        return CTSeriesType.UNKNOWN, CTCategory.UNKNOWN
    
    @staticmethod
    def _matches_any(text: str, patterns: list[str]) -> bool:
        """Check if text matches any pattern."""
        return any(p in text for p in patterns)


# =============================================================================
# CT Preprocessing
# =============================================================================

@dataclass
class CTPreprocessingConfig:
    """Configuration for CT preprocessing pipeline."""
    # HU conversion
    apply_hu_conversion: bool = True
    
    # Windowing
    apply_windowing: bool = True
    window_center: float | None = None
    window_width: float | None = None
    window_preset: str | None = None  # lung, soft_tissue, bone, brain
    
    # Resampling
    resample_to_isotropic: bool = True
    target_spacing_mm: tuple[float, float, float] | None = None
    
    # Noise reduction
    apply_denoising: bool = False
    denoising_method: str = "bilateral"  # bilateral, nlm, dl
    
    # Body mask
    extract_body_mask: bool = False


# Window presets in HU
CT_WINDOW_PRESETS = {
    "lung": {"center": -600, "width": 1500},
    "soft_tissue": {"center": 40, "width": 400},
    "bone": {"center": 400, "width": 1800},
    "brain": {"center": 40, "width": 80},
    "liver": {"center": 60, "width": 160},
    "stroke": {"center": 40, "width": 40},
    "subdural": {"center": 75, "width": 215},
    "angio": {"center": 300, "width": 600},
}


class CTPreprocessor:
    """CT-specific preprocessing pipeline."""
    
    def __init__(self, config: CTPreprocessingConfig | None = None):
        self.config = config or CTPreprocessingConfig()
    
    def preprocess_volume(
        self,
        volume: np.ndarray,
        rescale_slope: float = 1.0,
        rescale_intercept: float = 0.0,
        spacing: tuple[float, float, float] | None = None,
    ) -> np.ndarray:
        """
        Preprocess a CT volume.
        
        Args:
            volume: Raw CT volume (stored values)
            rescale_slope: DICOM RescaleSlope
            rescale_intercept: DICOM RescaleIntercept
            spacing: Current voxel spacing (z, y, x)
        
        Returns:
            Preprocessed volume in HU
        """
        result = volume.astype(np.float32)
        
        # Convert to HU
        if self.config.apply_hu_conversion:
            result = self._convert_to_hu(result, rescale_slope, rescale_intercept)
        
        # Apply windowing
        if self.config.apply_windowing:
            result = self._apply_windowing(result)
        
        return result
    
    def _convert_to_hu(
        self,
        volume: np.ndarray,
        slope: float,
        intercept: float,
    ) -> np.ndarray:
        """
        Convert stored values to Hounsfield Units.
        
        HU = pixel_value * RescaleSlope + RescaleIntercept
        """
        return volume * slope + intercept
    
    def _apply_windowing(self, volume: np.ndarray) -> np.ndarray:
        """Apply window/level to volume."""
        if self.config.window_preset:
            preset = CT_WINDOW_PRESETS.get(self.config.window_preset)
            if preset:
                center = preset["center"]
                width = preset["width"]
            else:
                return volume
        elif self.config.window_center and self.config.window_width:
            center = self.config.window_center
            width = self.config.window_width
        else:
            return volume
        
        lower = center - width / 2
        upper = center + width / 2
        
        volume = np.clip(volume, lower, upper)
        volume = (volume - lower) / (upper - lower)
        
        return volume


# =============================================================================
# CT Physics and Reconstruction
# =============================================================================

class CTPhysics:
    """
    Core CT physics formulas for reconstruction.
    
    The CT imaging process follows the Radon transform:
    - X-ray attenuation follows Beer-Lambert law
    - Projections are line integrals of attenuation coefficients
    - Reconstruction inverts the Radon transform
    
    Reference: Kak & Slaney, "Principles of Computerized Tomographic Imaging"
    """
    
    @staticmethod
    def beer_lambert(
        i0: float | np.ndarray,
        mu: float | np.ndarray,
        path_length: float,
    ) -> float | np.ndarray:
        """
        Beer-Lambert law for X-ray attenuation.
        
        I = I0 * exp(-mu * L)
        
        Args:
            i0: Incident intensity
            mu: Linear attenuation coefficient (1/cm)
            path_length: Path length through material (cm)
        
        Returns:
            Transmitted intensity
        """
        return i0 * np.exp(-mu * path_length)
    
    @staticmethod
    def hu_from_mu(
        mu: float | np.ndarray,
        mu_water: float = 0.019,
    ) -> float | np.ndarray:
        """
        Convert linear attenuation coefficient to HU.
        
        HU = 1000 * (mu - mu_water) / mu_water
        
        Args:
            mu: Linear attenuation coefficient
            mu_water: Attenuation coefficient of water (approx 0.019/mm at 70keV)
        
        Returns:
            Hounsfield Units
        """
        return 1000 * (mu - mu_water) / mu_water
    
    @staticmethod
    def mu_from_hu(
        hu: float | np.ndarray,
        mu_water: float = 0.019,
    ) -> float | np.ndarray:
        """
        Convert HU to linear attenuation coefficient.
        
        mu = mu_water * (1 + HU/1000)
        """
        return mu_water * (1 + hu / 1000)


# =============================================================================
# CT Quality Metrics
# =============================================================================

@dataclass
class CTQualityMetrics:
    """Quality metrics for CT volumes."""
    noise_hu: float | None = None  # Standard deviation in uniform region
    snr: float | None = None
    cnr: float | None = None
    
    # Dose metrics
    ctdi_vol: float | None = None
    dlp: float | None = None
    
    # Artifacts
    has_motion_artifact: bool = False
    has_beam_hardening: bool = False
    has_streak_artifact: bool = False


def estimate_ct_noise(
    volume: np.ndarray,
    roi_mask: np.ndarray | None = None,
) -> float:
    """
    Estimate CT noise as standard deviation in uniform region.
    
    Args:
        volume: CT volume in HU
        roi_mask: Optional mask for uniform region
    
    Returns:
        Noise estimate in HU
    """
    if roi_mask is not None:
        values = volume[roi_mask > 0]
    else:
        # Use central region as approximation
        z, y, x = volume.shape
        values = volume[z//3:2*z//3, y//3:2*y//3, x//3:2*x//3].flatten()
    
    return float(np.std(values))
