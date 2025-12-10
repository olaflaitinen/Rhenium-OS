# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Rhenium OS MRI Modality Support
================================

Comprehensive MRI modality handling for all major MRI subtypes and sequences.

Supported MRI Categories:
- Structural: T1, T2, FLAIR, PD, STIR, SWI, T2*, 3D volumetric (MP-RAGE, SPGR)
- Diffusion: DWI, ADC, DTI, advanced models (NODDI conceptual)
- Perfusion: DSC, DCE, ASL, fMRI (BOLD)
- Angiography: TOF-MRA, PC-MRA, CE-MRA, MRV
- Quantitative: T1/T2/T2* mapping, proton density, QSM
- Spectroscopy: Single-voxel MRS, multi-voxel CSI
- Cardiac: Cine, LGE, T1/T2 mapping, phase-contrast flow
- Whole-body/Oncologic: DWIBS, mpMRI prostate, breast DCE



"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any

import numpy as np


# =============================================================================
# MRI Sequence Type Enumeration
# =============================================================================

class MRISequenceCategory(Enum):
    """Top-level MRI sequence categories."""
    STRUCTURAL = auto()
    DIFFUSION = auto()
    PERFUSION = auto()
    ANGIOGRAPHY = auto()
    QUANTITATIVE = auto()
    SPECTROSCOPY = auto()
    CARDIAC = auto()
    FUNCTIONAL = auto()
    UNKNOWN = auto()


class MRISequenceType(Enum):
    """
    Comprehensive enumeration of MRI sequence types.
    
    Classification based on acquisition parameters, sequence names,
    and clinical applications.
    """
    # Structural sequences
    T1_WEIGHTED = "t1_weighted"
    T1_WEIGHTED_GRE = "t1_weighted_gre"
    T1_WEIGHTED_SE = "t1_weighted_se"
    T1_MPRAGE = "t1_mprage"
    T1_SPGR = "t1_spgr"
    T1_VIBE = "t1_vibe"
    
    T2_WEIGHTED = "t2_weighted"
    T2_WEIGHTED_FSE = "t2_weighted_fse"
    T2_WEIGHTED_TSE = "t2_weighted_tse"
    T2_STAR = "t2_star"
    
    PROTON_DENSITY = "proton_density"
    PD_FSE = "pd_fse"
    
    FLAIR = "flair"
    FLAIR_3D = "flair_3d"
    
    STIR = "stir"
    TIRM = "tirm"
    
    SWI = "swi"
    SWAN = "swan"
    
    # Diffusion sequences
    DWI = "dwi"
    DWI_EPI = "dwi_epi"
    ADC_MAP = "adc_map"
    DTI = "dti"
    DTI_FA_MAP = "dti_fa_map"
    DTI_MD_MAP = "dti_md_map"
    DTI_TRACTOGRAPHY = "dti_tractography"
    HARDI = "hardi"
    NODDI = "noddi"
    
    # Perfusion sequences
    DSC_PERFUSION = "dsc_perfusion"
    DCE_PERFUSION = "dce_perfusion"
    ASL = "asl"
    ASL_PCASL = "asl_pcasl"
    ASL_PASL = "asl_pasl"
    BOLD_FMRI = "bold_fmri"
    
    # Angiography sequences
    TOF_MRA = "tof_mra"
    TOF_MRA_3D = "tof_mra_3d"
    PC_MRA = "pc_mra"
    CE_MRA = "ce_mra"
    MRV = "mrv"
    TIME_RESOLVED_MRA = "time_resolved_mra"
    
    # Quantitative sequences
    T1_MAP = "t1_map"
    T1_MAP_MOLLI = "t1_map_molli"
    T1_MAP_SHMOLLI = "t1_map_shmolli"
    T2_MAP = "t2_map"
    T2_STAR_MAP = "t2_star_map"
    PD_MAP = "pd_map"
    QSM = "qsm"
    
    # Spectroscopy
    MRS_SVS = "mrs_svs"
    MRSI = "mrsi"
    
    # Cardiac sequences
    CARDIAC_CINE = "cardiac_cine"
    CARDIAC_CINE_SSFP = "cardiac_cine_ssfp"
    LGE = "lge"
    CARDIAC_T1_MAP = "cardiac_t1_map"
    CARDIAC_T2_MAP = "cardiac_t2_map"
    CARDIAC_PHASE_CONTRAST = "cardiac_phase_contrast"
    CARDIAC_PERFUSION = "cardiac_perfusion"
    
    # Whole-body and oncologic
    DWIBS = "dwibs"
    MP_MRI_PROSTATE = "mp_mri_prostate"
    BREAST_DCE = "breast_dce"
    
    # Unknown
    UNKNOWN = "unknown"


# =============================================================================
# MRI Acquisition Parameters
# =============================================================================

@dataclass
class MRIAcquisitionParameters:
    """
    MRI acquisition parameters extracted from DICOM or protocol.
    
    These parameters are critical for:
    - Sequence classification
    - Reconstruction optimization
    - Physics-informed model constraints
    - Quality assessment
    """
    # Timing parameters (ms)
    repetition_time_ms: float | None = None  # TR
    echo_time_ms: float | None = None  # TE
    inversion_time_ms: float | None = None  # TI
    echo_train_length: int | None = None  # ETL
    
    # Flip angle (degrees)
    flip_angle_deg: float | None = None
    
    # Spatial parameters
    field_of_view_mm: tuple[float, float, float] | None = None
    matrix_size: tuple[int, int, int] | None = None
    voxel_size_mm: tuple[float, float, float] | None = None
    slice_thickness_mm: float | None = None
    slice_gap_mm: float | None = None
    
    # K-space and acquisition
    readout_bandwidth_hz_per_pixel: float | None = None
    parallel_imaging_factor: int | None = None
    partial_fourier_factor: float | None = None
    number_of_averages: int | None = None
    
    # Diffusion parameters
    b_values: list[float] | None = None
    diffusion_directions: int | None = None
    
    # Contrast parameters
    contrast_agent: str | None = None
    contrast_timing_s: float | None = None
    
    # Hardware
    field_strength_tesla: float | None = None
    coil_name: str | None = None
    number_of_channels: int | None = None
    
    # Vendor-specific
    manufacturer: str | None = None
    sequence_name: str | None = None
    protocol_name: str | None = None
    series_description: str | None = None
    
    @property
    def is_diffusion_weighted(self) -> bool:
        """Check if sequence has diffusion weighting."""
        if self.b_values is None:
            return False
        return any(b > 50 for b in self.b_values)
    
    @property
    def is_multi_echo(self) -> bool:
        """Check if sequence acquires multiple echoes."""
        return self.echo_train_length is not None and self.echo_train_length > 1
    
    @property
    def is_inversion_recovery(self) -> bool:
        """Check if sequence uses inversion recovery."""
        return self.inversion_time_ms is not None and self.inversion_time_ms > 0


# =============================================================================
# MRI Volume and Study Structures
# =============================================================================

@dataclass
class MRIVolumeInfo:
    """
    Information about a single MRI volume or series.
    
    Contains metadata, acquisition parameters, and derived classifications.
    """
    # Identification
    series_uid: str
    series_number: int
    series_description: str
    
    # Classification
    sequence_type: MRISequenceType = MRISequenceType.UNKNOWN
    sequence_category: MRISequenceCategory = MRISequenceCategory.UNKNOWN
    
    # Acquisition parameters
    acquisition_params: MRIAcquisitionParameters = field(
        default_factory=MRIAcquisitionParameters
    )
    
    # Geometry
    image_orientation: str = ""  # Axial, Sagittal, Coronal, Oblique
    number_of_slices: int = 0
    number_of_timepoints: int = 1
    
    # Data reference
    file_paths: list[str] = field(default_factory=list)
    volume_shape: tuple[int, ...] | None = None
    
    # Derived values
    estimated_snr: float | None = None
    quality_flags: list[str] = field(default_factory=list)


@dataclass
class MRIStudyMetadata:
    """
    Metadata for a complete MRI study containing multiple series.
    
    Aggregates information across all sequences in a study for
    multi-parametric analysis.
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
    volumes: list[MRIVolumeInfo] = field(default_factory=list)
    
    # Scanner information
    institution_name: str | None = None
    manufacturer: str | None = None
    scanner_model: str | None = None
    field_strength_tesla: float | None = None
    
    # Study classification
    body_part: str | None = None
    study_type: str | None = None  # e.g., "Brain MRI", "Knee MRI"
    
    def get_volumes_by_type(
        self, sequence_type: MRISequenceType
    ) -> list[MRIVolumeInfo]:
        """Filter volumes by sequence type."""
        return [v for v in self.volumes if v.sequence_type == sequence_type]
    
    def get_volumes_by_category(
        self, category: MRISequenceCategory
    ) -> list[MRIVolumeInfo]:
        """Filter volumes by category."""
        return [v for v in self.volumes if v.sequence_category == category]
    
    def has_multiparametric_data(self) -> bool:
        """Check if study contains multi-parametric sequences."""
        categories = {v.sequence_category for v in self.volumes}
        return len(categories) >= 2


# =============================================================================
# MRI Sequence Classification Engine
# =============================================================================

class MRISequenceClassifier:
    """
    Heuristic-based MRI sequence classifier.
    
    Classifies MRI series into sequence types based on:
    - DICOM tags (SeriesDescription, SequenceName, ProtocolName)
    - Acquisition parameters (TR, TE, TI, flip angle)
    - B-values for diffusion
    - Temporal dynamics for perfusion
    
    Note: Classification heuristics may require site-specific customization
    due to variations in naming conventions across institutions and vendors.
    """
    
    # Sequence name patterns (case-insensitive)
    _T1_PATTERNS = ["t1", "t1w", "mprage", "spgr", "bravo", "vibe", "flash"]
    _T2_PATTERNS = ["t2", "t2w", "fse_t2", "tse_t2", "frfse"]
    _FLAIR_PATTERNS = ["flair", "tirm_flair", "dark_fluid"]
    _STIR_PATTERNS = ["stir", "tirm"]
    _PD_PATTERNS = ["pd", "proton", "pdw"]
    _SWI_PATTERNS = ["swi", "swan", "susceptibility"]
    _T2_STAR_PATTERNS = ["t2*", "t2star", "gre", "medic"]
    
    _DWI_PATTERNS = ["dwi", "dti", "diffusion", "trace", "adc"]
    _DTI_PATTERNS = ["dti", "tensor", "hardi"]
    
    _PERFUSION_PATTERNS = ["dsc", "dce", "perfusion", "perf", "dynamic"]
    _ASL_PATTERNS = ["asl", "arterial_spin", "pcasl", "pasl"]
    _FMRI_PATTERNS = ["fmri", "bold", "resting", "task"]
    
    _MRA_PATTERNS = ["mra", "tof", "angio"]
    _MRV_PATTERNS = ["mrv", "venography"]
    
    _CARDIAC_PATTERNS = ["cine", "lge", "delayed", "cardiac", "heart"]
    _LGE_PATTERNS = ["lge", "delayed", "de-mri", "psir"]
    
    _MRS_PATTERNS = ["mrs", "spectro", "svs", "press", "steam"]
    _MAP_PATTERNS = ["map", "mapping", "molli", "shmolli"]
    
    @classmethod
    def classify(
        cls,
        params: MRIAcquisitionParameters,
        series_description: str = "",
    ) -> tuple[MRISequenceType, MRISequenceCategory]:
        """
        Classify MRI sequence based on parameters and description.
        
        Args:
            params: Acquisition parameters from DICOM
            series_description: Series description or protocol name
        
        Returns:
            Tuple of (sequence_type, sequence_category)
        """
        desc_lower = series_description.lower()
        
        # Check patterns in priority order
        if cls._matches_any(desc_lower, cls._MRS_PATTERNS):
            return MRISequenceType.MRS_SVS, MRISequenceCategory.SPECTROSCOPY
        
        if cls._matches_any(desc_lower, cls._MAP_PATTERNS):
            if "t1" in desc_lower:
                return MRISequenceType.T1_MAP, MRISequenceCategory.QUANTITATIVE
            if "t2" in desc_lower:
                return MRISequenceType.T2_MAP, MRISequenceCategory.QUANTITATIVE
            return MRISequenceType.T1_MAP, MRISequenceCategory.QUANTITATIVE
        
        if cls._matches_any(desc_lower, cls._LGE_PATTERNS):
            return MRISequenceType.LGE, MRISequenceCategory.CARDIAC
        
        if cls._matches_any(desc_lower, cls._CARDIAC_PATTERNS):
            return MRISequenceType.CARDIAC_CINE, MRISequenceCategory.CARDIAC
        
        if cls._matches_any(desc_lower, cls._MRA_PATTERNS):
            if "tof" in desc_lower:
                return MRISequenceType.TOF_MRA, MRISequenceCategory.ANGIOGRAPHY
            return MRISequenceType.CE_MRA, MRISequenceCategory.ANGIOGRAPHY
        
        if cls._matches_any(desc_lower, cls._MRV_PATTERNS):
            return MRISequenceType.MRV, MRISequenceCategory.ANGIOGRAPHY
        
        if cls._matches_any(desc_lower, cls._FMRI_PATTERNS):
            return MRISequenceType.BOLD_FMRI, MRISequenceCategory.FUNCTIONAL
        
        if cls._matches_any(desc_lower, cls._ASL_PATTERNS):
            return MRISequenceType.ASL, MRISequenceCategory.PERFUSION
        
        if cls._matches_any(desc_lower, cls._PERFUSION_PATTERNS):
            return MRISequenceType.DCE_PERFUSION, MRISequenceCategory.PERFUSION
        
        if params.is_diffusion_weighted or cls._matches_any(desc_lower, cls._DWI_PATTERNS):
            if cls._matches_any(desc_lower, cls._DTI_PATTERNS):
                return MRISequenceType.DTI, MRISequenceCategory.DIFFUSION
            if "adc" in desc_lower:
                return MRISequenceType.ADC_MAP, MRISequenceCategory.DIFFUSION
            return MRISequenceType.DWI, MRISequenceCategory.DIFFUSION
        
        if cls._matches_any(desc_lower, cls._SWI_PATTERNS):
            return MRISequenceType.SWI, MRISequenceCategory.STRUCTURAL
        
        if cls._matches_any(desc_lower, cls._FLAIR_PATTERNS):
            return MRISequenceType.FLAIR, MRISequenceCategory.STRUCTURAL
        
        if cls._matches_any(desc_lower, cls._STIR_PATTERNS):
            return MRISequenceType.STIR, MRISequenceCategory.STRUCTURAL
        
        if cls._matches_any(desc_lower, cls._PD_PATTERNS):
            return MRISequenceType.PROTON_DENSITY, MRISequenceCategory.STRUCTURAL
        
        if cls._matches_any(desc_lower, cls._T1_PATTERNS):
            if "mprage" in desc_lower:
                return MRISequenceType.T1_MPRAGE, MRISequenceCategory.STRUCTURAL
            return MRISequenceType.T1_WEIGHTED, MRISequenceCategory.STRUCTURAL
        
        if cls._matches_any(desc_lower, cls._T2_PATTERNS):
            return MRISequenceType.T2_WEIGHTED, MRISequenceCategory.STRUCTURAL
        
        return cls._classify_by_parameters(params)
    
    @classmethod
    def _classify_by_parameters(
        cls, params: MRIAcquisitionParameters
    ) -> tuple[MRISequenceType, MRISequenceCategory]:
        """Classify based on TR/TE/TI heuristics."""
        tr = params.repetition_time_ms
        te = params.echo_time_ms
        ti = params.inversion_time_ms
        
        if tr is None or te is None:
            return MRISequenceType.UNKNOWN, MRISequenceCategory.UNKNOWN
        
        if ti is not None and ti > 1500:
            return MRISequenceType.FLAIR, MRISequenceCategory.STRUCTURAL
        elif ti is not None and 100 < ti < 300:
            return MRISequenceType.STIR, MRISequenceCategory.STRUCTURAL
        
        if tr < 800 and te < 30:
            return MRISequenceType.T1_WEIGHTED, MRISequenceCategory.STRUCTURAL
        elif tr > 2000 and te > 80:
            return MRISequenceType.T2_WEIGHTED, MRISequenceCategory.STRUCTURAL
        elif tr > 2000 and te < 30:
            return MRISequenceType.PROTON_DENSITY, MRISequenceCategory.STRUCTURAL
        
        return MRISequenceType.UNKNOWN, MRISequenceCategory.UNKNOWN
    
    @staticmethod
    def _matches_any(text: str, patterns: list[str]) -> bool:
        """Check if text matches any pattern."""
        return any(p in text for p in patterns)


# =============================================================================
# MRI Preprocessing
# =============================================================================

@dataclass
class MRIPreprocessingConfig:
    """Configuration for MRI preprocessing pipeline."""
    normalize_intensity: bool = True
    normalization_method: str = "zscore"
    percentile_clip: tuple[float, float] | None = (0.5, 99.5)
    bias_field_correction: bool = True
    motion_correction: bool = False
    skull_stripping: bool = False
    resample_to_isotropic: bool = False
    target_spacing_mm: float | None = None


class MRIPreprocessor:
    """MRI-specific preprocessing pipeline."""
    
    def __init__(self, config: MRIPreprocessingConfig | None = None):
        self.config = config or MRIPreprocessingConfig()
    
    def preprocess_volume(
        self,
        volume: np.ndarray,
        sequence_type: MRISequenceType,
        mask: np.ndarray | None = None,
    ) -> np.ndarray:
        """Preprocess a single MRI volume."""
        result = volume.astype(np.float32)
        
        if self.config.normalize_intensity:
            result = self._normalize_intensity(result, mask)
        
        return result
    
    def _normalize_intensity(
        self, volume: np.ndarray, mask: np.ndarray | None = None
    ) -> np.ndarray:
        """Normalize intensity values using configured method."""
        if self.config.percentile_clip:
            low, high = self.config.percentile_clip
            p_low = np.percentile(volume, low)
            p_high = np.percentile(volume, high)
            volume = np.clip(volume, p_low, p_high)
        
        values = volume[mask > 0] if mask is not None else volume[volume > 0]
        
        if self.config.normalization_method == "zscore":
            mean_val, std_val = np.mean(values), np.std(values)
            if std_val > 0:
                volume = (volume - mean_val) / std_val
        elif self.config.normalization_method == "minmax":
            min_val, max_val = np.min(values), np.max(values)
            if max_val > min_val:
                volume = (volume - min_val) / (max_val - min_val)
        
        return volume


# =============================================================================
# MRI Physics Formulas
# =============================================================================

class MRIPhysics:
    """
    Core MRI physics formulas for reconstruction and quantitative mapping.
    
    Used in Physics-Informed Neural Networks (PINNs) and signal simulation.
    
    Reference: Haacke et al., "Magnetic Resonance Imaging: Physical Principles
    and Sequence Design", 2nd Edition.
    """
    
    @staticmethod
    def signal_spin_echo(
        m0: float | np.ndarray, t1: float | np.ndarray,
        t2: float | np.ndarray, tr: float, te: float
    ) -> float | np.ndarray:
        """
        Spin echo signal: S = M0 * (1 - exp(-TR/T1)) * exp(-TE/T2)
        """
        return m0 * (1 - np.exp(-tr / t1)) * np.exp(-te / t2)
    
    @staticmethod
    def signal_gradient_echo(
        m0: float | np.ndarray, t1: float | np.ndarray,
        t2_star: float | np.ndarray, tr: float, te: float,
        flip_angle_rad: float
    ) -> float | np.ndarray:
        """
        Gradient echo signal (SPGR/FLASH):
        S = M0 * sin(a) * (1 - E1) / (1 - cos(a) * E1) * E2*
        """
        e1 = np.exp(-tr / t1)
        e2_star = np.exp(-te / t2_star)
        num = np.sin(flip_angle_rad) * (1 - e1)
        den = 1 - np.cos(flip_angle_rad) * e1
        return m0 * (num / den) * e2_star
    
    @staticmethod
    def signal_inversion_recovery(
        m0: float | np.ndarray, t1: float | np.ndarray,
        ti: float, tr: float
    ) -> float | np.ndarray:
        """
        Inversion recovery signal:
        S = M0 * |1 - 2 * exp(-TI/T1) + exp(-TR/T1)|
        """
        return m0 * np.abs(1 - 2 * np.exp(-ti / t1) + np.exp(-tr / t1))
    
    @staticmethod
    def adc_from_dwi(
        s0: float | np.ndarray, s_b: float | np.ndarray, b_value: float
    ) -> float | np.ndarray:
        """
        Calculate ADC: ADC = -ln(S_b / S_0) / b
        """
        ratio = np.clip(s_b / (s0 + 1e-10), 1e-10, 1.0)
        return -np.log(ratio) / b_value
    
    @staticmethod
    def dwi_signal(
        s0: float | np.ndarray, adc: float | np.ndarray, b_value: float
    ) -> float | np.ndarray:
        """
        DWI signal: S_b = S_0 * exp(-b * ADC)
        """
        return s0 * np.exp(-b_value * adc)


# =============================================================================
# MRI Quality Metrics
# =============================================================================

@dataclass
class MRIQualityMetrics:
    """Quality metrics for MRI volumes."""
    snr: float | None = None
    cnr: float | None = None
    motion_score: float | None = None
    coverage_complete: bool = True


def calculate_snr(signal_roi: np.ndarray, noise_roi: np.ndarray) -> float:
    """Calculate Signal-to-Noise Ratio: SNR = mean(signal) / std(noise)"""
    noise_std = np.std(noise_roi)
    return np.mean(signal_roi) / noise_std if noise_std > 0 else np.inf


def calculate_cnr(
    roi1: np.ndarray, roi2: np.ndarray, noise_roi: np.ndarray
) -> float:
    """Calculate Contrast-to-Noise Ratio: CNR = |mean1 - mean2| / std(noise)"""
    contrast = np.abs(np.mean(roi1) - np.mean(roi2))
    noise_std = np.std(noise_roi)
    return contrast / noise_std if noise_std > 0 else np.inf
