# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Raw Acquisition Data Module
===========================

Interfaces for raw acquisition data: k-space (MRI), sinograms (CT), RF/IQ (US).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from rhenium.core.errors import DataIngestionError
from rhenium.core.logging import get_data_logger

logger = get_data_logger()


class AcquisitionType(str, Enum):
    """Types of raw acquisition data."""
    KSPACE = "kspace"
    SINOGRAM = "sinogram"
    RF_DATA = "rf_data"
    IQ_DATA = "iq_data"


class SamplingPattern(str, Enum):
    """K-space sampling patterns."""
    CARTESIAN = "cartesian"
    RADIAL = "radial"
    SPIRAL = "spiral"
    EPI = "epi"
    CUSTOM = "custom"


@dataclass
class RawAcquisitionData(ABC):
    """Abstract base class for raw acquisition data."""
    data: np.ndarray
    metadata: dict[str, Any] = field(default_factory=dict)
    source_path: Path | None = None

    @property
    @abstractmethod
    def acquisition_type(self) -> AcquisitionType:
        pass

    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape


@dataclass
class KSpaceData(RawAcquisitionData):
    """MRI k-space data container."""
    sampling_mask: np.ndarray | None = None
    sensitivity_maps: np.ndarray | None = None
    sampling_pattern: SamplingPattern = SamplingPattern.CARTESIAN
    acceleration_factor: float = 1.0
    field_strength: float = 1.5

    @property
    def acquisition_type(self) -> AcquisitionType:
        return AcquisitionType.KSPACE

    @property
    def num_coils(self) -> int:
        return self.data.shape[0] if self.data.ndim >= 3 else 1

    @property
    def is_undersampled(self) -> bool:
        return self.acceleration_factor > 1.0


@dataclass
class SinogramData(RawAcquisitionData):
    """CT sinogram data container."""
    projection_angles: np.ndarray | None = None
    detector_spacing: float = 1.0
    source_detector_distance: float = 1000.0
    beam_type: str = "fan"
    kvp: float = 120.0

    @property
    def acquisition_type(self) -> AcquisitionType:
        return AcquisitionType.SINOGRAM

    @property
    def num_projections(self) -> int:
        return self.data.shape[-1]


@dataclass
class RFData(RawAcquisitionData):
    """Ultrasound RF data container."""
    sampling_frequency: float = 40e6
    center_frequency: float = 5e6
    speed_of_sound: float = 1540.0
    transducer_type: str = "linear"

    @property
    def acquisition_type(self) -> AcquisitionType:
        return AcquisitionType.RF_DATA


def load_kspace(path: str | Path, dataset_key: str = "kspace") -> KSpaceData:
    """Load k-space data from HDF5 file."""
    import h5py
    path = Path(path)
    if not path.exists():
        raise DataIngestionError(f"File not found: {path}", format_type="kspace")

    with h5py.File(path, "r") as f:
        data = np.array(f[dataset_key])
        mask = np.array(f["mask"]) if "mask" in f else None
        smaps = np.array(f["smaps"]) if "smaps" in f else None

    return KSpaceData(data=data, sampling_mask=mask, sensitivity_maps=smaps, source_path=path)


def load_sinogram(path: str | Path, dataset_key: str = "sinogram") -> SinogramData:
    """Load sinogram data from HDF5 file."""
    import h5py
    path = Path(path)
    if not path.exists():
        raise DataIngestionError(f"File not found: {path}", format_type="sinogram")

    with h5py.File(path, "r") as f:
        data = np.array(f[dataset_key])
        angles = np.array(f["angles"]) if "angles" in f else None

    return SinogramData(data=data, projection_angles=angles, source_path=path)
