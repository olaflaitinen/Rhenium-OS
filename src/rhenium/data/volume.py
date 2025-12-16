"""
Rhenium OS Volume Representation.

This module provides the core ImageVolume class for representing
3D medical imaging data with geometric metadata.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


class Modality(str, Enum):
    """Supported imaging modalities."""

    MRI = "MRI"
    CT = "CT"
    US = "US"          # Ultrasound
    XRAY = "XRAY"
    PET = "PET"
    SPECT = "SPECT"
    UNKNOWN = "UNKNOWN"


class MRISequence(str, Enum):
    """MRI sequence types."""

    T1 = "T1"
    T1_CE = "T1_CE"     # Contrast-enhanced
    T2 = "T2"
    FLAIR = "FLAIR"
    DWI = "DWI"
    ADC = "ADC"
    SWI = "SWI"
    PWI = "PWI"         # Perfusion
    MRA = "MRA"
    UNKNOWN = "UNKNOWN"


class CTWindow(str, Enum):
    """CT window presets."""

    SOFT_TISSUE = "soft_tissue"
    LUNG = "lung"
    BONE = "bone"
    BRAIN = "brain"
    LIVER = "liver"
    SPINE = "spine"


# CT window preset values (center, width)
CT_WINDOW_PRESETS: dict[CTWindow, tuple[float, float]] = {
    CTWindow.SOFT_TISSUE: (40, 400),
    CTWindow.LUNG: (-600, 1500),
    CTWindow.BONE: (400, 1800),
    CTWindow.BRAIN: (40, 80),
    CTWindow.LIVER: (60, 150),
    CTWindow.SPINE: (60, 300),
}


@dataclass
class VolumeMetadata:
    """Metadata associated with an image volume."""

    study_uid: str = ""
    series_uid: str = ""
    patient_id: str = ""  # De-identified
    acquisition_date: str = ""
    modality: Modality = Modality.UNKNOWN
    sequence_type: MRISequence | None = None
    body_part: str = ""
    manufacturer: str = ""
    model_name: str = ""
    slice_thickness: float | None = None
    repetition_time: float | None = None  # MRI TR
    echo_time: float | None = None       # MRI TE
    flip_angle: float | None = None
    magnetic_field_strength: float | None = None
    kvp: float | None = None             # CT kVp
    mas: float | None = None             # CT mAs
    custom: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        result = {
            "study_uid": self.study_uid,
            "series_uid": self.series_uid,
            "modality": self.modality.value,
            "body_part": self.body_part,
        }
        if self.sequence_type:
            result["sequence_type"] = self.sequence_type.value
        return result


@dataclass
class ImageVolume:
    """
    3D medical image volume with geometric metadata.

    This is the core data structure for representing medical images
    in the Rhenium OS platform.

    Attributes:
        array: 3D or 4D numpy array (D, H, W) or (D, H, W, C)
        spacing: Voxel spacing in mm (z, y, x)
        origin: World coordinate origin (z, y, x) in mm
        orientation: 3x3 direction cosine matrix
        modality: Imaging modality
        metadata: Additional metadata
    """

    array: np.ndarray
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: np.ndarray = field(default_factory=lambda: np.eye(3))
    modality: Modality = Modality.UNKNOWN
    metadata: VolumeMetadata = field(default_factory=VolumeMetadata)

    def __post_init__(self) -> None:
        """Validate array and orientation."""
        if self.array.ndim not in (3, 4):
            raise ValueError(f"Array must be 3D or 4D, got {self.array.ndim}D")
        if self.orientation.shape != (3, 3):
            raise ValueError(f"Orientation must be 3x3, got {self.orientation.shape}")

    @property
    def shape(self) -> tuple[int, ...]:
        """Return array shape."""
        return self.array.shape

    @property
    def ndim(self) -> int:
        """Return number of dimensions."""
        return self.array.ndim

    @property
    def dtype(self) -> np.dtype:
        """Return array dtype."""
        return self.array.dtype

    @property
    def is_multichannel(self) -> bool:
        """Check if volume has multiple channels."""
        return self.array.ndim == 4

    @property
    def num_slices(self) -> int:
        """Return number of slices (depth dimension)."""
        return self.array.shape[0]

    @property
    def affine(self) -> np.ndarray:
        """
        Get 4x4 affine transformation matrix.

        Maps voxel indices to world coordinates.
        """
        affine = np.eye(4)
        # Scale by spacing
        affine[:3, :3] = self.orientation @ np.diag(self.spacing)
        # Set origin
        affine[:3, 3] = self.origin
        return affine

    @property
    def world_to_voxel(self) -> np.ndarray:
        """Get inverse affine (world to voxel)."""
        return np.linalg.inv(self.affine)

    def get_slice(self, axis: int, index: int) -> np.ndarray:
        """
        Extract 2D slice along specified axis.

        Args:
            axis: Axis to slice (0=axial, 1=coronal, 2=sagittal)
            index: Slice index

        Returns:
            2D numpy array
        """
        if axis == 0:
            return self.array[index, :, :]
        elif axis == 1:
            return self.array[:, index, :]
        elif axis == 2:
            return self.array[:, :, index]
        else:
            raise ValueError(f"Invalid axis: {axis}")

    def to_tensor(self, device: str = "cpu", add_batch: bool = True) -> "torch.Tensor":
        """
        Convert to PyTorch tensor.

        Args:
            device: Target device
            add_batch: Add batch dimension

        Returns:
            PyTorch tensor with shape (B, C, D, H, W) or (C, D, H, W)
        """
        import torch

        # Ensure float32
        data = self.array.astype(np.float32)

        # Add channel dimension if needed
        if data.ndim == 3:
            data = data[np.newaxis, ...]  # (C, D, H, W)

        # Convert to tensor
        tensor = torch.from_numpy(data)

        # Add batch dimension
        if add_batch:
            tensor = tensor.unsqueeze(0)  # (B, C, D, H, W)

        return tensor.to(device)

    def resample(
        self,
        target_spacing: tuple[float, float, float],
        order: int = 1,
    ) -> "ImageVolume":
        """
        Resample volume to target spacing.

        Args:
            target_spacing: Target voxel spacing (z, y, x) in mm
            order: Interpolation order (0=nearest, 1=linear, 3=cubic)

        Returns:
            Resampled ImageVolume
        """
        from scipy import ndimage

        # Calculate scale factors
        scale = tuple(s / t for s, t in zip(self.spacing, target_spacing))

        # Resample
        resampled = ndimage.zoom(self.array, scale, order=order)

        return ImageVolume(
            array=resampled,
            spacing=target_spacing,
            origin=self.origin,
            orientation=self.orientation.copy(),
            modality=self.modality,
            metadata=self.metadata,
        )

    def apply_window(
        self,
        window_center: float,
        window_width: float,
        output_range: tuple[float, float] = (0.0, 1.0),
    ) -> "ImageVolume":
        """
        Apply intensity windowing (typically for CT).

        Args:
            window_center: Window center value
            window_width: Window width
            output_range: Output intensity range

        Returns:
            Windowed ImageVolume
        """
        low = window_center - window_width / 2
        high = window_center + window_width / 2

        # Clip and scale
        windowed = np.clip(self.array, low, high)
        windowed = (windowed - low) / (high - low)
        windowed = windowed * (output_range[1] - output_range[0]) + output_range[0]

        return ImageVolume(
            array=windowed.astype(np.float32),
            spacing=self.spacing,
            origin=self.origin,
            orientation=self.orientation.copy(),
            modality=self.modality,
            metadata=self.metadata,
        )

    def apply_ct_preset(self, preset: CTWindow) -> "ImageVolume":
        """Apply a CT window preset."""
        center, width = CT_WINDOW_PRESETS[preset]
        return self.apply_window(center, width)

    def normalize(
        self,
        method: str = "minmax",
        clip_percentile: tuple[float, float] | None = None,
    ) -> "ImageVolume":
        """
        Normalize intensity values.

        Args:
            method: "minmax", "zscore", or "percentile"
            clip_percentile: Clip to percentile range before normalizing

        Returns:
            Normalized ImageVolume
        """
        data = self.array.astype(np.float32)

        # Optional percentile clipping
        if clip_percentile is not None:
            low = np.percentile(data, clip_percentile[0])
            high = np.percentile(data, clip_percentile[1])
            data = np.clip(data, low, high)

        if method == "minmax":
            data_min, data_max = data.min(), data.max()
            if data_max > data_min:
                data = (data - data_min) / (data_max - data_min)
        elif method == "zscore":
            mean, std = data.mean(), data.std()
            if std > 0:
                data = (data - mean) / std
        elif method == "percentile":
            p01 = np.percentile(data, 1)
            p99 = np.percentile(data, 99)
            if p99 > p01:
                data = np.clip(data, p01, p99)
                data = (data - p01) / (p99 - p01)
        else:
            raise ValueError(f"Unknown normalization method: {method}")

        return ImageVolume(
            array=data,
            spacing=self.spacing,
            origin=self.origin,
            orientation=self.orientation.copy(),
            modality=self.modality,
            metadata=self.metadata,
        )

    def save_nifti(self, path: Path | str) -> Path:
        """Save volume as NIfTI file."""
        from rhenium.data.nifti import save_nifti

        return save_nifti(self, Path(path))

    @classmethod
    def from_numpy(
        cls,
        array: np.ndarray,
        spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
        modality: Modality = Modality.UNKNOWN,
    ) -> "ImageVolume":
        """Create ImageVolume from numpy array."""
        return cls(
            array=array,
            spacing=spacing,
            modality=modality,
        )
