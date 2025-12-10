# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
NIfTI Input/Output Module
=========================

Provides NIfTI file handling for Rhenium OS, primarily for research workflows
and evaluation pipelines. Supports reading and writing NIfTI-1 and NIfTI-2
formats with proper handling of orientation and spacing.

Usage:
    from rhenium.data.nifti_io import load_nifti, save_nifti

    volume = load_nifti("/path/to/volume.nii.gz")
    print(f"Shape: {volume.array.shape}, Spacing: {volume.spacing}")

    save_nifti(volume, "/path/to/output.nii.gz")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from rhenium.core.errors import DataIngestionError
from rhenium.core.logging import get_data_logger


logger = get_data_logger()


@dataclass
class NIfTIVolume:
    """
    NIfTI volume with associated metadata.

    Attributes:
        array: The N-dimensional numpy array containing voxel values.
        affine: 4x4 affine transformation matrix.
        spacing: Voxel spacing in mm.
        header: NIfTI header information.
        file_path: Original file path (if loaded from disk).
    """

    array: np.ndarray
    affine: np.ndarray
    spacing: tuple[float, ...] = field(default_factory=lambda: (1.0, 1.0, 1.0))
    header: dict[str, Any] = field(default_factory=dict)
    file_path: Path | None = None

    def __post_init__(self) -> None:
        """Validate and compute derived properties."""
        if self.affine is None:
            self.affine = np.eye(4)

        # Extract spacing from affine if not provided
        if self.spacing == (1.0, 1.0, 1.0) and self.affine is not None:
            self.spacing = tuple(
                float(np.sqrt(np.sum(self.affine[:3, i] ** 2)))
                for i in range(min(3, self.array.ndim))
            )

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the volume shape."""
        return self.array.shape

    @property
    def ndim(self) -> int:
        """Return the number of dimensions."""
        return self.array.ndim

    @property
    def dtype(self) -> np.dtype:
        """Return the data type."""
        return self.array.dtype

    def get_slice(self, axis: int, index: int) -> np.ndarray:
        """
        Extract a 2D slice along the specified axis.

        Args:
            axis: Axis along which to slice (0, 1, or 2).
            index: Slice index.

        Returns:
            2D numpy array.
        """
        return np.take(self.array, index, axis=axis)

    def to_ras(self) -> "NIfTIVolume":
        """
        Reorient volume to RAS+ orientation.

        Returns:
            New NIfTIVolume in RAS+ orientation.
        """
        # Create nibabel image for reorientation
        img = nib.Nifti1Image(self.array, self.affine)

        # Get current orientation
        ornt = nib.orientations.io_orientation(self.affine)
        target_ornt = nib.orientations.axcodes2ornt(("R", "A", "S"))

        # Compute transform
        transform = nib.orientations.ornt_transform(ornt, target_ornt)
        reoriented = img.as_reoriented(transform)

        return NIfTIVolume(
            array=np.asarray(reoriented.dataobj),
            affine=reoriented.affine,
            header=dict(reoriented.header),
            file_path=self.file_path,
        )


def load_nifti(
    path: str | Path,
    dtype: np.dtype | None = None,
    reorient_to_ras: bool = False,
) -> NIfTIVolume:
    """
    Load a NIfTI file.

    Args:
        path: Path to the NIfTI file (.nii or .nii.gz).
        dtype: Optional dtype to cast the data to.
        reorient_to_ras: If True, reorient to RAS+ orientation.

    Returns:
        NIfTIVolume object.

    Raises:
        DataIngestionError: If loading fails.
    """
    path = Path(path)

    if not path.exists():
        raise DataIngestionError(
            f"NIfTI file not found: {path}",
            file_path=str(path),
            format_type="nifti",
        )

    logger.debug("Loading NIfTI file", path=str(path))

    try:
        img = nib.load(str(path))

        # Get array data
        array = np.asarray(img.dataobj)

        if dtype is not None:
            array = array.astype(dtype)

        # Extract header info
        header_info = {}
        if hasattr(img, "header"):
            hdr = img.header
            header_info = {
                "sizeof_hdr": int(hdr.get("sizeof_hdr", 0)),
                "dim_info": int(hdr.get("dim_info", 0)),
                "datatype": int(hdr.get("datatype", 0)),
                "bitpix": int(hdr.get("bitpix", 0)),
                "qform_code": int(hdr.get("qform_code", 0)),
                "sform_code": int(hdr.get("sform_code", 0)),
                "xyzt_units": int(hdr.get("xyzt_units", 0)),
            }

        volume = NIfTIVolume(
            array=array,
            affine=img.affine.copy(),
            header=header_info,
            file_path=path,
        )

        if reorient_to_ras:
            volume = volume.to_ras()

        logger.info(
            "NIfTI file loaded",
            path=str(path),
            shape=volume.shape,
            spacing=volume.spacing,
        )

        return volume

    except nib.filebasedimages.ImageFileError as e:
        raise DataIngestionError(
            f"Invalid NIfTI file: {e}",
            file_path=str(path),
            format_type="nifti",
        ) from e
    except Exception as e:
        raise DataIngestionError(
            f"Failed to load NIfTI file: {e}",
            file_path=str(path),
            format_type="nifti",
        ) from e


def save_nifti(
    volume: NIfTIVolume,
    path: str | Path,
    compress: bool = True,
) -> Path:
    """
    Save a volume as a NIfTI file.

    Args:
        volume: NIfTIVolume to save.
        path: Output path. Extension will be adjusted if needed.
        compress: If True, save as .nii.gz (gzip compressed).

    Returns:
        Path to the saved file.

    Raises:
        DataIngestionError: If saving fails.
    """
    path = Path(path)

    # Ensure correct extension
    if compress:
        if not path.suffix == ".gz":
            if path.suffix == ".nii":
                path = path.with_suffix(".nii.gz")
            elif not str(path).endswith(".nii.gz"):
                path = Path(str(path) + ".nii.gz")
    else:
        if str(path).endswith(".nii.gz"):
            path = Path(str(path)[:-3])
        elif path.suffix != ".nii":
            path = path.with_suffix(".nii")

    logger.debug("Saving NIfTI file", path=str(path))

    try:
        # Ensure parent directory exists
        path.parent.mkdir(parents=True, exist_ok=True)

        # Create image
        img = nib.Nifti1Image(volume.array, volume.affine)

        # Save
        nib.save(img, str(path))

        logger.info("NIfTI file saved", path=str(path), shape=volume.shape)

        return path

    except Exception as e:
        raise DataIngestionError(
            f"Failed to save NIfTI file: {e}",
            file_path=str(path),
            format_type="nifti",
        ) from e


def nifti_to_numpy(
    path: str | Path,
    dtype: np.dtype | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load NIfTI file and return raw array and affine.

    Convenience function for when only the array and affine are needed.

    Args:
        path: Path to NIfTI file.
        dtype: Optional dtype to cast array to.

    Returns:
        Tuple of (array, affine).
    """
    volume = load_nifti(path, dtype=dtype)
    return volume.array, volume.affine


def numpy_to_nifti(
    array: np.ndarray,
    affine: np.ndarray | None = None,
    path: str | Path | None = None,
    compress: bool = True,
) -> NIfTIVolume | Path:
    """
    Create NIfTI volume from numpy array.

    Args:
        array: The numpy array.
        affine: 4x4 affine matrix (identity if not provided).
        path: If provided, save to this path and return the path.
        compress: Whether to compress when saving.

    Returns:
        NIfTIVolume if path is None, else Path to saved file.
    """
    if affine is None:
        affine = np.eye(4)

    volume = NIfTIVolume(array=array, affine=affine)

    if path is not None:
        return save_nifti(volume, path, compress=compress)

    return volume
