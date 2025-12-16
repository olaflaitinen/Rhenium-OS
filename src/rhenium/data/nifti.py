"""
Rhenium OS NIfTI I/O.

This module provides NIfTI file reading and writing for medical imaging data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rhenium.core.errors import NIfTIError
from rhenium.data.volume import ImageVolume, Modality, VolumeMetadata


def load_nifti(path: Path | str) -> ImageVolume:
    """
    Load a NIfTI file.

    Args:
        path: Path to NIfTI file (.nii or .nii.gz)

    Returns:
        ImageVolume with loaded data
    """
    path = Path(path)

    try:
        import nibabel as nib

        img = nib.load(str(path))
        data = np.asanyarray(img.dataobj)

        # Extract affine
        affine = img.affine

        # Extract spacing from affine diagonal
        spacing = tuple(np.abs(np.diag(affine[:3, :3])))

        # Extract origin from affine
        origin = tuple(affine[:3, 3])

        # Extract orientation (normalize columns)
        orientation = affine[:3, :3].copy()
        for i in range(3):
            norm = np.linalg.norm(orientation[:, i])
            if norm > 0:
                orientation[:, i] /= norm

        # Ensure 3D (squeeze extra dimensions if needed)
        if data.ndim > 3:
            # Take first volume if 4D
            data = data[..., 0] if data.shape[-1] < data.shape[0] else data[:, :, :, 0]

        return ImageVolume(
            array=data,
            spacing=spacing,  # type: ignore
            origin=origin,    # type: ignore
            orientation=orientation,
            modality=Modality.UNKNOWN,
            metadata=VolumeMetadata(),
        )

    except ImportError as e:
        raise NIfTIError("nibabel is required for NIfTI handling") from e
    except Exception as e:
        raise NIfTIError(f"Failed to load NIfTI file {path}: {e}") from e


def save_nifti(volume: ImageVolume, path: Path | str) -> Path:
    """
    Save ImageVolume to NIfTI file.

    Args:
        volume: ImageVolume to save
        path: Output path (.nii or .nii.gz)

    Returns:
        Path to saved file
    """
    path = Path(path)

    try:
        import nibabel as nib

        # Build affine from volume geometry
        affine = volume.affine

        # Create NIfTI image
        img = nib.Nifti1Image(volume.array, affine)

        # Save
        nib.save(img, str(path))

        return path

    except ImportError as e:
        raise NIfTIError("nibabel is required for NIfTI handling") from e
    except Exception as e:
        raise NIfTIError(f"Failed to save NIfTI file {path}: {e}") from e


def dicom_to_nifti(
    dicom_dir: Path | str,
    output_path: Path | str,
    series_uid: str | None = None,
) -> Path:
    """
    Convert DICOM series to NIfTI.

    Args:
        dicom_dir: Directory containing DICOM files
        output_path: Output NIfTI path
        series_uid: Specific series UID to convert (first if None)

    Returns:
        Path to saved NIfTI file
    """
    from rhenium.data.dicom import load_dicom_directory

    dicom_dir = Path(dicom_dir)
    output_path = Path(output_path)

    study = load_dicom_directory(dicom_dir)

    if not study.series:
        raise NIfTIError("No DICOM series found in directory")

    if series_uid:
        series = study.get_series(series_uid)
        if not series:
            raise NIfTIError(f"Series {series_uid} not found")
    else:
        series = list(study.series.values())[0]

    volume = series.to_volume()
    return save_nifti(volume, output_path)
