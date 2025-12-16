"""
Rhenium OS DICOM I/O.

This module provides DICOM file reading and writing with support for
de-identification and volume reconstruction from series.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from rhenium.core.errors import DICOMError
from rhenium.data.volume import ImageVolume, Modality, MRISequence, VolumeMetadata


class DeidentificationProfile(str, Enum):
    """De-identification profiles based on DICOM PS3.15."""

    BASIC = "basic"
    RETAIN_DATES = "retain_dates"
    RETAIN_DEVICE = "retain_device"
    FULL = "full"


# Tags to remove/modify for basic de-identification
BASIC_DEIDENT_TAGS = [
    (0x0010, 0x0010),  # PatientName
    (0x0010, 0x0020),  # PatientID
    (0x0010, 0x0030),  # PatientBirthDate
    (0x0010, 0x0040),  # PatientSex
    (0x0010, 0x1010),  # PatientAge
    (0x0008, 0x0080),  # InstitutionName
    (0x0008, 0x0081),  # InstitutionAddress
    (0x0008, 0x0090),  # ReferringPhysicianName
    (0x0008, 0x1050),  # PerformingPhysicianName
    (0x0020, 0x4000),  # ImageComments
]


@dataclass
class DICOMInstance:
    """
    Single DICOM file/instance representation.

    Attributes:
        sop_instance_uid: Unique instance identifier
        instance_number: Instance number within series
        pixel_array: Image data (loaded on demand)
        metadata: Extracted metadata
        file_path: Path to the DICOM file
    """

    sop_instance_uid: str
    instance_number: int = 0
    pixel_array: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    file_path: Path | None = None

    def load_pixels(self) -> np.ndarray:
        """Load pixel data from file."""
        if self.pixel_array is not None:
            return self.pixel_array

        if self.file_path is None:
            raise DICOMError("No file path to load pixels from")

        try:
            import pydicom

            ds = pydicom.dcmread(str(self.file_path))
            self.pixel_array = ds.pixel_array
            return self.pixel_array
        except Exception as e:
            raise DICOMError(f"Failed to load pixels: {e}") from e


@dataclass
class DICOMSeries:
    """
    Collection of DICOM instances forming a 3D volume.

    Attributes:
        series_instance_uid: Unique series identifier
        study_instance_uid: Parent study identifier
        modality: Imaging modality
        instances: List of DICOM instances
        metadata: Series-level metadata
    """

    series_instance_uid: str
    study_instance_uid: str = ""
    modality: Modality = Modality.UNKNOWN
    instances: list[DICOMInstance] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_instance(self, instance: DICOMInstance) -> None:
        """Add an instance to the series."""
        self.instances.append(instance)

    def sort_instances(self) -> None:
        """Sort instances by instance number."""
        self.instances.sort(key=lambda x: x.instance_number)

    def to_volume(self) -> ImageVolume:
        """
        Construct 3D volume from series instances.

        Returns:
            ImageVolume with reconstructed 3D data
        """
        if not self.instances:
            raise DICOMError("No instances in series")

        # Sort instances
        self.sort_instances()

        # Load all pixel arrays
        slices = []
        for instance in self.instances:
            pixels = instance.load_pixels()
            slices.append(pixels)

        # Stack into 3D
        array = np.stack(slices, axis=0)

        # Extract spacing
        spacing = self._extract_spacing()

        # Extract origin
        origin = self._extract_origin()

        # Extract orientation
        orientation = self._extract_orientation()

        # Build volume metadata
        vol_metadata = VolumeMetadata(
            study_uid=self.study_instance_uid,
            series_uid=self.series_instance_uid,
            modality=self.modality,
        )

        return ImageVolume(
            array=array,
            spacing=spacing,
            origin=origin,
            orientation=orientation,
            modality=self.modality,
            metadata=vol_metadata,
        )

    def _extract_spacing(self) -> tuple[float, float, float]:
        """Extract voxel spacing from metadata."""
        pixel_spacing = self.metadata.get("PixelSpacing", [1.0, 1.0])
        slice_thickness = self.metadata.get("SliceThickness", 1.0)

        if isinstance(pixel_spacing, (list, tuple)) and len(pixel_spacing) >= 2:
            return (float(slice_thickness), float(pixel_spacing[0]), float(pixel_spacing[1]))
        return (float(slice_thickness), 1.0, 1.0)

    def _extract_origin(self) -> tuple[float, float, float]:
        """Extract volume origin from first instance."""
        if self.instances:
            pos = self.instances[0].metadata.get("ImagePositionPatient", [0, 0, 0])
            if len(pos) >= 3:
                return (float(pos[2]), float(pos[1]), float(pos[0]))
        return (0.0, 0.0, 0.0)

    def _extract_orientation(self) -> np.ndarray:
        """Extract orientation matrix."""
        iop = self.metadata.get("ImageOrientationPatient", [1, 0, 0, 0, 1, 0])
        if len(iop) >= 6:
            row_vec = np.array([float(iop[0]), float(iop[1]), float(iop[2])])
            col_vec = np.array([float(iop[3]), float(iop[4]), float(iop[5])])
            slice_vec = np.cross(row_vec, col_vec)
            return np.column_stack([row_vec, col_vec, slice_vec])
        return np.eye(3)


@dataclass
class DICOMStudy:
    """
    Collection of series from one examination.

    Attributes:
        study_instance_uid: Unique study identifier
        series: Dictionary of series by series UID
        metadata: Study-level metadata
    """

    study_instance_uid: str
    series: dict[str, DICOMSeries] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def add_series(self, series: DICOMSeries) -> None:
        """Add a series to the study."""
        self.series[series.series_instance_uid] = series

    def get_series(self, series_uid: str) -> DICOMSeries | None:
        """Get series by UID."""
        return self.series.get(series_uid)

    def list_series(self) -> list[str]:
        """List all series UIDs."""
        return list(self.series.keys())


def load_dicom_file(path: Path) -> DICOMInstance:
    """
    Load a single DICOM file.

    Args:
        path: Path to DICOM file

    Returns:
        DICOMInstance with loaded data
    """
    try:
        import pydicom

        ds = pydicom.dcmread(str(path))

        instance = DICOMInstance(
            sop_instance_uid=str(ds.SOPInstanceUID),
            instance_number=int(getattr(ds, "InstanceNumber", 0)),
            pixel_array=ds.pixel_array if hasattr(ds, "pixel_array") else None,
            metadata={
                "Modality": getattr(ds, "Modality", ""),
                "PixelSpacing": getattr(ds, "PixelSpacing", [1.0, 1.0]),
                "SliceThickness": getattr(ds, "SliceThickness", 1.0),
                "ImagePositionPatient": getattr(ds, "ImagePositionPatient", [0, 0, 0]),
                "ImageOrientationPatient": getattr(ds, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0]),
            },
            file_path=path,
        )
        return instance
    except Exception as e:
        raise DICOMError(f"Failed to load DICOM file {path}: {e}") from e


def load_dicom_directory(directory: Path) -> DICOMStudy:
    """
    Load all DICOM files from a directory.

    Args:
        directory: Path to directory containing DICOM files

    Returns:
        DICOMStudy with all series
    """
    try:
        import pydicom
    except ImportError as e:
        raise DICOMError("pydicom is required for DICOM handling") from e

    study = DICOMStudy(study_instance_uid="")

    for file_path in directory.rglob("*"):
        if not file_path.is_file():
            continue

        try:
            ds = pydicom.dcmread(str(file_path), stop_before_pixels=True)

            # Get UIDs
            study_uid = str(ds.StudyInstanceUID)
            series_uid = str(ds.SeriesInstanceUID)
            sop_uid = str(ds.SOPInstanceUID)

            # Set study UID
            if not study.study_instance_uid:
                study.study_instance_uid = study_uid

            # Get or create series
            if series_uid not in study.series:
                modality_str = getattr(ds, "Modality", "")
                modality = Modality(modality_str) if modality_str in [m.value for m in Modality] else Modality.UNKNOWN

                series = DICOMSeries(
                    series_instance_uid=series_uid,
                    study_instance_uid=study_uid,
                    modality=modality,
                    metadata={
                        "PixelSpacing": getattr(ds, "PixelSpacing", [1.0, 1.0]),
                        "SliceThickness": getattr(ds, "SliceThickness", 1.0),
                        "ImageOrientationPatient": getattr(ds, "ImageOrientationPatient", [1, 0, 0, 0, 1, 0]),
                    },
                )
                study.add_series(series)

            # Add instance
            instance = DICOMInstance(
                sop_instance_uid=sop_uid,
                instance_number=int(getattr(ds, "InstanceNumber", 0)),
                metadata={
                    "ImagePositionPatient": getattr(ds, "ImagePositionPatient", [0, 0, 0]),
                },
                file_path=file_path,
            )
            study.series[series_uid].add_instance(instance)

        except Exception:
            # Skip non-DICOM files
            continue

    return study


def deidentify_dataset(
    dataset: Any,
    profile: DeidentificationProfile = DeidentificationProfile.BASIC,
) -> Any:
    """
    Apply de-identification to a pydicom Dataset.

    Args:
        dataset: pydicom Dataset object
        profile: De-identification profile

    Returns:
        De-identified Dataset
    """
    import pydicom

    # Remove basic identifying information
    for tag in BASIC_DEIDENT_TAGS:
        if tag in dataset:
            del dataset[tag]

    # Set de-identification method
    dataset.PatientIdentityRemoved = "YES"
    dataset.DeidentificationMethod = f"Rhenium OS {profile.value}"

    return dataset
