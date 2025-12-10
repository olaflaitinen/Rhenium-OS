# Copyright (c) 2025 Skolyn LLC. All rights reserved.
# SPDX-License-Identifier: EUPL-1.1

"""
DICOM Input/Output Module
=========================

Provides comprehensive DICOM file handling for Rhenium OS, including study
and series loading, metadata extraction, and vendor-agnostic abstractions.

Key classes:
    - DICOMStudy: Represents a complete imaging study
    - DICOMSeries: Represents an image series within a study
    - ImageVolume: 3D image array with associated metadata

Usage:
    from rhenium.data.dicom_io import load_dicom_study

    study = load_dicom_study("/path/to/dicom")
    for series in study.series:
        volume = series.get_volume()
        print(f"Series: {series.description}, Shape: {volume.array.shape}")
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import date, time
from enum import Enum
from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pydicom
from pydicom import Dataset
from pydicom.pixels import pixel_array

from rhenium.core.errors import DataIngestionError
from rhenium.core.logging import get_data_logger


logger = get_data_logger()


class Modality(str, Enum):
    """Supported imaging modalities."""

    MR = "MR"
    CT = "CT"
    CR = "CR"
    DX = "DX"
    US = "US"
    PT = "PT"
    NM = "NM"
    MG = "MG"
    XA = "XA"
    RF = "RF"
    OT = "OT"
    UNKNOWN = "UNKNOWN"


class BodyPart(str, Enum):
    """Common body parts for imaging."""

    HEAD = "HEAD"
    BRAIN = "BRAIN"
    NECK = "NECK"
    CHEST = "CHEST"
    ABDOMEN = "ABDOMEN"
    PELVIS = "PELVIS"
    SPINE = "SPINE"
    KNEE = "KNEE"
    ANKLE = "ANKLE"
    SHOULDER = "SHOULDER"
    HIP = "HIP"
    HAND = "HAND"
    FOOT = "FOOT"
    BREAST = "BREAST"
    PROSTATE = "PROSTATE"
    WHOLE_BODY = "WHOLE BODY"
    EXTREMITY = "EXTREMITY"
    UNKNOWN = "UNKNOWN"


@dataclass
class ImageVolume:
    """
    3D image volume with associated metadata.

    Attributes:
        array: The 3D numpy array containing voxel values.
        spacing: Voxel spacing in mm (x, y, z).
        origin: Volume origin in patient coordinates (x, y, z).
        orientation: Direction cosines for patient orientation.
        modality: Imaging modality.
        series_uid: DICOM Series Instance UID.
        window_center: Default window center for display.
        window_width: Default window width for display.
    """

    array: np.ndarray
    spacing: tuple[float, float, float]
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orientation: tuple[float, ...] = (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)
    modality: Modality = Modality.UNKNOWN
    series_uid: str = ""
    window_center: float | None = None
    window_width: float | None = None

    def __post_init__(self) -> None:
        """Validate volume properties."""
        if self.array.ndim not in (3, 4):
            raise DataIngestionError(
                f"ImageVolume requires 3D or 4D array, got {self.array.ndim}D",
                format_type="volume",
            )

    @property
    def shape(self) -> tuple[int, ...]:
        """Return the volume shape."""
        return self.array.shape

    @property
    def dtype(self) -> np.dtype:
        """Return the data type."""
        return self.array.dtype

    def get_slice(self, axis: int, index: int) -> np.ndarray:
        """Extract a 2D slice along the specified axis."""
        return np.take(self.array, index, axis=axis)


@dataclass
class DICOMSeries:
    """
    Represents a DICOM series within a study.

    Attributes:
        series_instance_uid: Unique series identifier.
        series_number: Series number within the study.
        series_description: Human-readable series description.
        modality: Imaging modality.
        body_part: Examined body part.
        acquisition_date: Date of acquisition.
        acquisition_time: Time of acquisition.
        instances: List of DICOM file paths in this series.
        pixel_spacing: In-plane pixel spacing (row, col) in mm.
        slice_thickness: Slice thickness in mm.
        slice_locations: Slice positions for 3D reconstruction.
        rows: Number of rows in each image.
        columns: Number of columns in each image.
        manufacturer: Equipment manufacturer.
        station_name: Imaging station name.
        sequence_name: MRI sequence name (if applicable).
        protocol_name: Acquisition protocol name.
    """

    series_instance_uid: str
    series_number: int | None = None
    series_description: str = ""
    modality: Modality = Modality.UNKNOWN
    body_part: BodyPart = BodyPart.UNKNOWN
    acquisition_date: date | None = None
    acquisition_time: time | None = None
    instances: list[Path] = field(default_factory=list)
    pixel_spacing: tuple[float, float] = (1.0, 1.0)
    slice_thickness: float = 1.0
    slice_locations: list[float] = field(default_factory=list)
    rows: int = 0
    columns: int = 0
    manufacturer: str = ""
    station_name: str = ""
    sequence_name: str = ""
    protocol_name: str = ""

    # Internal cache for loaded pixel data
    _volume_cache: ImageVolume | None = field(default=None, repr=False)

    def get_volume(self, force_reload: bool = False) -> ImageVolume:
        """
        Load and return the 3D image volume for this series.

        Args:
            force_reload: If True, reload from disk even if cached.

        Returns:
            ImageVolume containing the reconstructed 3D array.

        Raises:
            DataIngestionError: If volume reconstruction fails.
        """
        if self._volume_cache is not None and not force_reload:
            return self._volume_cache

        if not self.instances:
            raise DataIngestionError(
                "No DICOM instances in series",
                format_type="dicom",
            )

        logger.debug(
            "Loading volume from series",
            series_uid=self.series_instance_uid[:20] + "...",
            num_instances=len(self.instances),
        )

        try:
            # Sort instances by slice location or instance number
            sorted_instances = self._sort_instances()

            # Load all slices
            slices = []
            for instance_path in sorted_instances:
                ds = pydicom.dcmread(str(instance_path))
                arr = pixel_array(ds)
                slices.append(arr)

            # Stack into 3D volume
            volume_array = np.stack(slices, axis=0)

            # Compute spacing
            z_spacing = self.slice_thickness
            if len(self.slice_locations) >= 2:
                z_spacing = abs(self.slice_locations[1] - self.slice_locations[0])

            spacing = (self.pixel_spacing[0], self.pixel_spacing[1], z_spacing)

            # Get first dataset for additional metadata
            first_ds = pydicom.dcmread(str(sorted_instances[0]))
            origin = self._get_origin(first_ds)
            orientation = self._get_orientation(first_ds)
            window_center, window_width = self._get_window(first_ds)

            self._volume_cache = ImageVolume(
                array=volume_array,
                spacing=spacing,
                origin=origin,
                orientation=orientation,
                modality=self.modality,
                series_uid=self.series_instance_uid,
                window_center=window_center,
                window_width=window_width,
            )

            logger.info(
                "Volume loaded successfully",
                shape=volume_array.shape,
                spacing=spacing,
            )

            return self._volume_cache

        except Exception as e:
            raise DataIngestionError(
                f"Failed to load volume: {e}",
                format_type="dicom",
            ) from e

    def _sort_instances(self) -> list[Path]:
        """Sort instances by slice location or instance number."""
        instance_info = []

        for path in self.instances:
            try:
                ds = pydicom.dcmread(str(path), stop_before_pixels=True)
                slice_loc = getattr(ds, "SliceLocation", None)
                instance_num = getattr(ds, "InstanceNumber", 0)

                if slice_loc is not None:
                    instance_info.append((float(slice_loc), instance_num, path))
                else:
                    instance_info.append((float(instance_num), instance_num, path))
            except Exception:
                instance_info.append((0.0, 0, path))

        instance_info.sort(key=lambda x: (x[0], x[1]))
        return [info[2] for info in instance_info]

    @staticmethod
    def _get_origin(ds: Dataset) -> tuple[float, float, float]:
        """Extract image origin from DICOM dataset."""
        if hasattr(ds, "ImagePositionPatient"):
            pos = ds.ImagePositionPatient
            return (float(pos[0]), float(pos[1]), float(pos[2]))
        return (0.0, 0.0, 0.0)

    @staticmethod
    def _get_orientation(ds: Dataset) -> tuple[float, ...]:
        """Extract image orientation from DICOM dataset."""
        if hasattr(ds, "ImageOrientationPatient"):
            orient = ds.ImageOrientationPatient
            return tuple(float(x) for x in orient)
        return (1.0, 0.0, 0.0, 0.0, 1.0, 0.0)

    @staticmethod
    def _get_window(ds: Dataset) -> tuple[float | None, float | None]:
        """Extract window center and width from DICOM dataset."""
        center = getattr(ds, "WindowCenter", None)
        width = getattr(ds, "WindowWidth", None)

        if center is not None:
            center = float(center) if not isinstance(center, list) else float(center[0])
        if width is not None:
            width = float(width) if not isinstance(width, list) else float(width[0])

        return center, width


@dataclass
class DICOMStudy:
    """
    Represents a complete DICOM study.

    Attributes:
        study_instance_uid: Unique study identifier.
        study_id: Human-readable study ID.
        study_date: Date of study.
        study_time: Time of study.
        study_description: Study description.
        accession_number: Accession number (pseudonymized in logs).
        patient_id_hash: SHA-256 hash of patient ID for tracking.
        referring_physician: Referring physician name.
        institution_name: Institution name.
        series: List of series in this study.
    """

    study_instance_uid: str
    study_id: str = ""
    study_date: date | None = None
    study_time: time | None = None
    study_description: str = ""
    accession_number: str = ""
    patient_id_hash: str = ""
    referring_physician: str = ""
    institution_name: str = ""
    series: list[DICOMSeries] = field(default_factory=list)

    def __iter__(self) -> Iterator[DICOMSeries]:
        """Iterate over series in the study."""
        return iter(self.series)

    def __len__(self) -> int:
        """Return number of series."""
        return len(self.series)

    def get_series_by_modality(self, modality: Modality) -> list[DICOMSeries]:
        """Get all series of a specific modality."""
        return [s for s in self.series if s.modality == modality]

    def get_series_by_description(self, description: str) -> list[DICOMSeries]:
        """Get series matching a description pattern."""
        pattern = description.lower()
        return [s for s in self.series if pattern in s.series_description.lower()]


def load_dicom_directory(
    path: str | Path,
    recursive: bool = True,
) -> list[Path]:
    """
    Find all DICOM files in a directory.

    Args:
        path: Path to search for DICOM files.
        recursive: If True, search subdirectories recursively.

    Returns:
        List of paths to DICOM files.
    """
    path = Path(path)
    dicom_files = []

    if recursive:
        pattern = "**/*"
    else:
        pattern = "*"

    for file_path in path.glob(pattern):
        if file_path.is_file():
            # Check if file is DICOM
            try:
                with open(file_path, "rb") as f:
                    # Check for DICOM magic bytes
                    f.seek(128)
                    magic = f.read(4)
                    if magic == b"DICM":
                        dicom_files.append(file_path)
            except Exception:
                continue

    logger.debug(f"Found {len(dicom_files)} DICOM files in {path}")
    return dicom_files


def load_dicom_series(
    files: list[Path] | list[str],
) -> DICOMSeries:
    """
    Load a DICOM series from a list of files.

    Args:
        files: List of paths to DICOM files in the series.

    Returns:
        DICOMSeries object.

    Raises:
        DataIngestionError: If loading fails.
    """
    if not files:
        raise DataIngestionError("No files provided for series", format_type="dicom")

    paths = [Path(f) for f in files]

    # Read first file for metadata
    try:
        ds = pydicom.dcmread(str(paths[0]), stop_before_pixels=True)
    except Exception as e:
        raise DataIngestionError(f"Failed to read DICOM file: {e}", format_type="dicom") from e

    # Extract series metadata
    modality_str = getattr(ds, "Modality", "UNKNOWN")
    try:
        modality = Modality(modality_str)
    except ValueError:
        modality = Modality.UNKNOWN

    body_part_str = getattr(ds, "BodyPartExamined", "UNKNOWN")
    try:
        body_part = BodyPart(body_part_str.upper())
    except ValueError:
        body_part = BodyPart.UNKNOWN

    # Parse dates
    acq_date = None
    if hasattr(ds, "AcquisitionDate") and ds.AcquisitionDate:
        try:
            d = ds.AcquisitionDate
            acq_date = date(int(d[:4]), int(d[4:6]), int(d[6:8]))
        except Exception:
            pass

    acq_time = None
    if hasattr(ds, "AcquisitionTime") and ds.AcquisitionTime:
        try:
            t = ds.AcquisitionTime
            acq_time = time(int(t[:2]), int(t[2:4]), int(t[4:6]) if len(t) >= 6 else 0)
        except Exception:
            pass

    # Extract pixel spacing
    pixel_spacing = (1.0, 1.0)
    if hasattr(ds, "PixelSpacing") and ds.PixelSpacing:
        pixel_spacing = (float(ds.PixelSpacing[0]), float(ds.PixelSpacing[1]))

    # Extract slice locations
    slice_locations = []
    for path in paths:
        try:
            ds_slice = pydicom.dcmread(str(path), stop_before_pixels=True)
            if hasattr(ds_slice, "SliceLocation"):
                slice_locations.append(float(ds_slice.SliceLocation))
        except Exception:
            pass

    series = DICOMSeries(
        series_instance_uid=getattr(ds, "SeriesInstanceUID", ""),
        series_number=getattr(ds, "SeriesNumber", None),
        series_description=getattr(ds, "SeriesDescription", ""),
        modality=modality,
        body_part=body_part,
        acquisition_date=acq_date,
        acquisition_time=acq_time,
        instances=paths,
        pixel_spacing=pixel_spacing,
        slice_thickness=float(getattr(ds, "SliceThickness", 1.0)),
        slice_locations=sorted(slice_locations),
        rows=int(getattr(ds, "Rows", 0)),
        columns=int(getattr(ds, "Columns", 0)),
        manufacturer=getattr(ds, "Manufacturer", ""),
        station_name=getattr(ds, "StationName", ""),
        sequence_name=getattr(ds, "SequenceName", ""),
        protocol_name=getattr(ds, "ProtocolName", ""),
    )

    return series


def load_dicom_study(
    path: str | Path,
    recursive: bool = True,
) -> DICOMStudy:
    """
    Load a complete DICOM study from a directory.

    Args:
        path: Path to directory containing DICOM files.
        recursive: If True, search subdirectories.

    Returns:
        DICOMStudy object containing all series.

    Raises:
        DataIngestionError: If loading fails.
    """
    path = Path(path)
    logger.info("Loading DICOM study", path=str(path))

    # Find all DICOM files
    dicom_files = load_dicom_directory(path, recursive=recursive)

    if not dicom_files:
        raise DataIngestionError(
            f"No DICOM files found in {path}",
            file_path=str(path),
            format_type="dicom",
        )

    # Group files by series
    series_files: dict[str, list[Path]] = {}
    study_metadata: dict[str, Any] = {}

    for file_path in dicom_files:
        try:
            ds = pydicom.dcmread(str(file_path), stop_before_pixels=True)

            series_uid = getattr(ds, "SeriesInstanceUID", "unknown")
            if series_uid not in series_files:
                series_files[series_uid] = []
            series_files[series_uid].append(file_path)

            # Capture study-level metadata from first file
            if not study_metadata:
                study_metadata = {
                    "study_instance_uid": getattr(ds, "StudyInstanceUID", ""),
                    "study_id": getattr(ds, "StudyID", ""),
                    "study_description": getattr(ds, "StudyDescription", ""),
                    "accession_number": getattr(ds, "AccessionNumber", ""),
                    "referring_physician": getattr(ds, "ReferringPhysicianName", ""),
                    "institution_name": getattr(ds, "InstitutionName", ""),
                }

                # Create pseudonymized patient ID hash
                patient_id = getattr(ds, "PatientID", "")
                if patient_id:
                    study_metadata["patient_id_hash"] = hashlib.sha256(
                        patient_id.encode()
                    ).hexdigest()[:16]

                # Parse study date
                if hasattr(ds, "StudyDate") and ds.StudyDate:
                    try:
                        d = ds.StudyDate
                        study_metadata["study_date"] = date(int(d[:4]), int(d[4:6]), int(d[6:8]))
                    except Exception:
                        pass

                # Parse study time
                if hasattr(ds, "StudyTime") and ds.StudyTime:
                    try:
                        t = ds.StudyTime
                        study_metadata["study_time"] = time(
                            int(t[:2]), int(t[2:4]), int(t[4:6]) if len(t) >= 6 else 0
                        )
                    except Exception:
                        pass

        except Exception as e:
            logger.warning(f"Failed to read DICOM file: {e}", file=str(file_path))
            continue

    # Create series objects
    series_list = []
    for series_uid, files in series_files.items():
        try:
            series = load_dicom_series(files)
            series_list.append(series)
        except Exception as e:
            logger.warning(f"Failed to load series {series_uid}: {e}")

    # Sort series by number
    series_list.sort(key=lambda s: s.series_number or 0)

    study = DICOMStudy(
        series=series_list,
        **study_metadata,
    )

    logger.info(
        "DICOM study loaded",
        study_uid=study.study_instance_uid[:20] + "...",
        num_series=len(series_list),
    )

    return study
