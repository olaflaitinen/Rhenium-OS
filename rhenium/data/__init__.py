# Copyright (c) 2025 Skolyn LLC. All rights reserved.
# SPDX-License-Identifier: EUPL-1.1

"""
Rhenium OS Data Module
======================

Medical imaging data ingestion, preprocessing, and metadata management.
Supports DICOM, NIfTI, and raw acquisition data formats (k-space, sinograms).
"""

from rhenium.data.dicom_io import (
    DICOMStudy,
    DICOMSeries,
    ImageVolume,
    load_dicom_study,
    load_dicom_series,
    load_dicom_directory,
)
from rhenium.data.nifti_io import (
    load_nifti,
    save_nifti,
    NIfTIVolume,
)
from rhenium.data.raw_io import (
    KSpaceData,
    SinogramData,
    RFData,
)
from rhenium.data.preprocess import (
    normalize_intensity,
    resample_volume,
    crop_or_pad,
    PreprocessingPipeline,
)
from rhenium.data.metadata import (
    StudyMetadata,
    SeriesMetadata,
    DataLineage,
)


__all__ = [
    # DICOM
    "DICOMStudy",
    "DICOMSeries",
    "ImageVolume",
    "load_dicom_study",
    "load_dicom_series",
    "load_dicom_directory",
    # NIfTI
    "load_nifti",
    "save_nifti",
    "NIfTIVolume",
    # Raw data
    "KSpaceData",
    "SinogramData",
    "RFData",
    # Preprocessing
    "normalize_intensity",
    "resample_volume",
    "crop_or_pad",
    "PreprocessingPipeline",
    # Metadata
    "StudyMetadata",
    "SeriesMetadata",
    "DataLineage",
]
