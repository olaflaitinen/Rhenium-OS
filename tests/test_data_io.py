# Copyright (c) 2025 Skolyn LLC. All rights reserved.
# SPDX-License-Identifier: EUPL-1.1

"""Tests for data IO modules."""

import pytest
import numpy as np


class TestNIfTIIO:
    """Test NIfTI input/output."""

    def test_nifti_volume_creation(self, sample_image):
        """Test NIfTIVolume creation."""
        from rhenium.data.nifti_io import NIfTIVolume

        volume = NIfTIVolume(
            array=sample_image,
            affine=np.eye(4),
        )
        assert volume.shape == sample_image.shape
        assert volume.ndim == 3

    def test_nifti_save_load(self, sample_image, tmp_path):
        """Test NIfTI save and load."""
        from rhenium.data.nifti_io import NIfTIVolume, save_nifti, load_nifti

        volume = NIfTIVolume(array=sample_image, affine=np.eye(4))
        path = save_nifti(volume, tmp_path / "test.nii.gz")

        loaded = load_nifti(path)
        assert loaded.shape == sample_image.shape


class TestPreprocessing:
    """Test preprocessing utilities."""

    def test_normalize_zscore(self, sample_image):
        """Test z-score normalization."""
        from rhenium.data.preprocess import normalize_intensity, NormalizationMethod

        normalized = normalize_intensity(sample_image, NormalizationMethod.ZSCORE)
        assert np.isclose(np.mean(normalized), 0, atol=1e-5)
        assert np.isclose(np.std(normalized), 1, atol=1e-5)

    def test_normalize_minmax(self, sample_image):
        """Test min-max normalization."""
        from rhenium.data.preprocess import normalize_intensity, NormalizationMethod

        normalized = normalize_intensity(sample_image, NormalizationMethod.MINMAX)
        assert normalized.min() >= 0
        assert normalized.max() <= 1

    def test_crop_or_pad(self, sample_image):
        """Test crop/pad operation."""
        from rhenium.data.preprocess import crop_or_pad

        target_shape = (32, 32, 16)
        result = crop_or_pad(sample_image, target_shape)
        assert result.shape == target_shape
