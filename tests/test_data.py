"""Data module tests."""

import pytest
import numpy as np
from rhenium.data.volume import ImageVolume, Modality
from rhenium.data.preprocessing import Normalize, Resample, CropOrPad, PreprocessingPipeline
from rhenium.testing.synthetic import SyntheticDataGenerator


class TestImageVolume:
    def test_create_volume(self, sample_volume):
        vol = ImageVolume(
            array=sample_volume,
            spacing=(1.0, 1.0, 1.0),
            modality=Modality.MRI,
        )
        assert vol.shape == (32, 64, 64)
        assert vol.modality == Modality.MRI

    def test_to_tensor(self, sample_volume):
        vol = ImageVolume(array=sample_volume)
        tensor = vol.to_tensor(device="cpu")
        assert tensor.shape == (1, 1, 32, 64, 64)

    def test_normalize(self, sample_volume):
        vol = ImageVolume(array=sample_volume)
        normalized = vol.normalize(method="minmax")
        assert normalized.array.min() >= 0.0
        assert normalized.array.max() <= 1.0


class TestPreprocessing:
    def test_normalize(self, sample_volume):
        vol = ImageVolume(array=sample_volume)
        norm = Normalize(method="minmax")
        result = norm(vol)
        assert result.array.max() <= 1.0

    def test_crop_or_pad(self, sample_volume):
        vol = ImageVolume(array=sample_volume)
        cop = CropOrPad(target_size=(48, 48, 48))
        result = cop(vol)
        assert result.shape[:3] == (48, 48, 48)

    def test_pipeline(self, sample_volume):
        vol = ImageVolume(array=sample_volume)
        pipeline = PreprocessingPipeline.for_mri(target_spacing=(2.0, 2.0, 2.0))
        result = pipeline(vol)
        assert result.spacing == (2.0, 2.0, 2.0)


class TestSyntheticGenerator:
    def test_generate_volume(self):
        gen = SyntheticDataGenerator(seed=42)
        vol = gen.generate_volume(shape=(32, 64, 64), modality="MRI")
        assert vol.shape == (32, 64, 64)
        assert vol.modality == Modality.MRI

    def test_generate_mask(self):
        gen = SyntheticDataGenerator(seed=42)
        mask = gen.generate_segmentation_mask(shape=(32, 64, 64), num_classes=3)
        assert mask.shape == (32, 64, 64)
        assert mask.max() <= 2

    def test_generate_kspace(self):
        gen = SyntheticDataGenerator(seed=42)
        kspace, mask = gen.generate_kspace(image_shape=(64, 64), acceleration=4)
        assert kspace.shape == (64, 64)
        assert mask.sum() > 0
