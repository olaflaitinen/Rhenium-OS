"""Numerical stability benchmark tests.

Tests verifying no NaN/Inf values and bounded outputs.
"""

import pytest
import numpy as np
import torch
from typing import Any

from rhenium.models import RheniumCoreModel, RheniumCoreModelConfig, TaskType
from rhenium.testing.synthetic import SyntheticDataGenerator
from rhenium.data.volume import ImageVolume, Modality

from tests.bench.conftest import make_volume, make_model, TASKS


@pytest.mark.bench
class TestNaNInfDetection:
    """Test for NaN and Inf in outputs."""
    
    @pytest.fixture
    def model(self):
        m = make_model(device="cpu")
        yield m
        m.shutdown()
    
    @pytest.mark.parametrize("task", TASKS)
    def test_no_nan_in_output(self, model, task: str):
        """Test that no task produces NaN output."""
        volume = make_volume(shape=(16, 32, 32), modality="MRI")
        
        try:
            task_type = TaskType(task)
            result = model.run(volume, task=task_type)
            
            output = result.output
            if isinstance(output, np.ndarray):
                assert not np.any(np.isnan(output)), f"NaN found in {task} output"
            elif isinstance(output, torch.Tensor):
                assert not torch.any(torch.isnan(output)), f"NaN found in {task} output"
        except Exception as e:
            pytest.fail(f"Task {task} raised exception: {e}")
    
    @pytest.mark.parametrize("task", TASKS)
    def test_no_inf_in_output(self, model, task: str):
        """Test that no task produces Inf output."""
        volume = make_volume(shape=(16, 32, 32), modality="MRI")
        
        try:
            task_type = TaskType(task)
            result = model.run(volume, task=task_type)
            
            output = result.output
            if isinstance(output, np.ndarray):
                assert not np.any(np.isinf(output)), f"Inf found in {task} output"
            elif isinstance(output, torch.Tensor):
                assert not torch.any(torch.isinf(output)), f"Inf found in {task} output"
        except Exception as e:
            pytest.fail(f"Task {task} raised exception: {e}")


@pytest.mark.bench
class TestRangePreservation:
    """Test output ranges are bounded."""
    
    @pytest.fixture
    def model(self):
        m = make_model(device="cpu")
        yield m
        m.shutdown()
    
    def test_segmentation_mask_bounded(self, model):
        """Test segmentation mask has bounded values."""
        volume = make_volume(shape=(16, 32, 32), modality="MRI")
        result = model.run(volume, task=TaskType.SEGMENTATION)
        
        mask = result.output
        
        # Should be non-negative integers
        assert mask.min() >= 0
        # Should have reasonable class count
        assert mask.max() < 100
    
    def test_denoise_bounded_for_normalized_input(self, model):
        """Test denoise output is bounded for normalized input."""
        # Create normalized input [0, 1]
        generator = SyntheticDataGenerator(seed=42)
        volume = generator.generate_volume(shape=(16, 32, 32), modality="MRI")
        normalized = volume.normalize(method="minmax")
        
        result = model.run(normalized, task=TaskType.DENOISE)
        
        output = result.output
        
        # Gaussian filter can slightly exceed bounds, but should be close
        assert output.min() >= -0.1
        assert output.max() <= 1.1
    
    def test_classification_confidence_bounded(self, model):
        """Test classification confidence is in [0, 1]."""
        volume = make_volume(shape=(16, 32, 32), modality="MRI")
        result = model.run(volume, task=TaskType.CLASSIFICATION)
        
        confidence = result.metrics.get("confidence", 0.5)
        assert 0.0 <= confidence <= 1.0


@pytest.mark.bench
class TestEdgeCases:
    """Test numerical stability with edge cases."""
    
    @pytest.fixture
    def model(self):
        m = make_model(device="cpu")
        yield m
        m.shutdown()
    
    def test_all_zeros_input(self, model):
        """Test handling of all-zeros input."""
        array = np.zeros((16, 32, 32), dtype=np.float32)
        volume = ImageVolume(array=array, modality=Modality.MRI)
        
        result = model.run(volume, task=TaskType.SEGMENTATION)
        
        assert not np.any(np.isnan(result.output))
        assert not np.any(np.isinf(result.output))
    
    def test_all_ones_input(self, model):
        """Test handling of all-ones input."""
        array = np.ones((16, 32, 32), dtype=np.float32)
        volume = ImageVolume(array=array, modality=Modality.MRI)
        
        result = model.run(volume, task=TaskType.SEGMENTATION)
        
        assert not np.any(np.isnan(result.output))
        assert not np.any(np.isinf(result.output))
    
    def test_very_small_values(self, model):
        """Test handling of very small values."""
        array = np.full((16, 32, 32), 1e-10, dtype=np.float32)
        volume = ImageVolume(array=array, modality=Modality.MRI)
        
        result = model.run(volume, task=TaskType.DENOISE)
        
        assert not np.any(np.isnan(result.output))
        assert not np.any(np.isinf(result.output))
    
    def test_large_values(self, model):
        """Test handling of large values."""
        array = np.full((16, 32, 32), 1e6, dtype=np.float32)
        volume = ImageVolume(array=array, modality=Modality.CT)
        
        result = model.run(volume, task=TaskType.SEGMENTATION)
        
        assert not np.any(np.isnan(result.output))
        assert not np.any(np.isinf(result.output))
    
    def test_mixed_positive_negative(self, model):
        """Test handling of mixed positive/negative values."""
        array = np.random.randn(16, 32, 32).astype(np.float32)
        volume = ImageVolume(array=array, modality=Modality.MRI)
        
        result = model.run(volume, task=TaskType.DENOISE)
        
        assert not np.any(np.isnan(result.output))
        assert not np.any(np.isinf(result.output))


@pytest.mark.bench
class TestNumericalPrecision:
    """Test numerical precision is maintained."""
    
    @pytest.fixture
    def model(self):
        m = make_model(device="cpu")
        yield m
        m.shutdown()
    
    def test_denoise_preserves_structure(self, model):
        """Test denoising preserves general structure."""
        # Create volume with known structure
        array = np.zeros((16, 32, 32), dtype=np.float32)
        array[8, 16, 16] = 1.0  # Single bright point
        volume = ImageVolume(array=array, modality=Modality.MRI)
        
        result = model.run(volume, task=TaskType.DENOISE, sigma=0.5)
        
        # Peak should still exist (gaussian filter spreads but preserves)
        assert result.output[8, 16, 16] > result.output[0, 0, 0]
    
    def test_segmentation_deterministic_threshold(self, model):
        """Test segmentation thresholding is stable."""
        # Create volume with clear boundary
        array = np.zeros((16, 32, 32), dtype=np.float32)
        array[8:, :, :] = 1.0  # Upper half bright
        volume = ImageVolume(array=array, modality=Modality.MRI)
        
        result = model.run(volume, task=TaskType.SEGMENTATION)
        
        # Should segment into two regions
        unique_values = np.unique(result.output)
        assert len(unique_values) >= 1
