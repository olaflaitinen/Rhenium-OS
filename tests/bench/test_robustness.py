"""Robustness benchmark tests.

Tests verifying model handles degraded inputs gracefully.
"""

import pytest
import numpy as np
from scipy.ndimage import gaussian_filter, zoom

from rhenium.models import RheniumCoreModel, RheniumCoreModelConfig, TaskType
from rhenium.data.volume import ImageVolume, Modality
from rhenium.testing.synthetic import SyntheticDataGenerator

from tests.bench.conftest import make_volume, make_model, TASKS


@pytest.mark.bench
class TestNoiseRobustness:
    """Test robustness to noise."""
    
    @pytest.fixture
    def model(self):
        m = make_model(device="cpu")
        yield m
        m.shutdown()
    
    @pytest.mark.parametrize("noise_sigma", [0.1, 0.2, 0.3, 0.5])
    def test_gaussian_noise_robustness(self, model, noise_sigma: float):
        """Test model handles Gaussian noise without crashing."""
        # Create clean volume
        generator = SyntheticDataGenerator(seed=42)
        volume = generator.generate_volume(shape=(16, 32, 32), modality="MRI")
        
        # Add noise
        noisy_array = volume.array + np.random.normal(0, noise_sigma, volume.shape)
        noisy_volume = ImageVolume(array=noisy_array.astype(np.float32), modality=Modality.MRI)
        
        # Should not crash
        result = model.run(noisy_volume, task=TaskType.SEGMENTATION)
        
        assert result.output is not None
        assert not np.any(np.isnan(result.output))
    
    @pytest.mark.parametrize("task", ["segmentation", "classification", "denoise"])
    def test_heavy_noise_all_tasks(self, model, task: str):
        """Test all tasks handle heavy noise."""
        array = np.random.randn(16, 32, 32).astype(np.float32)
        volume = ImageVolume(array=array, modality=Modality.MRI)
        
        task_type = TaskType(task)
        result = model.run(volume, task=task_type)
        
        assert result.output is not None


@pytest.mark.bench
class TestBlurRobustness:
    """Test robustness to blur."""
    
    @pytest.fixture
    def model(self):
        m = make_model(device="cpu")
        yield m
        m.shutdown()
    
    @pytest.mark.parametrize("blur_sigma", [1, 2, 3, 5])
    def test_blur_robustness(self, model, blur_sigma: int):
        """Test model handles blurred inputs."""
        volume = make_volume(shape=(16, 32, 32), modality="MRI")
        
        # Apply blur
        blurred_array = gaussian_filter(volume.array, sigma=blur_sigma)
        blurred_volume = ImageVolume(array=blurred_array.astype(np.float32), modality=Modality.MRI)
        
        result = model.run(blurred_volume, task=TaskType.SEGMENTATION)
        
        assert result.output is not None
        assert not np.any(np.isnan(result.output))


@pytest.mark.bench
class TestIntensityShiftRobustness:
    """Test robustness to intensity shifts."""
    
    @pytest.fixture
    def model(self):
        m = make_model(device="cpu")
        yield m
        m.shutdown()
    
    @pytest.mark.parametrize("shift", [-0.2, -0.1, 0.1, 0.2])
    def test_additive_shift(self, model, shift: float):
        """Test model handles additive intensity shifts."""
        volume = make_volume(shape=(16, 32, 32), modality="MRI")
        
        shifted_array = volume.array + shift
        shifted_volume = ImageVolume(array=shifted_array.astype(np.float32), modality=Modality.MRI)
        
        result = model.run(shifted_volume, task=TaskType.SEGMENTATION)
        
        assert result.output is not None
    
    @pytest.mark.parametrize("scale", [0.5, 0.8, 1.2, 1.5])
    def test_multiplicative_scale(self, model, scale: float):
        """Test model handles multiplicative intensity scaling."""
        volume = make_volume(shape=(16, 32, 32), modality="MRI")
        
        scaled_array = volume.array * scale
        scaled_volume = ImageVolume(array=scaled_array.astype(np.float32), modality=Modality.MRI)
        
        result = model.run(scaled_volume, task=TaskType.SEGMENTATION)
        
        assert result.output is not None


@pytest.mark.bench
class TestMissingDataRobustness:
    """Test robustness to missing data."""
    
    @pytest.fixture
    def model(self):
        m = make_model(device="cpu")
        yield m
        m.shutdown()
    
    @pytest.mark.parametrize("drop_fraction", [0.05, 0.1, 0.2])
    def test_missing_slices(self, model, drop_fraction: float):
        """Test model handles missing slices (zeroed out)."""
        volume = make_volume(shape=(32, 32, 32), modality="MRI")
        
        # Zero out random slices
        n_drop = int(volume.shape[0] * drop_fraction)
        drop_indices = np.random.choice(volume.shape[0], n_drop, replace=False)
        
        array = volume.array.copy()
        array[drop_indices] = 0
        
        modified_volume = ImageVolume(array=array, modality=Modality.MRI)
        
        result = model.run(modified_volume, task=TaskType.SEGMENTATION)
        
        assert result.output is not None
    
    def test_partial_volume(self, model):
        """Test model handles partial volumes."""
        # Create a volume with only part of data valid
        array = np.zeros((32, 32, 32), dtype=np.float32)
        array[8:24, 8:24, 8:24] = 0.5  # Only center is valid
        
        volume = ImageVolume(array=array, modality=Modality.MRI)
        
        result = model.run(volume, task=TaskType.SEGMENTATION)
        
        assert result.output is not None


@pytest.mark.bench
class TestAnisotropicSpacingRobustness:
    """Test robustness to anisotropic spacing."""
    
    @pytest.fixture
    def model(self):
        m = make_model(device="cpu")
        yield m
        m.shutdown()
    
    @pytest.mark.parametrize("spacing", [
        (1.0, 1.0, 1.0),
        (3.0, 1.0, 1.0),
        (5.0, 1.0, 1.0),
        (1.0, 0.5, 0.5),
    ])
    def test_various_spacing(self, model, spacing: tuple):
        """Test model handles various spacing values."""
        generator = SyntheticDataGenerator(seed=42)
        volume = generator.generate_volume(
            shape=(16, 32, 32),
            modality="MRI",
            spacing=spacing,
        )
        
        result = model.run(volume, task=TaskType.SEGMENTATION)
        
        assert result.output is not None
        assert result.provenance["input_spacing"] == list(spacing)


@pytest.mark.bench
class TestOrientationRobustness:
    """Test robustness to orientation changes."""
    
    @pytest.fixture
    def model(self):
        m = make_model(device="cpu")
        yield m
        m.shutdown()
    
    def test_flipped_lr(self, model):
        """Test model handles left-right flip."""
        volume = make_volume(shape=(16, 32, 32), modality="MRI")
        
        flipped_array = np.flip(volume.array, axis=2)
        flipped_volume = ImageVolume(array=flipped_array.copy(), modality=Modality.MRI)
        
        result = model.run(flipped_volume, task=TaskType.SEGMENTATION)
        
        assert result.output is not None
    
    def test_flipped_ap(self, model):
        """Test model handles anterior-posterior flip."""
        volume = make_volume(shape=(16, 32, 32), modality="MRI")
        
        flipped_array = np.flip(volume.array, axis=1)
        flipped_volume = ImageVolume(array=flipped_array.copy(), modality=Modality.MRI)
        
        result = model.run(flipped_volume, task=TaskType.SEGMENTATION)
        
        assert result.output is not None
    
    def test_flipped_si(self, model):
        """Test model handles superior-inferior flip."""
        volume = make_volume(shape=(16, 32, 32), modality="MRI")
        
        flipped_array = np.flip(volume.array, axis=0)
        flipped_volume = ImageVolume(array=flipped_array.copy(), modality=Modality.MRI)
        
        result = model.run(flipped_volume, task=TaskType.SEGMENTATION)
        
        assert result.output is not None
    
    def test_transposed(self, model):
        """Test model handles transposed volume."""
        volume = make_volume(shape=(16, 32, 32), modality="MRI")
        
        # Transpose to different orientation
        transposed_array = np.transpose(volume.array, (0, 2, 1))
        transposed_volume = ImageVolume(
            array=transposed_array.copy(),
            modality=Modality.MRI,
            spacing=(volume.spacing[0], volume.spacing[2], volume.spacing[1]),
        )
        
        result = model.run(transposed_volume, task=TaskType.SEGMENTATION)
        
        assert result.output is not None
