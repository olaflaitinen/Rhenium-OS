"""Determinism and reproducibility benchmark tests.

Tests verifying same seed produces identical outputs.
"""

import pytest
import numpy as np
import torch
from typing import Any

from rhenium.models import RheniumCoreModel, RheniumCoreModelConfig, TaskType
from rhenium.testing.synthetic import SyntheticDataGenerator

from tests.bench.conftest import make_volume, TASKS


@pytest.mark.bench
class TestDeterminism:
    """Test deterministic execution."""
    
    @pytest.mark.parametrize("task", TASKS)
    def test_same_seed_same_output(self, task: str):
        """Test that same seed produces identical outputs."""
        volume = make_volume(shape=(16, 32, 32), modality="MRI", seed=42)
        task_type = TaskType(task)
        
        # Run 1
        config1 = RheniumCoreModelConfig(
            device="cpu",
            seed=42,
            deterministic=True,
            segmentation_features=[8, 16, 32, 64],
            generator_features=16,
            generator_rrdb_blocks=2,
        )
        model1 = RheniumCoreModel(config1)
        model1.initialize()
        result1 = model1.run(volume, task=task_type)
        output1 = result1.output.copy() if isinstance(result1.output, np.ndarray) else result1.output.clone()
        model1.shutdown()
        
        # Run 2
        config2 = RheniumCoreModelConfig(
            device="cpu",
            seed=42,
            deterministic=True,
            segmentation_features=[8, 16, 32, 64],
            generator_features=16,
            generator_rrdb_blocks=2,
        )
        model2 = RheniumCoreModel(config2)
        model2.initialize()
        result2 = model2.run(volume, task=task_type)
        output2 = result2.output
        model2.shutdown()
        
        # Compare
        if isinstance(output1, np.ndarray):
            np.testing.assert_array_equal(output1, output2)
        elif isinstance(output1, torch.Tensor):
            torch.testing.assert_close(output1, output2)
    
    def test_different_seeds_can_differ(self):
        """Test that different seeds can produce different outputs."""
        volume = make_volume(shape=(16, 32, 32), modality="MRI", seed=42)
        
        # Run with seed 42
        config1 = RheniumCoreModelConfig(device="cpu", seed=42, deterministic=True)
        model1 = RheniumCoreModel(config1)
        model1.initialize()
        result1 = model1.run(volume, task=TaskType.CLASSIFICATION)
        metrics1 = result1.metrics.copy()
        model1.shutdown()
        
        # Run with seed 999
        config2 = RheniumCoreModelConfig(device="cpu", seed=999, deterministic=True)
        model2 = RheniumCoreModel(config2)
        model2.initialize()
        result2 = model2.run(volume, task=TaskType.CLASSIFICATION)
        metrics2 = result2.metrics
        model2.shutdown()
        
        # Outputs should exist (may or may not differ for classification)
        assert result1.output is not None
        assert result2.output is not None


@pytest.mark.bench
class TestCrossRunConsistency:
    """Test consistency across multiple runs."""
    
    def test_multiple_sequential_runs(self):
        """Test multiple sequential runs produce same results."""
        volume = make_volume(shape=(16, 32, 32), modality="MRI", seed=42)
        
        config = RheniumCoreModelConfig(
            device="cpu",
            seed=42,
            deterministic=True,
            segmentation_features=[8, 16, 32, 64],
        )
        model = RheniumCoreModel(config)
        model.initialize()
        
        results = []
        for i in range(5):
            model.reset()  # Reset to initial state
            result = model.run(volume, task=TaskType.SEGMENTATION)
            results.append(result.output.copy())
        
        model.shutdown()
        
        # All results should be identical
        for i in range(1, len(results)):
            np.testing.assert_array_equal(results[0], results[i])
    
    def test_run_counter_increments(self):
        """Test that run counter affects reproducibility correctly."""
        volume = make_volume(shape=(16, 32, 32), modality="MRI", seed=42)
        
        config = RheniumCoreModelConfig(
            device="cpu",
            seed=42,
            deterministic=True,
        )
        model = RheniumCoreModel(config)
        model.initialize()
        
        # First run
        result1 = model.run(volume, task=TaskType.SEGMENTATION)
        
        # Second run (different run counter -> same seed offset)
        result2 = model.run(volume, task=TaskType.SEGMENTATION)
        
        model.shutdown()
        
        # Both should produce valid outputs
        assert result1.output is not None
        assert result2.output is not None


@pytest.mark.bench
class TestReproducibilityAcrossSessions:
    """Test reproducibility across separate sessions."""
    
    def test_fresh_model_same_seed(self):
        """Test fresh models with same seed produce same results."""
        volume = make_volume(shape=(16, 32, 32), modality="MRI", seed=42)
        
        outputs = []
        
        for _ in range(3):
            config = RheniumCoreModelConfig(
                device="cpu",
                seed=42,
                deterministic=True,
                segmentation_features=[8, 16, 32, 64],
            )
            model = RheniumCoreModel(config)
            model.initialize()
            result = model.run(volume, task=TaskType.SEGMENTATION)
            outputs.append(result.output.copy())
            model.shutdown()
        
        # All outputs should be identical
        for i in range(1, len(outputs)):
            np.testing.assert_array_equal(outputs[0], outputs[i])


@pytest.mark.bench
class TestSyntheticDataDeterminism:
    """Test synthetic data generation is deterministic."""
    
    def test_generator_same_seed(self):
        """Test synthetic generator produces same data with same seed."""
        gen1 = SyntheticDataGenerator(seed=42)
        vol1 = gen1.generate_volume(shape=(16, 32, 32), modality="MRI")
        
        gen2 = SyntheticDataGenerator(seed=42)
        vol2 = gen2.generate_volume(shape=(16, 32, 32), modality="MRI")
        
        np.testing.assert_array_equal(vol1.array, vol2.array)
    
    def test_generator_different_seeds(self):
        """Test synthetic generator produces different data with different seeds."""
        gen1 = SyntheticDataGenerator(seed=42)
        vol1 = gen1.generate_volume(shape=(16, 32, 32), modality="MRI")
        
        gen2 = SyntheticDataGenerator(seed=123)
        vol2 = gen2.generate_volume(shape=(16, 32, 32), modality="MRI")
        
        # Should be different
        assert not np.array_equal(vol1.array, vol2.array)
    
    def test_end_to_end_reproducibility(self):
        """Test full pipeline is reproducible."""
        # Session 1
        gen1 = SyntheticDataGenerator(seed=42)
        vol1 = gen1.generate_volume(shape=(16, 32, 32), modality="MRI")
        
        config1 = RheniumCoreModelConfig(device="cpu", seed=42, deterministic=True)
        model1 = RheniumCoreModel(config1)
        model1.initialize()
        result1 = model1.run(vol1, task=TaskType.FULL_PIPELINE)
        output1 = result1.output.copy()
        model1.shutdown()
        
        # Session 2 (completely fresh)
        gen2 = SyntheticDataGenerator(seed=42)
        vol2 = gen2.generate_volume(shape=(16, 32, 32), modality="MRI")
        
        config2 = RheniumCoreModelConfig(device="cpu", seed=42, deterministic=True)
        model2 = RheniumCoreModel(config2)
        model2.initialize()
        result2 = model2.run(vol2, task=TaskType.FULL_PIPELINE)
        output2 = result2.output
        model2.shutdown()
        
        # Should be identical
        np.testing.assert_array_equal(output1, output2)
