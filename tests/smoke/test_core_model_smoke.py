"""Smoke tests for RheniumCoreModel.

These tests verify that the core model can be imported, instantiated,
and run end-to-end on synthetic data. All tests should complete in
< 60 seconds on CPU.

IMPORTANT: This is a research and development system. It is NOT intended
for clinical use and makes NO claims of clinical performance.
"""

import pytest
import numpy as np
import torch

from rhenium.models import RheniumCoreModel, RheniumCoreModelConfig, TaskType
from rhenium.data.volume import ImageVolume, Modality
from rhenium.testing.synthetic import SyntheticDataGenerator


class TestCoreModelImport:
    """Test that core model can be imported."""

    def test_import_core_model(self):
        """Verify RheniumCoreModel can be imported."""
        from rhenium.models.core import RheniumCoreModel
        assert RheniumCoreModel is not None

    def test_import_config(self):
        """Verify RheniumCoreModelConfig can be imported."""
        from rhenium.models.core import RheniumCoreModelConfig
        assert RheniumCoreModelConfig is not None

    def test_import_task_type(self):
        """Verify TaskType enum can be imported."""
        from rhenium.models.core import TaskType
        assert TaskType.SEGMENTATION is not None


class TestCoreModelInstantiation:
    """Test that core model can be instantiated."""

    def test_default_config(self):
        """Test instantiation with default config."""
        config = RheniumCoreModelConfig()
        model = RheniumCoreModel(config)
        assert model is not None
        assert not model.is_initialized

    def test_custom_config(self):
        """Test instantiation with custom config."""
        config = RheniumCoreModelConfig(
            device="cpu",
            seed=123,
            deterministic=True,
            segmentation_features=[8, 16, 32, 64],
        )
        model = RheniumCoreModel(config)
        assert model is not None
        assert model.config.seed == 123

    def test_initialize(self):
        """Test model initialization."""
        config = RheniumCoreModelConfig(device="cpu")
        model = RheniumCoreModel(config)
        model.initialize()
        assert model.is_initialized

    def test_available_tasks(self):
        """Test available tasks after initialization."""
        config = RheniumCoreModelConfig(device="cpu")
        model = RheniumCoreModel(config)
        model.initialize()
        tasks = model.available_tasks
        assert TaskType.SEGMENTATION in tasks
        assert TaskType.FULL_PIPELINE in tasks


class TestCoreModelEndToEnd:
    """End-to-end smoke tests on synthetic data."""

    @pytest.fixture
    def synthetic_volume(self):
        """Generate a small synthetic volume."""
        generator = SyntheticDataGenerator(seed=42)
        return generator.generate_volume(
            shape=(16, 32, 32),
            modality="MRI",
            noise_level=0.1,
        )

    @pytest.fixture
    def initialized_model(self):
        """Create an initialized model."""
        config = RheniumCoreModelConfig(
            device="cpu",
            seed=42,
            segmentation_features=[8, 16, 32, 64],
            generator_features=16,
            generator_rrdb_blocks=2,
        )
        model = RheniumCoreModel(config)
        model.initialize()
        return model

    def test_segmentation_task(self, initialized_model, synthetic_volume):
        """Test segmentation task runs end-to-end."""
        result = initialized_model.run(synthetic_volume, task=TaskType.SEGMENTATION)

        assert result is not None
        assert result.task == TaskType.SEGMENTATION
        assert result.output is not None
        assert isinstance(result.output, np.ndarray)
        assert result.provenance is not None
        assert "model_name" in result.provenance

    def test_classification_task(self, initialized_model, synthetic_volume):
        """Test classification task runs end-to-end."""
        result = initialized_model.run(synthetic_volume, task=TaskType.CLASSIFICATION)

        assert result is not None
        assert result.task == TaskType.CLASSIFICATION
        assert result.output is not None
        assert "confidence" in result.metrics

    def test_detection_task(self, initialized_model, synthetic_volume):
        """Test detection task runs end-to-end."""
        result = initialized_model.run(
            synthetic_volume,
            task=TaskType.DETECTION,
            threshold=0.3,
        )

        assert result is not None
        assert result.task == TaskType.DETECTION
        assert "num_detections" in result.metrics

    def test_denoise_task(self, initialized_model, synthetic_volume):
        """Test denoising task runs end-to-end."""
        result = initialized_model.run(
            synthetic_volume,
            task=TaskType.DENOISE,
            sigma=1.0,
        )

        assert result is not None
        assert result.task == TaskType.DENOISE
        assert result.output.shape == synthetic_volume.shape
        assert result.generation_metadata is not None
        assert "disclosure" in result.generation_metadata

    def test_super_resolution_task(self, initialized_model, synthetic_volume):
        """Test super-resolution task runs end-to-end."""
        result = initialized_model.run(synthetic_volume, task=TaskType.SUPER_RESOLUTION)

        assert result is not None
        assert result.task == TaskType.SUPER_RESOLUTION
        assert result.generation_metadata is not None
        assert "disclosure" in result.generation_metadata

    def test_full_pipeline_task(self, initialized_model, synthetic_volume):
        """Test full pipeline runs end-to-end."""
        result = initialized_model.run(synthetic_volume, task=TaskType.FULL_PIPELINE)

        assert result is not None
        assert result.task == TaskType.FULL_PIPELINE
        assert "segmentation_volume_mm3" in result.metrics


class TestEvidenceDossier:
    """Test XAI evidence dossier generation."""

    @pytest.fixture
    def model_with_xai(self):
        """Create model with XAI enabled."""
        config = RheniumCoreModelConfig(
            device="cpu",
            seed=42,
            xai_enabled=True,
            segmentation_features=[8, 16],
        )
        model = RheniumCoreModel(config)
        model.initialize()
        return model

    @pytest.fixture
    def synthetic_volume(self):
        """Generate synthetic volume."""
        generator = SyntheticDataGenerator(seed=42)
        return generator.generate_volume(shape=(8, 16, 16), modality="CT")

    def test_dossier_generated(self, model_with_xai, synthetic_volume):
        """Test that evidence dossier is generated."""
        result = model_with_xai.run(synthetic_volume, task=TaskType.SEGMENTATION)

        assert result.evidence_dossier is not None
        assert "dossier_id" in result.evidence_dossier
        assert "finding" in result.evidence_dossier

    def test_dossier_has_required_keys(self, model_with_xai, synthetic_volume):
        """Test that evidence dossier contains required keys."""
        result = model_with_xai.run(
            synthetic_volume,
            task=TaskType.SEGMENTATION,
            study_uid="TEST_STUDY_001",
        )

        dossier = result.evidence_dossier
        assert "dossier_id" in dossier
        assert "finding" in dossier
        assert "pipeline" in dossier
        assert "created_at" in dossier

    def test_finding_has_structure(self, model_with_xai, synthetic_volume):
        """Test that finding has proper structure."""
        result = model_with_xai.run(synthetic_volume, task=TaskType.SEGMENTATION)

        finding = result.evidence_dossier["finding"]
        assert "id" in finding
        assert "type" in finding
        assert "description" in finding
        assert "confidence" in finding


class TestDeterminism:
    """Test reproducibility with fixed seeds."""

    def test_same_seed_same_output(self):
        """Test that same seed produces identical outputs."""
        generator = SyntheticDataGenerator(seed=42)
        volume = generator.generate_volume(shape=(8, 16, 16), modality="MRI")

        # Run 1
        config1 = RheniumCoreModelConfig(device="cpu", seed=42, deterministic=True)
        model1 = RheniumCoreModel(config1)
        model1.initialize()
        result1 = model1.run(volume, task=TaskType.SEGMENTATION)

        # Run 2
        config2 = RheniumCoreModelConfig(device="cpu", seed=42, deterministic=True)
        model2 = RheniumCoreModel(config2)
        model2.initialize()
        result2 = model2.run(volume, task=TaskType.SEGMENTATION)

        # Compare
        np.testing.assert_array_equal(result1.output, result2.output)

    def test_different_seed_different_output_structure(self):
        """Test that model works with different seeds."""
        generator = SyntheticDataGenerator(seed=42)
        volume = generator.generate_volume(shape=(8, 16, 16), modality="MRI")

        config = RheniumCoreModelConfig(device="cpu", seed=999, deterministic=True)
        model = RheniumCoreModel(config)
        model.initialize()
        result = model.run(volume, task=TaskType.SEGMENTATION)

        assert result.output is not None


class TestGenerativeDisclosure:
    """Test that generative outputs include disclosure metadata."""

    @pytest.fixture
    def model(self):
        """Create model with generative enabled."""
        config = RheniumCoreModelConfig(
            device="cpu",
            seed=42,
            generative_enabled=True,
            generator_features=8,
            generator_rrdb_blocks=1,
        )
        model = RheniumCoreModel(config)
        model.initialize()
        return model

    @pytest.fixture
    def synthetic_volume(self):
        """Generate synthetic volume."""
        generator = SyntheticDataGenerator(seed=42)
        return generator.generate_volume(shape=(8, 16, 16), modality="MRI")

    def test_sr_has_disclosure(self, model, synthetic_volume):
        """Test super-resolution output has disclosure."""
        result = model.run(synthetic_volume, task=TaskType.SUPER_RESOLUTION)

        assert result.generation_metadata is not None
        assert "disclosure" in result.generation_metadata
        assert "AI" in result.generation_metadata["disclosure"]

    def test_denoise_has_disclosure(self, model, synthetic_volume):
        """Test denoise output has disclosure."""
        result = model.run(synthetic_volume, task=TaskType.DENOISE)

        assert result.generation_metadata is not None
        assert "disclosure" in result.generation_metadata


class TestProvenance:
    """Test provenance metadata generation."""

    def test_provenance_contains_required_fields(self):
        """Test that provenance has all required fields."""
        generator = SyntheticDataGenerator(seed=42)
        volume = generator.generate_volume(shape=(8, 16, 16), modality="CT")

        config = RheniumCoreModelConfig(device="cpu", seed=42)
        model = RheniumCoreModel(config)
        model.initialize()
        result = model.run(volume, task=TaskType.SEGMENTATION)

        provenance = result.provenance
        assert "model_name" in provenance
        assert "model_version" in provenance
        assert "task" in provenance
        assert "device" in provenance
        assert "seed" in provenance
        assert "timestamp" in provenance
        assert "input_shape" in provenance
        assert "input_modality" in provenance


class TestErrorHandling:
    """Test error handling."""

    def test_run_without_initialize_raises(self):
        """Test that running without init raises error."""
        config = RheniumCoreModelConfig(device="cpu")
        model = RheniumCoreModel(config)

        generator = SyntheticDataGenerator(seed=42)
        volume = generator.generate_volume(shape=(8, 16, 16), modality="MRI")

        with pytest.raises(RuntimeError, match="not initialized"):
            model.run(volume, task=TaskType.SEGMENTATION)

    def test_invalid_task_raises(self):
        """Test that invalid task raises error."""
        config = RheniumCoreModelConfig(device="cpu")
        model = RheniumCoreModel(config)
        model.initialize()

        generator = SyntheticDataGenerator(seed=42)
        volume = generator.generate_volume(shape=(8, 16, 16), modality="MRI")

        with pytest.raises(ValueError):
            model.run(volume, task="invalid_task")
