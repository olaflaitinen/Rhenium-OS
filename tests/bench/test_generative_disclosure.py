"""Generative disclosure compliance benchmark tests.

Tests verifying all AI-generated content includes disclosure metadata.
"""

import pytest
import numpy as np
from typing import Any

from rhenium.models import RheniumCoreModel, RheniumCoreModelConfig, TaskType
from rhenium.testing.synthetic import SyntheticDataGenerator

from tests.bench.conftest import make_volume, make_model


@pytest.mark.bench
class TestGenerativeDisclosure:
    """Test generative outputs include disclosure."""
    
    @pytest.fixture
    def model(self):
        m = make_model(device="cpu")
        yield m
        m.shutdown()
    
    @pytest.fixture
    def volume(self):
        return make_volume(shape=(16, 32, 32), modality="MRI")
    
    def test_super_resolution_has_disclosure(self, model, volume):
        """Test super-resolution output has disclosure metadata."""
        result = model.run(volume, task=TaskType.SUPER_RESOLUTION)
        
        assert result.generation_metadata is not None
        assert "disclosure" in result.generation_metadata
    
    def test_denoise_has_disclosure(self, model, volume):
        """Test denoise output has disclosure metadata."""
        result = model.run(volume, task=TaskType.DENOISE)
        
        assert result.generation_metadata is not None
        assert "disclosure" in result.generation_metadata
    
    def test_disclosure_contains_ai_text(self, model, volume):
        """Test disclosure text mentions AI or generated."""
        result = model.run(volume, task=TaskType.SUPER_RESOLUTION)
        
        disclosure = result.generation_metadata["disclosure"]
        disclosure_lower = disclosure.lower()
        
        assert "ai" in disclosure_lower or "generated" in disclosure_lower
    
    def test_disclosure_contains_research_text(self, model, volume):
        """Test disclosure mentions research use."""
        result = model.run(volume, task=TaskType.SUPER_RESOLUTION)
        
        disclosure = result.generation_metadata["disclosure"]
        disclosure_lower = disclosure.lower()
        
        assert "research" in disclosure_lower


@pytest.mark.bench
class TestGenerationMetadataStructure:
    """Test generation metadata structure."""
    
    @pytest.fixture
    def model(self):
        m = make_model(device="cpu")
        yield m
        m.shutdown()
    
    @pytest.fixture
    def volume(self):
        return make_volume(shape=(16, 32, 32), modality="MRI")
    
    def test_metadata_has_required_fields(self, model, volume):
        """Test generation metadata has all required fields."""
        result = model.run(volume, task=TaskType.SUPER_RESOLUTION)
        
        metadata = result.generation_metadata
        
        required_fields = ["generated_at", "generator", "disclosure"]
        for field in required_fields:
            assert field in metadata, f"Missing field: {field}"
    
    def test_metadata_has_generator_info(self, model, volume):
        """Test metadata includes generator name and version."""
        result = model.run(volume, task=TaskType.SUPER_RESOLUTION)
        
        metadata = result.generation_metadata
        
        assert "generator" in metadata
        # Generator should include version info
        generator = metadata["generator"]
        assert ":" in generator or "1.0" in generator
    
    def test_metadata_has_timestamp(self, model, volume):
        """Test metadata includes timestamp."""
        result = model.run(volume, task=TaskType.SUPER_RESOLUTION)
        
        metadata = result.generation_metadata
        
        assert "generated_at" in metadata
        # Should be ISO format timestamp
        timestamp = metadata["generated_at"]
        assert "T" in timestamp or "-" in timestamp
    
    def test_metadata_has_input_hash(self, model, volume):
        """Test metadata includes input hash for traceability."""
        result = model.run(volume, task=TaskType.SUPER_RESOLUTION)
        
        metadata = result.generation_metadata
        
        assert "input_hash" in metadata
        # Hash should be non-empty
        assert len(metadata["input_hash"]) > 0


@pytest.mark.bench
class TestNonGenerativeNoDisclosure:
    """Test non-generative tasks don't have (unnecessary) disclosure."""
    
    @pytest.fixture
    def model(self):
        m = make_model(device="cpu")
        yield m
        m.shutdown()
    
    @pytest.fixture
    def volume(self):
        return make_volume(shape=(16, 32, 32), modality="MRI")
    
    def test_segmentation_no_generation_metadata(self, model, volume):
        """Test segmentation doesn't produce generation metadata."""
        result = model.run(volume, task=TaskType.SEGMENTATION)
        
        # Segmentation is not generative, should not have generation metadata
        # (or it should be None)
        assert result.generation_metadata is None
    
    def test_classification_no_generation_metadata(self, model, volume):
        """Test classification doesn't produce generation metadata."""
        result = model.run(volume, task=TaskType.CLASSIFICATION)
        
        assert result.generation_metadata is None
    
    def test_detection_no_generation_metadata(self, model, volume):
        """Test detection doesn't produce generation metadata."""
        result = model.run(volume, task=TaskType.DETECTION)
        
        assert result.generation_metadata is None


@pytest.mark.bench
class TestDisclosureConsistency:
    """Test disclosure is consistent across runs."""
    
    def test_same_disclosure_multiple_runs(self):
        """Test disclosure text is consistent."""
        volume = make_volume(shape=(16, 32, 32), modality="MRI")
        
        disclosures = []
        
        for _ in range(3):
            config = RheniumCoreModelConfig(device="cpu", seed=42)
            model = RheniumCoreModel(config)
            model.initialize()
            
            result = model.run(volume, task=TaskType.SUPER_RESOLUTION)
            disclosures.append(result.generation_metadata["disclosure"])
            
            model.shutdown()
        
        # All disclosures should be identical
        assert all(d == disclosures[0] for d in disclosures)
    
    def test_disclosure_across_tasks(self):
        """Test disclosure structure is consistent across generative tasks."""
        volume = make_volume(shape=(16, 32, 32), modality="MRI")
        
        config = RheniumCoreModelConfig(device="cpu", seed=42)
        model = RheniumCoreModel(config)
        model.initialize()
        
        try:
            sr_result = model.run(volume, task=TaskType.SUPER_RESOLUTION)
            denoise_result = model.run(volume, task=TaskType.DENOISE)
            
            # Both should have disclosure
            assert sr_result.generation_metadata is not None
            assert denoise_result.generation_metadata is not None
            
            # Both should have same structure
            sr_keys = set(sr_result.generation_metadata.keys())
            denoise_keys = set(denoise_result.generation_metadata.keys())
            
            assert sr_keys == denoise_keys
        finally:
            model.shutdown()
