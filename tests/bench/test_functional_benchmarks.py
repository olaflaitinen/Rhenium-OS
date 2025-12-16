"""Functional correctness benchmark tests.

Tests verifying outputs conform to expected schemas and invariants.
"""

import pytest
import numpy as np
import json
from typing import Any

from rhenium.models import RheniumCoreModel, RheniumCoreModelConfig, TaskType
from rhenium.testing.synthetic import SyntheticDataGenerator

from tests.bench.conftest import make_volume, make_model


@pytest.mark.bench
class TestSchemaValidation:
    """Test output schema validation."""
    
    @pytest.fixture
    def model(self):
        """Create model for schema tests."""
        m = make_model(device="cpu")
        yield m
        m.shutdown()
    
    @pytest.fixture
    def volume(self):
        """Create test volume."""
        return make_volume(shape=(16, 32, 32), modality="MRI")
    
    def test_output_has_required_fields(self, model, volume):
        """Test CoreModelOutput has all required fields."""
        result = model.run(volume, task=TaskType.SEGMENTATION)
        
        # Required fields
        assert hasattr(result, "task")
        assert hasattr(result, "output")
        assert hasattr(result, "evidence_dossier")
        assert hasattr(result, "generation_metadata")
        assert hasattr(result, "provenance")
        assert hasattr(result, "metrics")
        assert hasattr(result, "raw_outputs")
    
    def test_to_dict_produces_valid_json(self, model, volume):
        """Test that to_dict produces JSON-serializable output."""
        result = model.run(volume, task=TaskType.SEGMENTATION)
        
        result_dict = result.to_dict()
        
        # Should be JSON serializable
        json_str = json.dumps(result_dict)
        assert json_str is not None
        
        # Parse back
        parsed = json.loads(json_str)
        assert parsed["task"] == "segmentation"
    
    def test_evidence_dossier_schema(self, model, volume):
        """Test evidence dossier has required schema."""
        result = model.run(volume, task=TaskType.SEGMENTATION)
        
        dossier = result.evidence_dossier
        assert dossier is not None
        
        # Required top-level keys
        assert "dossier_id" in dossier
        assert "finding" in dossier
        assert "pipeline" in dossier
        assert "created_at" in dossier
        
        # Finding structure
        finding = dossier["finding"]
        assert "id" in finding
        assert "type" in finding
        assert "description" in finding
        assert "confidence" in finding
    
    def test_provenance_schema(self, model, volume):
        """Test provenance has required schema."""
        result = model.run(volume, task=TaskType.SEGMENTATION)
        
        provenance = result.provenance
        
        # Required provenance fields
        required = [
            "model_name", "model_version", "task", "device",
            "seed", "deterministic", "input_shape", "input_modality",
            "timestamp", "run_id"
        ]
        
        for key in required:
            assert key in provenance, f"Missing provenance key: {key}"


@pytest.mark.bench
class TestShapeInvariants:
    """Test shape invariants are preserved."""
    
    @pytest.fixture
    def model(self):
        m = make_model(device="cpu")
        yield m
        m.shutdown()
    
    @pytest.mark.parametrize("shape", [(8, 16, 16), (16, 32, 32), (32, 64, 64)])
    def test_segmentation_preserves_shape(self, model, shape):
        """Test segmentation output matches input shape."""
        volume = make_volume(shape=shape, modality="MRI")
        result = model.run(volume, task=TaskType.SEGMENTATION)
        
        assert result.output.shape == shape
    
    @pytest.mark.parametrize("shape", [(8, 16, 16), (16, 32, 32)])
    def test_denoise_preserves_shape(self, model, shape):
        """Test denoise output matches input shape."""
        volume = make_volume(shape=shape, modality="MRI")
        result = model.run(volume, task=TaskType.DENOISE)
        
        assert result.output.shape == shape
    
    def test_reconstruction_produces_valid_shape(self, model):
        """Test reconstruction produces valid output shape."""
        volume = make_volume(shape=(16, 32, 32), modality="MRI")
        result = model.run(volume, task=TaskType.RECONSTRUCTION)
        
        # Should produce some output
        assert result.output is not None
        assert len(result.output.shape) >= 2


@pytest.mark.bench
class TestDtypePreservation:
    """Test dtype handling is correct."""
    
    @pytest.fixture
    def model(self):
        m = make_model(device="cpu")
        yield m
        m.shutdown()
    
    def test_segmentation_returns_integer_mask(self, model):
        """Test segmentation returns integer dtype."""
        volume = make_volume(shape=(16, 32, 32), modality="MRI")
        result = model.run(volume, task=TaskType.SEGMENTATION)
        
        # Segmentation mask should be integer
        assert np.issubdtype(result.output.dtype, np.integer)
    
    def test_denoise_returns_float(self, model):
        """Test denoise returns float dtype."""
        volume = make_volume(shape=(16, 32, 32), modality="MRI")
        result = model.run(volume, task=TaskType.DENOISE)
        
        assert np.issubdtype(result.output.dtype, np.floating)
    
    def test_classification_returns_valid_class(self, model):
        """Test classification returns valid class index."""
        volume = make_volume(shape=(16, 32, 32), modality="MRI")
        result = model.run(volume, task=TaskType.CLASSIFICATION)
        
        # Should be integer or valid array
        output = result.output
        if isinstance(output, np.ndarray):
            assert len(output) > 0


@pytest.mark.bench
class TestMetricsValidity:
    """Test metrics are valid."""
    
    @pytest.fixture
    def model(self):
        m = make_model(device="cpu")
        yield m
        m.shutdown()
    
    def test_segmentation_volume_nonnegative(self, model):
        """Test segmentation volume metric is non-negative."""
        volume = make_volume(shape=(16, 32, 32), modality="MRI")
        result = model.run(volume, task=TaskType.SEGMENTATION)
        
        assert "volume_voxels" in result.metrics
        assert result.metrics["volume_voxels"] >= 0
    
    def test_classification_confidence_in_range(self, model):
        """Test classification confidence is in [0, 1]."""
        volume = make_volume(shape=(16, 32, 32), modality="MRI")
        result = model.run(volume, task=TaskType.CLASSIFICATION)
        
        assert "confidence" in result.metrics
        conf = result.metrics["confidence"]
        assert 0.0 <= conf <= 1.0
    
    def test_detection_count_nonnegative(self, model):
        """Test detection count is non-negative."""
        volume = make_volume(shape=(16, 32, 32), modality="MRI")
        result = model.run(volume, task=TaskType.DETECTION)
        
        assert "num_detections" in result.metrics
        assert result.metrics["num_detections"] >= 0
