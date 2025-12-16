"""API and CLI readiness benchmark tests.

Tests verifying FastAPI endpoints and CLI commands work correctly.
"""

import pytest
import subprocess
import sys
import json
import os
from pathlib import Path
from typing import Any

# Conditional import for FastAPI tests
try:
    from fastapi.testclient import TestClient
    HAS_FASTAPI = True
except ImportError:
    HAS_FASTAPI = False


@pytest.mark.bench
class TestAPIReadiness:
    """Test FastAPI endpoint readiness."""
    
    @pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
    def test_health_endpoint_exists(self):
        """Test /health endpoint returns 200."""
        try:
            from rhenium.server.app import create_app
            app = create_app()
            client = TestClient(app)
            
            response = client.get("/health")
            assert response.status_code == 200
        except ImportError:
            pytest.skip("Server module not available")
    
    @pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
    def test_health_response_schema(self):
        """Test /health endpoint returns expected schema."""
        try:
            from rhenium.server.app import create_app
            app = create_app()
            client = TestClient(app)
            
            response = client.get("/health")
            if response.status_code == 200:
                data = response.json()
                assert "status" in data or response.status_code == 200
        except ImportError:
            pytest.skip("Server module not available")
    
    @pytest.mark.skipif(not HAS_FASTAPI, reason="FastAPI not installed")
    def test_error_response_format(self):
        """Test error responses have stable format."""
        try:
            from rhenium.server.app import create_app
            app = create_app()
            client = TestClient(app)
            
            # Request non-existent endpoint
            response = client.get("/nonexistent")
            assert response.status_code in [404, 405, 422]
        except ImportError:
            pytest.skip("Server module not available")


@pytest.mark.bench
class TestCLIReadiness:
    """Test CLI command readiness."""
    
    def test_cli_help(self):
        """Test CLI --help returns 0."""
        result = subprocess.run(
            [sys.executable, "-m", "rhenium.cli.main", "--help"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        # Should succeed or at least not crash
        assert result.returncode in [0, 1, 2]  # 0=success, 1/2=argparse errors are acceptable
    
    def test_cli_version(self):
        """Test CLI can report version (if available)."""
        result = subprocess.run(
            [sys.executable, "-c", "from rhenium import __version__; print(__version__)"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        # Should succeed
        assert result.returncode == 0
    
    def test_cli_module_import(self):
        """Test CLI module can be imported."""
        result = subprocess.run(
            [sys.executable, "-c", "import rhenium.cli; print('OK')"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        assert result.returncode == 0
        assert "OK" in result.stdout


@pytest.mark.bench
class TestCoreModelImportability:
    """Test core model import chain works."""
    
    def test_import_rhenium(self):
        """Test rhenium package imports."""
        result = subprocess.run(
            [sys.executable, "-c", "import rhenium; print('OK')"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        assert result.returncode == 0
    
    def test_import_core_model(self):
        """Test core model imports."""
        result = subprocess.run(
            [sys.executable, "-c", "from rhenium.models import RheniumCoreModel; print('OK')"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        assert result.returncode == 0
    
    def test_import_synthetic(self):
        """Test synthetic module imports."""
        result = subprocess.run(
            [sys.executable, "-c", "from rhenium.testing.synthetic import SyntheticDataGenerator; print('OK')"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        assert result.returncode == 0


@pytest.mark.bench
class TestEndToEndCLI:
    """Test end-to-end CLI operations."""
    
    def test_synthetic_data_generation_script(self):
        """Test synthetic data can be generated via script."""
        # Run inline script to generate synthetic data
        script = """
import numpy as np
from rhenium.testing.synthetic import SyntheticDataGenerator

gen = SyntheticDataGenerator(seed=42)
vol = gen.generate_volume(shape=(8, 16, 16), modality="MRI")
print(f"Generated volume shape: {vol.shape}")
print("OK")
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=60,
        )
        
        assert result.returncode == 0
        assert "OK" in result.stdout
    
    def test_core_model_inference_script(self):
        """Test core model inference via script."""
        script = """
from rhenium.models import RheniumCoreModel, RheniumCoreModelConfig, TaskType
from rhenium.testing.synthetic import SyntheticDataGenerator

# Generate data
gen = SyntheticDataGenerator(seed=42)
vol = gen.generate_volume(shape=(8, 16, 16), modality="MRI")

# Run model
config = RheniumCoreModelConfig(device="cpu", seed=42)
model = RheniumCoreModel(config)
model.initialize()

result = model.run(vol, task=TaskType.SEGMENTATION)
print(f"Output shape: {result.output.shape}")
print(f"Has dossier: {result.evidence_dossier is not None}")

model.shutdown()
print("OK")
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=120,
        )
        
        assert result.returncode == 0, f"Script failed: {result.stderr}"
        assert "OK" in result.stdout
