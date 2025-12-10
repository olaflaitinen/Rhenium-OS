# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""Pytest fixtures for Rhenium OS tests."""

import pytest
import numpy as np
from pathlib import Path


@pytest.fixture
def sample_image() -> np.ndarray:
    """Create a sample 3D image for testing."""
    return np.random.rand(64, 64, 32).astype(np.float32)


@pytest.fixture
def sample_mask() -> np.ndarray:
    """Create a sample binary mask."""
    mask = np.zeros((64, 64, 32), dtype=np.int32)
    mask[20:40, 20:40, 10:20] = 1
    return mask


@pytest.fixture
def tmp_data_dir(tmp_path: Path) -> Path:
    """Create temporary data directory."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture
def mock_settings(monkeypatch, tmp_path: Path):
    """Configure settings for testing."""
    from rhenium.core.config import clear_settings_cache

    monkeypatch.setenv("RHENIUM_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("RHENIUM_LOGS_DIR", str(tmp_path / "logs"))
    monkeypatch.setenv("RHENIUM_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("RHENIUM_MEDGEMMA_BACKEND", "stub")

    clear_settings_cache()
    yield
    clear_settings_cache()
