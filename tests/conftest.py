"""Pytest configuration and fixtures."""

import pytest
import numpy as np
import torch


@pytest.fixture
def sample_volume():
    """Generate sample 3D volume."""
    return np.random.rand(32, 64, 64).astype(np.float32)


@pytest.fixture
def sample_mask():
    """Generate sample binary mask."""
    mask = np.zeros((32, 64, 64), dtype=np.int64)
    mask[10:20, 20:40, 20:40] = 1
    return mask


@pytest.fixture
def sample_tensor():
    """Generate sample PyTorch tensor."""
    return torch.randn(1, 1, 32, 64, 64)


@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducibility."""
    np.random.seed(42)
    torch.manual_seed(42)
