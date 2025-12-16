"""Benchmark test fixtures and configuration.

Provides shared fixtures for benchmark tests including synthetic data
generators, core model instances, and test utilities.
"""

import pytest
import numpy as np
import torch
from typing import Any

from rhenium.models import RheniumCoreModel, RheniumCoreModelConfig, TaskType
from rhenium.data.volume import ImageVolume, Modality
from rhenium.testing.synthetic import SyntheticDataGenerator


# =============================================================================
# Test Matrix Parameters
# =============================================================================

MODALITIES = ["MRI", "CT", "US", "XR"]
TASKS = ["segmentation", "classification", "detection", "reconstruction", "super_resolution", "denoise"]
SHAPES_3D = [(16, 32, 32), (32, 64, 64), (64, 128, 128)]
SHAPES_2D = [(1, 64, 64), (1, 128, 128), (1, 256, 256)]
ALL_SHAPES = SHAPES_3D + SHAPES_2D
DTYPES = ["float32", "float16"]
DEVICES = ["cpu"]  # cuda added dynamically if available

# Add cuda if available
if torch.cuda.is_available():
    DEVICES.append("cuda")


# =============================================================================
# Smoke Test Subset (50 cases for CI)
# =============================================================================

SMOKE_MODALITIES = ["MRI", "CT"]
SMOKE_TASKS = ["segmentation", "classification", "denoise"]
SMOKE_SHAPES = [(16, 32, 32), (1, 64, 64)]
SMOKE_DTYPES = ["float32"]
SMOKE_DEVICES = ["cpu"]


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture(scope="session")
def synthetic_generator():
    """Session-scoped synthetic data generator."""
    return SyntheticDataGenerator(seed=42)


@pytest.fixture(scope="session")
def core_model_config():
    """Default core model configuration for benchmarks."""
    return RheniumCoreModelConfig(
        device="cpu",
        seed=42,
        deterministic=True,
        segmentation_features=[8, 16, 32, 64],
        generator_features=16,
        generator_rrdb_blocks=2,
    )


@pytest.fixture(scope="session")
def initialized_core_model(core_model_config):
    """Session-scoped initialized core model."""
    model = RheniumCoreModel(core_model_config)
    model.initialize()
    yield model
    model.shutdown()


@pytest.fixture
def fresh_core_model():
    """Function-scoped fresh core model (for isolation)."""
    config = RheniumCoreModelConfig(
        device="cpu",
        seed=42,
        deterministic=True,
        segmentation_features=[8, 16, 32, 64],
        generator_features=16,
        generator_rrdb_blocks=2,
    )
    model = RheniumCoreModel(config)
    model.initialize()
    yield model
    model.shutdown()


@pytest.fixture
def synthetic_volume_small(synthetic_generator):
    """Small synthetic volume for fast tests."""
    return synthetic_generator.generate_volume(
        shape=(8, 16, 16),
        modality="MRI",
        noise_level=0.1,
    )


@pytest.fixture
def synthetic_volume_medium(synthetic_generator):
    """Medium synthetic volume."""
    return synthetic_generator.generate_volume(
        shape=(32, 64, 64),
        modality="MRI",
        noise_level=0.1,
    )


def make_volume(shape: tuple, modality: str, dtype: str = "float32", seed: int = 42) -> ImageVolume:
    """Create a synthetic volume with given parameters."""
    generator = SyntheticDataGenerator(seed=seed)
    volume = generator.generate_volume(shape=shape, modality=modality, noise_level=0.1)
    
    # Convert dtype if needed
    if dtype == "float16":
        volume.array = volume.array.astype(np.float16)
    
    return volume


def make_model(device: str = "cpu", seed: int = 42) -> RheniumCoreModel:
    """Create an initialized core model."""
    config = RheniumCoreModelConfig(
        device=device,
        seed=seed,
        deterministic=True,
        segmentation_features=[8, 16, 32, 64],
        generator_features=16,
        generator_rrdb_blocks=2,
    )
    model = RheniumCoreModel(config)
    model.initialize()
    return model


# =============================================================================
# Skip Conditions
# =============================================================================

def skip_if_no_cuda():
    """Skip test if CUDA is not available."""
    return pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available"
    )


def skip_if_slow():
    """Mark test as slow."""
    return pytest.mark.slow


# =============================================================================
# Pytest Configuration
# =============================================================================

def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "smoke: fast subset for CI")
    config.addinivalue_line("markers", "bench: full benchmark matrix")
    config.addinivalue_line("markers", "gpu: requires GPU")
    config.addinivalue_line("markers", "slow: long-running test")
