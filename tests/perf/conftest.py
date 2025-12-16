"""Performance test fixtures and configuration."""

import os
import pytest
from pathlib import Path

# Import harness
try:
    from rhenium.bench import PerfHarness, PerfConfig
    HAS_HARNESS = True
except ImportError:
    HAS_HARNESS = False
    PerfHarness = None
    PerfConfig = None

# Test matrix parameters
MODALITIES = ["MRI", "CT", "US", "XR"]
TASKS_E2E = ["full_pipeline"]
TASKS_PERCEPTION = ["segmentation", "classification", "detection"]
TASKS_RECON = ["mri_zerofilled", "ct_fbp"]
TASKS_GAN = ["srgan", "pix2pix", "cyclegan"]
TASKS_XAI = ["saliency", "dossier_export"]

# Sizes: (depth, height, width)
SIZES_TINY = (8, 16, 16)
SIZES_SMALL = (16, 32, 32)
SIZES_MEDIUM = (32, 64, 64)
ALL_SIZES = [SIZES_TINY, SIZES_SMALL, SIZES_MEDIUM]

# Smoke subset
SMOKE_MODALITIES = ["MRI", "CT"]
SMOKE_SIZES = [SIZES_TINY]

# Device from environment or auto-detect
def get_device() -> str:
    """Get device from environment or auto-detect."""
    device = os.environ.get("RHENIUM_PERF_DEVICE", "auto")
    if device == "auto":
        try:
            import torch
            if torch.cuda.is_available():
                return "cuda"
        except ImportError:
            pass
        return "cpu"
    return device


@pytest.fixture(scope="session")
def perf_harness():
    """Session-scoped performance harness."""
    if not HAS_HARNESS:
        pytest.skip("rhenium.bench not available")
    
    config = PerfConfig(
        warmup_runs=2,
        measurement_runs=5,
        device=get_device(),
        collect_memory=True,
        gc_before_run=True,
    )
    harness = PerfHarness(config)
    yield harness
    
    # Save report after all tests
    output_dir = Path("artifacts/perf")
    output_dir.mkdir(parents=True, exist_ok=True)
    harness.save_report(output_dir / "report.json")


@pytest.fixture(scope="session")
def synthetic_generator():
    """Session-scoped synthetic data generator."""
    from rhenium.testing.synthetic import SyntheticDataGenerator
    return SyntheticDataGenerator(seed=42)


@pytest.fixture
def tiny_volume(synthetic_generator):
    """Tiny volume for fast tests."""
    return synthetic_generator.generate_volume(shape=SIZES_TINY, modality="MRI")


@pytest.fixture
def small_volume(synthetic_generator):
    """Small volume for standard tests."""
    return synthetic_generator.generate_volume(shape=SIZES_SMALL, modality="MRI")


def make_volume(shape: tuple, modality: str, seed: int = 42):
    """Create synthetic volume."""
    from rhenium.testing.synthetic import SyntheticDataGenerator
    gen = SyntheticDataGenerator(seed=seed)
    return gen.generate_volume(shape=shape, modality=modality)


def make_core_model(device: str = "cpu"):
    """Create initialized core model."""
    from rhenium.models import RheniumCoreModel, RheniumCoreModelConfig
    config = RheniumCoreModelConfig(
        device=device,
        seed=42,
        segmentation_features=[8, 16, 32, 64],
        generator_features=16,
        generator_rrdb_blocks=2,
    )
    model = RheniumCoreModel(config)
    model.initialize()
    return model


@pytest.fixture(scope="module")
def core_model():
    """Module-scoped core model."""
    device = get_device()
    model = make_core_model(device)
    yield model
    model.shutdown()
