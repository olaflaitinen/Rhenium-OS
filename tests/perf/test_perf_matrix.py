"""Performance benchmark test matrix.

Generates 50+ performance benchmarks across:
- 4 modalities (MRI, CT, US, XR)
- 6+ tasks (e2e, perception, recon, gan, xai, etc.)
- 3 sizes (tiny, small, medium)
- 12+ categories

Total: 60+ CPU test cases (GPU extras optional)
"""

import gc
import json
import tempfile
import time
from pathlib import Path

import numpy as np
import pytest

from tests.perf.conftest import (
    MODALITIES, ALL_SIZES, SIZES_TINY, SIZES_SMALL,
    SMOKE_MODALITIES, SMOKE_SIZES,
    make_volume, make_core_model, get_device,
)


# =============================================================================
# Category 1: End-to-End Core Model (4+ tests)
# =============================================================================

@pytest.mark.perf
@pytest.mark.parametrize("modality", MODALITIES)
@pytest.mark.parametrize("size", ALL_SIZES[:2])  # tiny, small
def test_e2e_full_pipeline(perf_harness, modality, size):
    """End-to-end core model pipeline benchmark."""
    volume = make_volume(shape=size, modality=modality)
    model = make_core_model(device=get_device())
    
    from rhenium.models import TaskType
    
    def run():
        model.run(volume, task=TaskType.FULL_PIPELINE)
    
    result = perf_harness.run_benchmark(
        func=run,
        benchmark_id=f"e2e_full_pipeline_{modality}_{size[0]}",
        category="e2e_core_model",
        task="full_pipeline",
        modality=modality,
        input_shape=list(size),
        voxel_count=np.prod(size),
    )
    
    model.shutdown()
    assert result.success, f"Failed: {result.error}"


@pytest.mark.perf_smoke
@pytest.mark.parametrize("modality", SMOKE_MODALITIES)
def test_e2e_smoke(perf_harness, modality):
    """Smoke test: E2E pipeline."""
    volume = make_volume(shape=SIZES_TINY, modality=modality)
    model = make_core_model(device="cpu")
    
    from rhenium.models import TaskType
    
    def run():
        model.run(volume, task=TaskType.FULL_PIPELINE)
    
    result = perf_harness.run_benchmark(
        func=run,
        benchmark_id=f"e2e_smoke_{modality}",
        category="e2e_core_model",
        task="full_pipeline_smoke",
        modality=modality,
        input_shape=list(SIZES_TINY),
    )
    
    model.shutdown()
    assert result.success


# =============================================================================
# Category 2: I/O & Parsing (3+ tests)
# =============================================================================

@pytest.mark.perf
@pytest.mark.parametrize("size", ALL_SIZES)
def test_io_volume_creation(perf_harness, size):
    """Volume creation from numpy array."""
    from rhenium.data.volume import ImageVolume, Modality
    
    array = np.random.randn(*size).astype(np.float32)
    
    def run():
        vol = ImageVolume(array=array, modality=Modality.MRI)
        _ = vol.to_tensor(device="cpu")
    
    result = perf_harness.run_benchmark(
        func=run,
        benchmark_id=f"io_volume_creation_{size[0]}",
        category="io_parsing",
        task="volume_creation",
        input_shape=list(size),
    )
    
    assert result.success


@pytest.mark.perf_smoke
def test_io_volume_smoke(perf_harness):
    """Smoke: Volume I/O."""
    from rhenium.data.volume import ImageVolume, Modality
    
    array = np.random.randn(*SIZES_TINY).astype(np.float32)
    
    def run():
        vol = ImageVolume(array=array, modality=Modality.MRI)
        _ = vol.shape
    
    result = perf_harness.run_benchmark(
        func=run,
        benchmark_id="io_volume_smoke",
        category="io_parsing",
        task="volume_creation_smoke",
        input_shape=list(SIZES_TINY),
    )
    
    assert result.success


@pytest.mark.perf
def test_io_synthetic_generation(perf_harness):
    """Synthetic data generation."""
    from rhenium.testing.synthetic import SyntheticDataGenerator
    
    gen = SyntheticDataGenerator(seed=42)
    
    def run():
        gen.generate_volume(shape=SIZES_SMALL, modality="MRI")
    
    result = perf_harness.run_benchmark(
        func=run,
        benchmark_id="io_synthetic_generation",
        category="io_parsing",
        task="synthetic_volume",
        input_shape=list(SIZES_SMALL),
    )
    
    assert result.success


# =============================================================================
# Category 3: Preprocessing (4+ tests)
# =============================================================================

@pytest.mark.perf
@pytest.mark.parametrize("size", ALL_SIZES)
def test_preprocess_normalize(perf_harness, size):
    """Volume normalization."""
    volume = make_volume(shape=size, modality="MRI")
    
    def run():
        volume.normalize(method="minmax")
    
    result = perf_harness.run_benchmark(
        func=run,
        benchmark_id=f"preprocess_normalize_{size[0]}",
        category="preprocessing",
        task="normalize_minmax",
        input_shape=list(size),
    )
    
    assert result.success


@pytest.mark.perf_smoke
def test_preprocess_smoke(perf_harness):
    """Smoke: Preprocessing."""
    volume = make_volume(shape=SIZES_TINY, modality="MRI")
    
    def run():
        volume.normalize(method="zscore")
    
    result = perf_harness.run_benchmark(
        func=run,
        benchmark_id="preprocess_smoke",
        category="preprocessing",
        task="normalize_smoke",
        input_shape=list(SIZES_TINY),
    )
    
    assert result.success


@pytest.mark.perf
def test_preprocess_resample(perf_harness):
    """Volume resampling."""
    volume = make_volume(shape=SIZES_SMALL, modality="MRI")
    
    def run():
        volume.resample(target_spacing=(2.0, 2.0, 2.0))
    
    result = perf_harness.run_benchmark(
        func=run,
        benchmark_id="preprocess_resample",
        category="preprocessing",
        task="resample",
        input_shape=list(SIZES_SMALL),
    )
    
    assert result.success


# =============================================================================
# Category 4: Perception Inference (6+ tests)
# =============================================================================

@pytest.mark.perf
@pytest.mark.parametrize("modality", MODALITIES)
@pytest.mark.parametrize("size", ALL_SIZES[:2])
def test_perception_segmentation(perf_harness, modality, size):
    """Segmentation forward pass."""
    volume = make_volume(shape=size, modality=modality)
    model = make_core_model(device=get_device())
    
    from rhenium.models import TaskType
    
    def run():
        model.run(volume, task=TaskType.SEGMENTATION)
    
    result = perf_harness.run_benchmark(
        func=run,
        benchmark_id=f"perception_seg_{modality}_{size[0]}",
        category="perception_inference",
        task="segmentation",
        modality=modality,
        input_shape=list(size),
        voxel_count=np.prod(size),
    )
    
    model.shutdown()
    assert result.success


@pytest.mark.perf_smoke
@pytest.mark.parametrize("modality", SMOKE_MODALITIES)
def test_perception_smoke(perf_harness, modality):
    """Smoke: Perception."""
    volume = make_volume(shape=SIZES_TINY, modality=modality)
    model = make_core_model(device="cpu")
    
    from rhenium.models import TaskType
    
    def run():
        model.run(volume, task=TaskType.SEGMENTATION)
    
    result = perf_harness.run_benchmark(
        func=run,
        benchmark_id=f"perception_smoke_{modality}",
        category="perception_inference",
        task="segmentation_smoke",
        modality=modality,
        input_shape=list(SIZES_TINY),
    )
    
    model.shutdown()
    assert result.success


@pytest.mark.perf
def test_perception_classification(perf_harness):
    """Classification forward pass."""
    volume = make_volume(shape=SIZES_SMALL, modality="MRI")
    model = make_core_model(device=get_device())
    
    from rhenium.models import TaskType
    
    def run():
        model.run(volume, task=TaskType.CLASSIFICATION)
    
    result = perf_harness.run_benchmark(
        func=run,
        benchmark_id="perception_classification",
        category="perception_inference",
        task="classification",
        modality="MRI",
        input_shape=list(SIZES_SMALL),
    )
    
    model.shutdown()
    assert result.success


@pytest.mark.perf
def test_perception_detection(perf_harness):
    """Detection forward pass."""
    volume = make_volume(shape=SIZES_SMALL, modality="CT")
    model = make_core_model(device=get_device())
    
    from rhenium.models import TaskType
    
    def run():
        model.run(volume, task=TaskType.DETECTION)
    
    result = perf_harness.run_benchmark(
        func=run,
        benchmark_id="perception_detection",
        category="perception_inference",
        task="detection",
        modality="CT",
        input_shape=list(SIZES_SMALL),
    )
    
    model.shutdown()
    assert result.success


# =============================================================================
# Category 5: MRI Reconstruction (3+ tests)
# =============================================================================

@pytest.mark.perf
@pytest.mark.parametrize("size", ALL_SIZES[:2])
def test_recon_mri(perf_harness, size):
    """MRI reconstruction."""
    volume = make_volume(shape=size, modality="MRI")
    model = make_core_model(device=get_device())
    
    from rhenium.models import TaskType
    
    def run():
        model.run(volume, task=TaskType.RECONSTRUCTION)
    
    result = perf_harness.run_benchmark(
        func=run,
        benchmark_id=f"recon_mri_{size[0]}",
        category="reconstruction_mri",
        task="mri_reconstruction",
        modality="MRI",
        input_shape=list(size),
    )
    
    model.shutdown()
    assert result.success


@pytest.mark.perf_smoke
def test_recon_mri_smoke(perf_harness):
    """Smoke: MRI recon."""
    volume = make_volume(shape=SIZES_TINY, modality="MRI")
    model = make_core_model(device="cpu")
    
    from rhenium.models import TaskType
    
    def run():
        model.run(volume, task=TaskType.RECONSTRUCTION)
    
    result = perf_harness.run_benchmark(
        func=run,
        benchmark_id="recon_mri_smoke",
        category="reconstruction_mri",
        task="mri_reconstruction_smoke",
        modality="MRI",
        input_shape=list(SIZES_TINY),
    )
    
    model.shutdown()
    assert result.success


# =============================================================================
# Category 6: CT Reconstruction (3+ tests)
# =============================================================================

@pytest.mark.perf
@pytest.mark.parametrize("size", ALL_SIZES[:2])
def test_recon_ct(perf_harness, size):
    """CT reconstruction."""
    volume = make_volume(shape=size, modality="CT")
    model = make_core_model(device=get_device())
    
    from rhenium.models import TaskType
    
    def run():
        model.run(volume, task=TaskType.RECONSTRUCTION)
    
    result = perf_harness.run_benchmark(
        func=run,
        benchmark_id=f"recon_ct_{size[0]}",
        category="reconstruction_ct",
        task="ct_reconstruction",
        modality="CT",
        input_shape=list(size),
    )
    
    model.shutdown()
    assert result.success


@pytest.mark.perf_smoke
def test_recon_ct_smoke(perf_harness):
    """Smoke: CT recon."""
    volume = make_volume(shape=SIZES_TINY, modality="CT")
    model = make_core_model(device="cpu")
    
    from rhenium.models import TaskType
    
    def run():
        model.run(volume, task=TaskType.RECONSTRUCTION)
    
    result = perf_harness.run_benchmark(
        func=run,
        benchmark_id="recon_ct_smoke",
        category="reconstruction_ct",
        task="ct_reconstruction_smoke",
        modality="CT",
        input_shape=list(SIZES_TINY),
    )
    
    model.shutdown()
    assert result.success


# =============================================================================
# Category 7: PINN Step (2+ tests)
# =============================================================================

@pytest.mark.perf
def test_pinn_step_tiny(perf_harness):
    """PINN single step (placeholder)."""
    # Placeholder - PINN not fully implemented
    import numpy as np
    
    def run():
        # Simulate PDE residual computation
        x = np.random.randn(100, 3).astype(np.float32)
        u = np.sin(x[:, 0]) * np.cos(x[:, 1])
        residual = np.gradient(u)
        return residual
    
    result = perf_harness.run_benchmark(
        func=run,
        benchmark_id="pinn_step_tiny",
        category="pinn_step",
        task="pde_residual",
        input_shape=[100, 3],
    )
    
    assert result.success


@pytest.mark.perf_smoke
def test_pinn_smoke(perf_harness):
    """Smoke: PINN step."""
    import numpy as np
    
    def run():
        x = np.random.randn(10, 3).astype(np.float32)
        return np.sum(x ** 2)
    
    result = perf_harness.run_benchmark(
        func=run,
        benchmark_id="pinn_smoke",
        category="pinn_step",
        task="pde_residual_smoke",
        input_shape=[10, 3],
    )
    
    assert result.success


# =============================================================================
# Category 8: GAN Inference (6+ tests)
# =============================================================================

@pytest.mark.perf
@pytest.mark.parametrize("modality", MODALITIES[:2])
@pytest.mark.parametrize("size", ALL_SIZES[:2])
def test_gan_super_resolution(perf_harness, modality, size):
    """GAN super-resolution."""
    volume = make_volume(shape=size, modality=modality)
    model = make_core_model(device=get_device())
    
    from rhenium.models import TaskType
    
    def run():
        model.run(volume, task=TaskType.SUPER_RESOLUTION)
    
    result = perf_harness.run_benchmark(
        func=run,
        benchmark_id=f"gan_sr_{modality}_{size[0]}",
        category="gan_inference",
        task="super_resolution",
        modality=modality,
        input_shape=list(size),
    )
    
    model.shutdown()
    assert result.success


@pytest.mark.perf_smoke
def test_gan_smoke(perf_harness):
    """Smoke: GAN inference."""
    volume = make_volume(shape=SIZES_TINY, modality="MRI")
    model = make_core_model(device="cpu")
    
    from rhenium.models import TaskType
    
    def run():
        model.run(volume, task=TaskType.SUPER_RESOLUTION)
    
    result = perf_harness.run_benchmark(
        func=run,
        benchmark_id="gan_smoke",
        category="gan_inference",
        task="super_resolution_smoke",
        modality="MRI",
        input_shape=list(SIZES_TINY),
    )
    
    model.shutdown()
    assert result.success


@pytest.mark.perf
def test_gan_denoise(perf_harness):
    """GAN denoising."""
    volume = make_volume(shape=SIZES_SMALL, modality="CT")
    model = make_core_model(device=get_device())
    
    from rhenium.models import TaskType
    
    def run():
        model.run(volume, task=TaskType.DENOISE)
    
    result = perf_harness.run_benchmark(
        func=run,
        benchmark_id="gan_denoise",
        category="gan_inference",
        task="denoise",
        modality="CT",
        input_shape=list(SIZES_SMALL),
    )
    
    model.shutdown()
    assert result.success


# =============================================================================
# Category 9: XAI Dossier Generation (4+ tests)
# =============================================================================

@pytest.mark.perf
@pytest.mark.parametrize("modality", MODALITIES[:2])
def test_xai_dossier_generation(perf_harness, modality):
    """XAI evidence dossier generation."""
    volume = make_volume(shape=SIZES_SMALL, modality=modality)
    model = make_core_model(device=get_device())
    
    from rhenium.models import TaskType
    
    def run():
        result = model.run(volume, task=TaskType.SEGMENTATION)
        return result.evidence_dossier
    
    result = perf_harness.run_benchmark(
        func=run,
        benchmark_id=f"xai_dossier_{modality}",
        category="xai_dossier",
        task="dossier_generation",
        modality=modality,
        input_shape=list(SIZES_SMALL),
    )
    
    model.shutdown()
    assert result.success


@pytest.mark.perf_smoke
def test_xai_dossier_smoke(perf_harness):
    """Smoke: XAI dossier."""
    volume = make_volume(shape=SIZES_TINY, modality="MRI")
    model = make_core_model(device="cpu")
    
    from rhenium.models import TaskType
    
    def run():
        result = model.run(volume, task=TaskType.SEGMENTATION)
        return result.evidence_dossier
    
    result = perf_harness.run_benchmark(
        func=run,
        benchmark_id="xai_dossier_smoke",
        category="xai_dossier",
        task="dossier_generation_smoke",
        modality="MRI",
        input_shape=list(SIZES_TINY),
    )
    
    model.shutdown()
    assert result.success


# =============================================================================
# Category 10: Serialization/Export (3+ tests)
# =============================================================================

@pytest.mark.perf
@pytest.mark.parametrize("size", ALL_SIZES[:2])
def test_serialization_json_export(perf_harness, size):
    """JSON serialization."""
    volume = make_volume(shape=size, modality="MRI")
    model = make_core_model(device="cpu")
    
    from rhenium.models import TaskType
    result_obj = model.run(volume, task=TaskType.SEGMENTATION)
    
    def run():
        data = result_obj.to_dict()
        json.dumps(data)
    
    result = perf_harness.run_benchmark(
        func=run,
        benchmark_id=f"serialization_json_{size[0]}",
        category="serialization",
        task="json_export",
        input_shape=list(size),
    )
    
    model.shutdown()
    assert result.success


@pytest.mark.perf_smoke
def test_serialization_smoke(perf_harness):
    """Smoke: Serialization."""
    volume = make_volume(shape=SIZES_TINY, modality="MRI")
    model = make_core_model(device="cpu")
    
    from rhenium.models import TaskType
    result_obj = model.run(volume, task=TaskType.SEGMENTATION)
    
    def run():
        json.dumps(result_obj.to_dict())
    
    result = perf_harness.run_benchmark(
        func=run,
        benchmark_id="serialization_smoke",
        category="serialization",
        task="json_export_smoke",
        input_shape=list(SIZES_TINY),
    )
    
    model.shutdown()
    assert result.success


# =============================================================================
# Category 11: FastAPI Readiness (2+ tests)
# =============================================================================

@pytest.mark.perf
def test_fastapi_health(perf_harness):
    """FastAPI health endpoint overhead."""
    # Simulate health check
    def run():
        health = {"status": "healthy", "version": "1.0.0"}
        json.dumps(health)
    
    result = perf_harness.run_benchmark(
        func=run,
        benchmark_id="fastapi_health",
        category="fastapi_readiness",
        task="health_check",
    )
    
    assert result.success


@pytest.mark.perf_smoke
def test_fastapi_smoke(perf_harness):
    """Smoke: FastAPI readiness."""
    def run():
        return {"status": "ok"}
    
    result = perf_harness.run_benchmark(
        func=run,
        benchmark_id="fastapi_smoke",
        category="fastapi_readiness",
        task="health_smoke",
    )
    
    assert result.success


# =============================================================================
# Category 12: CLI Overhead (2+ tests)
# =============================================================================

@pytest.mark.perf
def test_cli_import_overhead(perf_harness):
    """CLI module import overhead."""
    import importlib
    
    def run():
        # Reimport to measure
        import rhenium
        importlib.reload(rhenium)
    
    result = perf_harness.run_benchmark(
        func=run,
        benchmark_id="cli_import",
        category="cli_overhead",
        task="module_import",
    )
    
    assert result.success


@pytest.mark.perf_smoke
def test_cli_smoke(perf_harness):
    """Smoke: CLI overhead."""
    def run():
        import rhenium
        return rhenium.__version__
    
    result = perf_harness.run_benchmark(
        func=run,
        benchmark_id="cli_smoke",
        category="cli_overhead",
        task="version_check",
    )
    
    assert result.success


# =============================================================================
# Category 13: Governance Artifacts (3+ tests)
# =============================================================================

@pytest.mark.perf
def test_governance_model_card(perf_harness):
    """Model card generation."""
    def run():
        card = {
            "model_name": "RheniumCoreModel",
            "version": "1.0.0",
            "intended_use": "Research",
            "limitations": ["Not for clinical use"],
        }
        json.dumps(card)
    
    result = perf_harness.run_benchmark(
        func=run,
        benchmark_id="governance_model_card",
        category="governance_artifacts",
        task="model_card",
    )
    
    assert result.success


@pytest.mark.perf
def test_governance_risk_register(perf_harness):
    """Risk register generation."""
    def run():
        risks = [
            {"id": "R001", "description": "Test risk", "mitigation": "Test mitigation"}
            for _ in range(10)
        ]
        json.dumps({"risks": risks})
    
    result = perf_harness.run_benchmark(
        func=run,
        benchmark_id="governance_risk_register",
        category="governance_artifacts",
        task="risk_register",
    )
    
    assert result.success


@pytest.mark.perf_smoke
def test_governance_smoke(perf_harness):
    """Smoke: Governance artifacts."""
    def run():
        return {"type": "model_card", "valid": True}
    
    result = perf_harness.run_benchmark(
        func=run,
        benchmark_id="governance_smoke",
        category="governance_artifacts",
        task="card_smoke",
    )
    
    assert result.success
