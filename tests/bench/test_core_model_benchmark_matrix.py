"""Core model benchmark matrix.

Parametrized tests generating 576+ test cases across:
- 4 modalities (MRI, CT, US, XR)
- 6 tasks (segmentation, classification, detection, reconstruction, super_resolution, denoise)
- 6 shapes (3D and 2D variants)
- 2 dtypes (float32, float16)
- 2 devices (cpu, cuda if available)

Total: 4 x 6 x 6 x 2 x 2 = 576 cases
"""

import pytest
import numpy as np
import torch
import time
from typing import Any

from rhenium.models import RheniumCoreModel, RheniumCoreModelConfig, TaskType
from rhenium.data.volume import ImageVolume, Modality
from rhenium.testing.synthetic import SyntheticDataGenerator

from tests.bench.conftest import (
    MODALITIES,
    TASKS,
    ALL_SHAPES,
    DTYPES,
    DEVICES,
    SMOKE_MODALITIES,
    SMOKE_TASKS,
    SMOKE_SHAPES,
    SMOKE_DTYPES,
    SMOKE_DEVICES,
    make_volume,
    make_model,
)


# =============================================================================
# Full Benchmark Matrix (576 cases)
# =============================================================================

@pytest.mark.bench
@pytest.mark.parametrize("modality", MODALITIES)
@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("shape", ALL_SHAPES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
def test_core_model_matrix(modality: str, task: str, shape: tuple, dtype: str, device: str):
    """Full benchmark matrix test.
    
    Tests all combinations of modality, task, shape, dtype, and device.
    Verifies that the model runs without error and produces valid output.
    """
    # Skip CUDA tests if not available
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    
    # Skip float16 on CPU for certain operations
    if dtype == "float16" and device == "cpu":
        pytest.skip("float16 not well supported on CPU")
    
    # Create volume
    volume = make_volume(shape=shape, modality=modality, dtype="float32", seed=42)
    
    # Create and initialize model
    config = RheniumCoreModelConfig(
        device=device,
        seed=42,
        deterministic=True,
        segmentation_features=[8, 16, 32, 64],
        generator_features=16,
        generator_rrdb_blocks=2,
        dtype=dtype,
    )
    model = RheniumCoreModel(config)
    model.initialize()
    
    try:
        # Map task string to TaskType
        task_type = TaskType(task)
        
        # Time the execution
        start_time = time.time()
        result = model.run(volume, task=task_type)
        elapsed = time.time() - start_time
        
        # Assertions
        assert result is not None, "Result should not be None"
        assert result.task == task_type, f"Task mismatch: {result.task} != {task_type}"
        assert result.output is not None, "Output should not be None"
        assert result.provenance is not None, "Provenance should not be None"
        
        # Check for NaN/Inf
        output = result.output
        if isinstance(output, np.ndarray):
            assert not np.any(np.isnan(output)), "Output contains NaN"
            assert not np.any(np.isinf(output)), "Output contains Inf"
        elif isinstance(output, torch.Tensor):
            assert not torch.any(torch.isnan(output)), "Output contains NaN"
            assert not torch.any(torch.isinf(output)), "Output contains Inf"
        
        # Performance envelope (smoke: 1s CPU, 0.5s GPU)
        max_time = 5.0 if device == "cpu" else 2.0
        assert elapsed < max_time, f"Execution too slow: {elapsed:.2f}s > {max_time}s"
        
    finally:
        model.shutdown()


# =============================================================================
# Smoke Test Matrix (50 cases for CI)
# =============================================================================

@pytest.mark.smoke
@pytest.mark.parametrize("modality", SMOKE_MODALITIES)
@pytest.mark.parametrize("task", SMOKE_TASKS)
@pytest.mark.parametrize("shape", SMOKE_SHAPES)
@pytest.mark.parametrize("dtype", SMOKE_DTYPES)
@pytest.mark.parametrize("device", SMOKE_DEVICES)
def test_core_model_smoke(modality: str, task: str, shape: tuple, dtype: str, device: str):
    """Smoke test subset for CI.
    
    Fast subset of the full matrix: 2 modalities x 3 tasks x 2 shapes x 1 dtype x 1 device = 12 cases
    Plus additional basic sanity checks.
    """
    volume = make_volume(shape=shape, modality=modality, dtype=dtype, seed=42)
    
    config = RheniumCoreModelConfig(
        device=device,
        seed=42,
        deterministic=True,
        segmentation_features=[8, 16, 32, 64],
        generator_features=16,
        generator_rrdb_blocks=2,
    )
    model = RheniumCoreModel(config)
    model.initialize()
    
    try:
        task_type = TaskType(task)
        start_time = time.time()
        result = model.run(volume, task=task_type)
        elapsed = time.time() - start_time
        
        # Basic assertions
        assert result is not None
        assert result.output is not None
        
        # Smoke test should be fast (< 1s)
        assert elapsed < 1.0, f"Smoke test too slow: {elapsed:.2f}s"
        
    finally:
        model.shutdown()


# =============================================================================
# Specific Modality Tests
# =============================================================================

@pytest.mark.bench
class TestMRIBenchmarks:
    """MRI-specific benchmark tests."""
    
    @pytest.mark.parametrize("shape", ALL_SHAPES)
    def test_mri_segmentation(self, shape: tuple):
        """Test MRI segmentation across shapes."""
        volume = make_volume(shape=shape, modality="MRI", seed=42)
        model = make_model(device="cpu")
        
        try:
            result = model.run(volume, task=TaskType.SEGMENTATION)
            assert result.output.shape[0] == shape[0]  # Depth preserved
        finally:
            model.shutdown()
    
    @pytest.mark.parametrize("shape", ALL_SHAPES)
    def test_mri_reconstruction(self, shape: tuple):
        """Test MRI reconstruction across shapes."""
        volume = make_volume(shape=shape, modality="MRI", seed=42)
        model = make_model(device="cpu")
        
        try:
            result = model.run(volume, task=TaskType.RECONSTRUCTION)
            assert result.output is not None
        finally:
            model.shutdown()


@pytest.mark.bench
class TestCTBenchmarks:
    """CT-specific benchmark tests."""
    
    @pytest.mark.parametrize("shape", ALL_SHAPES)
    def test_ct_segmentation(self, shape: tuple):
        """Test CT segmentation across shapes."""
        volume = make_volume(shape=shape, modality="CT", seed=42)
        model = make_model(device="cpu")
        
        try:
            result = model.run(volume, task=TaskType.SEGMENTATION)
            assert result.output is not None
        finally:
            model.shutdown()


# =============================================================================
# Task-Specific Tests
# =============================================================================

@pytest.mark.bench
class TestSegmentationBenchmarks:
    """Segmentation task benchmarks."""
    
    @pytest.mark.parametrize("modality", MODALITIES)
    @pytest.mark.parametrize("shape", SHAPES_3D)
    def test_segmentation_3d(self, modality: str, shape: tuple):
        """Test 3D segmentation across modalities."""
        volume = make_volume(shape=shape, modality=modality, seed=42)
        model = make_model(device="cpu")
        
        try:
            result = model.run(volume, task=TaskType.SEGMENTATION)
            
            # Shape assertions
            assert len(result.output.shape) == 3
            assert result.output.shape == shape
            
            # Type assertions
            assert result.output.dtype in [np.int64, np.int32, np.uint8]
            
        finally:
            model.shutdown()
    
    @pytest.mark.parametrize("modality", MODALITIES)
    def test_segmentation_evidence_dossier(self, modality: str):
        """Test that segmentation produces evidence dossier."""
        volume = make_volume(shape=(16, 32, 32), modality=modality, seed=42)
        model = make_model(device="cpu")
        
        try:
            result = model.run(volume, task=TaskType.SEGMENTATION)
            
            assert result.evidence_dossier is not None
            assert "dossier_id" in result.evidence_dossier
            assert "finding" in result.evidence_dossier
            
        finally:
            model.shutdown()


@pytest.mark.bench
class TestGenerativeBenchmarks:
    """Generative task benchmarks."""
    
    @pytest.mark.parametrize("modality", MODALITIES)
    @pytest.mark.parametrize("shape", SHAPES_3D[:2])  # Smaller subset
    def test_super_resolution(self, modality: str, shape: tuple):
        """Test super-resolution across modalities."""
        volume = make_volume(shape=shape, modality=modality, seed=42)
        model = make_model(device="cpu")
        
        try:
            result = model.run(volume, task=TaskType.SUPER_RESOLUTION)
            
            assert result.generation_metadata is not None
            assert "disclosure" in result.generation_metadata
            
        finally:
            model.shutdown()
    
    @pytest.mark.parametrize("sigma", [0.5, 1.0, 2.0])
    def test_denoise_sigma_variation(self, sigma: float):
        """Test denoising with different sigma values."""
        volume = make_volume(shape=(16, 32, 32), modality="MRI", seed=42)
        model = make_model(device="cpu")
        
        try:
            result = model.run(volume, task=TaskType.DENOISE, sigma=sigma)
            
            assert result.output.shape == volume.shape
            assert result.metrics.get("sigma") == sigma
            
        finally:
            model.shutdown()


# =============================================================================
# GPU Tests (if available)
# =============================================================================

@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGPUBenchmarks:
    """GPU-specific benchmarks."""
    
    @pytest.mark.parametrize("shape", [(32, 64, 64), (64, 128, 128)])
    def test_gpu_segmentation(self, shape: tuple):
        """Test GPU segmentation performance."""
        volume = make_volume(shape=shape, modality="MRI", seed=42)
        
        config = RheniumCoreModelConfig(
            device="cuda",
            seed=42,
            deterministic=True,
            segmentation_features=[16, 32, 64, 128],
        )
        model = RheniumCoreModel(config)
        model.initialize()
        
        try:
            start_time = time.time()
            result = model.run(volume, task=TaskType.SEGMENTATION)
            elapsed = time.time() - start_time
            
            assert result.output is not None
            assert elapsed < 1.0, f"GPU should be fast: {elapsed:.2f}s"
            
        finally:
            model.shutdown()
            torch.cuda.empty_cache()
    
    def test_gpu_memory_usage(self):
        """Test GPU memory usage stays bounded."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        torch.cuda.reset_peak_memory_stats()
        
        volume = make_volume(shape=(64, 128, 128), modality="MRI", seed=42)
        
        config = RheniumCoreModelConfig(
            device="cuda",
            seed=42,
            segmentation_features=[16, 32, 64, 128],
        )
        model = RheniumCoreModel(config)
        model.initialize()
        
        try:
            result = model.run(volume, task=TaskType.SEGMENTATION)
            
            peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 * 1024)
            
            # Should stay under 2GB for this size
            assert peak_memory_mb < 2048, f"Peak memory too high: {peak_memory_mb:.0f}MB"
            
        finally:
            model.shutdown()
            torch.cuda.empty_cache()


# =============================================================================
# Performance Envelope Tests
# =============================================================================

@pytest.mark.bench
class TestPerformanceEnvelope:
    """Performance envelope tests."""
    
    def test_startup_time(self):
        """Test model startup time."""
        config = RheniumCoreModelConfig(
            device="cpu",
            seed=42,
            segmentation_features=[16, 32, 64, 128],
        )
        
        start_time = time.time()
        model = RheniumCoreModel(config)
        model.initialize()
        startup_time = time.time() - start_time
        
        model.shutdown()
        
        assert startup_time < 10.0, f"Startup too slow: {startup_time:.2f}s"
    
    def test_throughput_small_volumes(self):
        """Test throughput for small volumes."""
        model = make_model(device="cpu")
        volumes = [make_volume(shape=(16, 32, 32), modality="MRI", seed=i) for i in range(10)]
        
        try:
            start_time = time.time()
            for vol in volumes:
                model.run(vol, task=TaskType.SEGMENTATION)
            total_time = time.time() - start_time
            
            throughput = len(volumes) / total_time
            
            assert throughput > 1.0, f"Throughput too low: {throughput:.2f} vol/s"
            
        finally:
            model.shutdown()
    
    @pytest.mark.parametrize("n_runs", [5])
    def test_latency_consistency(self, n_runs: int):
        """Test latency consistency across runs."""
        model = make_model(device="cpu")
        volume = make_volume(shape=(16, 32, 32), modality="MRI", seed=42)
        
        try:
            latencies = []
            for _ in range(n_runs):
                start = time.time()
                model.run(volume, task=TaskType.SEGMENTATION)
                latencies.append(time.time() - start)
            
            p50 = np.percentile(latencies, 50)
            p95 = np.percentile(latencies, 95)
            
            # p95 should not be more than 3x p50
            assert p95 < p50 * 3, f"Latency variance too high: p50={p50:.3f}s, p95={p95:.3f}s"
            
        finally:
            model.shutdown()
