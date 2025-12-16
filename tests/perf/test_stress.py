
import concurrent.futures
import time
import pytest
import numpy as np
import psutil
import sys
from rhenium.models import RheniumCoreModel, RheniumCoreModelConfig, TaskType
from rhenium.testing.synthetic import SyntheticDataGenerator
from rhenium.bench.perf_harness import PerfHarness, PerfConfig

# Mark all tests in this file as performance tests
pytestmark = [pytest.mark.perf, pytest.mark.stress]

@pytest.fixture(scope="module")
def stress_harness(perf_harness):
    # Reuse the session harness but methods will override config
    return perf_harness

class TestStressConcurrency:
    """Stress tests for concurrent usage."""
    
    def test_concurrent_segmentation(self, core_model, stress_harness):
        """Test performance under concurrent request load."""
        N_THREADS = 4
        N_REQUESTS = 16
        
        gen = SyntheticDataGenerator(seed=42)
        volume = gen.generate_volume(shape=(64, 64, 32), modality="MRI")
        
        def _worker():
            return core_model.run(volume, task=TaskType.SEGMENTATION)
            
        start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=N_THREADS) as executor:
            futures = [executor.submit(_worker) for _ in range(N_REQUESTS)]
            results = [f.result() for f in futures]
            
        duration = time.time() - start
        rps = N_REQUESTS / duration
        
        # Verify all succeeded
        assert len(results) == N_REQUESTS
        assert all(r.output is not None for r in results)
        
        stress_harness.run_benchmark(
            func=lambda: None, # Dummy, we already measured
            benchmark_id="concurrent_seg_4threads",
            category="stress_concurrency",
            task=f"segmentation_{N_THREADS}threads",
            modality="MRI",
            input_shape=list(volume.shape),
            voxel_count=0,
            warmup_runs=0,
            measurement_runs=1
        )
        # Manually inject correct latency if desired, but this dummy run just logs it exists
        # To be perfect, we should log the RPS in 'extra'
        stress_harness.results[-1].extra["rps"] = round(rps, 2)
        
        print(f"\nConcurrent RPS: {rps:.2f}")

class TestStressStability:
    """Soak/Stability tests."""
    
    @pytest.mark.slow
    def test_memory_leak_check(self, core_model, stress_harness):
        """Run 50 iterations and ensure RSS doesn't grow significantly."""
        gen = SyntheticDataGenerator(seed=42)
        volume = gen.generate_volume(shape=(32, 32, 32), modality="MRI")
        
        process = psutil.Process()
        rss_start = process.memory_info().rss / 1024 / 1024
        
        # Warmup
        for _ in range(5):
             core_model.run(volume, task=TaskType.SEGMENTATION)
             
        rss_warm = process.memory_info().rss / 1024 / 1024
        
        # Soak
        for _ in range(50):
            core_model.run(volume, task=TaskType.SEGMENTATION)
            
        rss_end = process.memory_info().rss / 1024 / 1024
        growth = rss_end - rss_warm
        
        # Record output
        stress_harness.run_benchmark(
            func=lambda: None,
            benchmark_id="memory_leak_50iters",
            category="stress_stability",
            task="segmentation_soak",
            modality="MRI",
            input_shape=list(volume.shape),
            voxel_count=0,
            warmup_runs=0,
            measurement_runs=1
        )
        stress_harness.results[-1].extra["rss_growth_mb"] = round(growth, 2)

        print(f"\nMemory Start: {rss_start:.2f}MB, Warm: {rss_warm:.2f}MB, End: {rss_end:.2f}MB, Growth: {growth:.2f}MB")
        
        # Strict check: Less than 50MB growth over 50 runs (allow some fragmentation)
        assert growth < 50.0, f"Possible memory leak detected: {growth:.2f}MB growth"

class TestScalability:
    """Scalability tests with large volumes."""
    
    @pytest.mark.parametrize("shape", [
        (64, 64, 64),
        (128, 128, 64), 
        (128, 128, 128)
    ])
    def test_volume_scalability(self, core_model, stress_harness, shape):
        """Benchmark segmentation scaling with volume size."""
        gen = SyntheticDataGenerator(seed=42)
        volume = gen.generate_volume(shape=shape, modality="MRI")
        voxels = np.prod(shape)
        
        def _run():
            core_model.run(volume, task=TaskType.SEGMENTATION)
            
        stress_harness.run_benchmark(
            func=_run,
            benchmark_id=f"scale_sec_{shape[0]}x{shape[1]}x{shape[2]}",
            category="scalability",
            task="segmentation",
            input_shape=list(shape),
            voxel_count=voxels,
            enable_profiling=(shape == (128, 128, 128)), # Profile the largest one
            warmup_runs=1,
            measurement_runs=3
        )

if __name__ == "__main__":
    sys.exit(pytest.main(["-v", __file__]))
