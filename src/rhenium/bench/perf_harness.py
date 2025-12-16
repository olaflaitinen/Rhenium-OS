"""Performance Benchmarking Harness for Rhenium OS.

This module provides utilities for running performance benchmarks,
collecting timing/memory metrics, computing statistics, and generating reports.
"""

from __future__ import annotations

import cProfile
import gc
import hashlib
import json
import os
import pstats
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any, Callable

import numpy as np

# Optional imports
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False
    psutil = None

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None

try:
    import pynvml
    pynvml.nvmlInit()
    HAS_PYNVML = True
except (ImportError, Exception):
    HAS_PYNVML = False
    pynvml = None


@dataclass
class PerfConfig:
    """Configuration for performance benchmarking."""

    warmup_runs: int = 3
    measurement_runs: int = 10
    device: str = "cpu"
    collect_memory: bool = True
    collect_gpu_memory: bool = True
    gc_before_run: bool = True
    sync_cuda: bool = True
    seed: int = 42


@dataclass
class LatencyStats:
    """Latency statistics."""

    min_ms: float
    max_ms: float
    mean_ms: float
    std_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    raw_ms: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "min": round(self.min_ms, 3),
            "max": round(self.max_ms, 3),
            "mean": round(self.mean_ms, 3),
            "std": round(self.std_ms, 3),
            "p50": round(self.p50_ms, 3),
            "p95": round(self.p95_ms, 3),
            "p99": round(self.p99_ms, 3),
        }


@dataclass
class MemoryStats:
    """Memory statistics."""

    rss_mb_before: float = 0.0
    rss_mb_after: float = 0.0
    rss_mb_peak: float = 0.0
    vram_mb_before: float | None = None
    vram_mb_after: float | None = None
    vram_mb_peak: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "rss_mb_before": round(self.rss_mb_before, 2),
            "rss_mb_after": round(self.rss_mb_after, 2),
            "rss_mb_peak": round(self.rss_mb_peak, 2),
            "vram_mb_before": round(self.vram_mb_before, 2) if self.vram_mb_before else None,
            "vram_mb_after": round(self.vram_mb_after, 2) if self.vram_mb_after else None,
            "vram_mb_peak": round(self.vram_mb_peak, 2) if self.vram_mb_peak else None,
        }


@dataclass
class ThroughputStats:
    """Throughput statistics."""

    items_per_sec: float
    voxels_per_sec: float | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "items_per_sec": round(self.items_per_sec, 2),
            "voxels_per_sec": round(self.voxels_per_sec, 2) if self.voxels_per_sec else None,
        }


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    id: str
    category: str
    task: str
    success: bool
    latency: LatencyStats | None = None
    throughput: ThroughputStats | None = None
    memory: MemoryStats | None = None
    modality: str | None = None
    input_shape: list[int] | None = None
    dtype: str = "float32"
    device: str = "cpu"
    warmup_runs: int = 0
    measurement_runs: int = 0
    error: str | None = None
    skipped_reason: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "id": self.id,
            "category": self.category,
            "task": self.task,
            "modality": self.modality,
            "input_shape": self.input_shape,
            "dtype": self.dtype,
            "device": self.device,
            "warmup_runs": self.warmup_runs,
            "measurement_runs": self.measurement_runs,
            "success": self.success,
            "error": self.error,
            "skipped_reason": self.skipped_reason,
        }

        if self.latency:
            result["latency_ms"] = self.latency.to_dict()
        if self.throughput:
            result["throughput"] = self.throughput.to_dict()
        if self.memory:
            result["memory"] = self.memory.to_dict()
        if self.extra:
            result["extra"] = self.extra

        return result


class PerfHarness:
    """Performance benchmarking harness."""

    def __init__(self, config: PerfConfig | None = None):
        """Initialize harness.

        Args:
            config: Performance configuration
        """
        self.config = config or PerfConfig()
        self.results: list[BenchmarkResult] = []

    def run_benchmark(
        self,
        func: Callable[[], Any],
        benchmark_id: str,
        category: str,
        task: str,
        modality: str | None = None,
        input_shape: list[int] | None = None,
        dtype: str = "float32",
        voxel_count: int | None = None,
        enable_profiling: bool = False,
        warmup_runs: int | None = None,
        measurement_runs: int | None = None,
    ) -> BenchmarkResult:
        """Run a single benchmark.

        Args:
            func: Function to benchmark (no arguments)
            benchmark_id: Unique identifier
            category: Benchmark category
            task: Task description
            modality: Optional modality (MRI, CT, etc.)
            input_shape: Optional input shape
            dtype: Data type
            voxel_count: Optional voxel count for throughput
            enable_profiling: Whether to run cProfile
            warmup_runs: Override config warmup runs
            measurement_runs: Override config measurement runs

        Returns:
            BenchmarkResult with collected metrics
        """
        # Determine runs
        n_warmup = warmup_runs if warmup_runs is not None else self.config.warmup_runs
        n_measure = measurement_runs if measurement_runs is not None else self.config.measurement_runs

        # Prepare
        if self.config.gc_before_run:
            gc.collect()
            if HAS_TORCH and torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Collect memory before
        memory = MemoryStats()
        if self.config.collect_memory and HAS_PSUTIL:
            memory.rss_mb_before = self._get_rss_mb()
        if self.config.collect_gpu_memory and HAS_TORCH and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
            memory.vram_mb_before = torch.cuda.memory_allocated() / (1024 * 1024)

        # Warmup
        try:
            for _ in range(n_warmup):
                func()
                if self.config.sync_cuda and HAS_TORCH and torch.cuda.is_available():
                    torch.cuda.synchronize()
        except Exception as e:
            return BenchmarkResult(
                id=benchmark_id,
                category=category,
                task=task,
                modality=modality,
                input_shape=input_shape,
                dtype=dtype,
                device=self.config.device,
                success=False,
                error=f"Warmup failed: {e}",
            )

        # Measurement
        latencies_ms = []
        try:
            for _ in range(n_measure):
                if self.config.sync_cuda and HAS_TORCH and torch.cuda.is_available():
                    torch.cuda.synchronize()

                start = time.perf_counter()
                func()

                if self.config.sync_cuda and HAS_TORCH and torch.cuda.is_available():
                    torch.cuda.synchronize()

                elapsed_ms = (time.perf_counter() - start) * 1000
                latencies_ms.append(elapsed_ms)

        except Exception as e:
            return BenchmarkResult(
                id=benchmark_id,
                category=category,
                task=task,
                modality=modality,
                input_shape=input_shape,
                dtype=dtype,
                device=self.config.device,
                success=False,
                error=f"Measurement failed: {e}",
            )

        # Collect memory after
        if self.config.collect_memory and HAS_PSUTIL:
            memory.rss_mb_after = self._get_rss_mb()
            memory.rss_mb_peak = max(memory.rss_mb_before, memory.rss_mb_after)
        if self.config.collect_gpu_memory and HAS_TORCH and torch.cuda.is_available():
            memory.vram_mb_after = torch.cuda.memory_allocated() / (1024 * 1024)
            memory.vram_mb_peak = torch.cuda.max_memory_allocated() / (1024 * 1024)

        # Compute statistics
        latency_stats = self._compute_latency_stats(latencies_ms)

        # Compute throughput
        mean_latency_sec = latency_stats.mean_ms / 1000
        items_per_sec = 1.0 / mean_latency_sec if mean_latency_sec > 0 else 0
        voxels_per_sec = None
        if voxel_count:
            voxels_per_sec = voxel_count / mean_latency_sec if mean_latency_sec > 0 else 0

        throughput = ThroughputStats(
            items_per_sec=items_per_sec,
            voxels_per_sec=voxels_per_sec,
        )

        # Profiling
        extra = {}
        if enable_profiling:
            try:
                profiler = cProfile.Profile()
                profiler.enable()
                func()
                profiler.disable()
                
                s = StringIO()
                ps = pstats.Stats(profiler, stream=s).sort_stats("cumulative")
                ps.print_stats(20)  # Top 20 functions
                extra["profile_summary"] = s.getvalue()
            except Exception as e:
                extra["profile_error"] = str(e)

        result = BenchmarkResult(
            id=benchmark_id,
            category=category,
            task=task,
            modality=modality,
            input_shape=input_shape,
            dtype=dtype,
            device=self.config.device,
            warmup_runs=n_warmup,
            measurement_runs=n_measure,
            success=True,
            latency=latency_stats,
            throughput=throughput,
            memory=memory,
            extra=extra,
        )

        self.results.append(result)
        return result

    def profile_benchmark(
        self,
        func: Callable[[], Any],
        top_n: int = 20,
        sort_by: str = "cumulative",
    ) -> str:
        """Run profiling on the function and return formatted stats."""
        profiler = cProfile.Profile()
        profiler.enable()
        try:
            func()
        finally:
            profiler.disable()
        
        s = StringIO()
        ps = pstats.Stats(profiler, stream=s).sort_stats(sort_by)
        ps.print_stats(top_n)
        return s.getvalue()

    def _compute_latency_stats(self, latencies_ms: list[float]) -> LatencyStats:
        """Compute latency statistics."""
        arr = np.array(latencies_ms)
        return LatencyStats(
            min_ms=float(np.min(arr)),
            max_ms=float(np.max(arr)),
            mean_ms=float(np.mean(arr)),
            std_ms=float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            p50_ms=float(np.percentile(arr, 50)),
            p95_ms=float(np.percentile(arr, 95)),
            p99_ms=float(np.percentile(arr, 99)),
            raw_ms=latencies_ms,
        )

    def _get_rss_mb(self) -> float:
        """Get current RSS in MB."""
        if HAS_PSUTIL:
            return psutil.Process().memory_info().rss / (1024 * 1024)
        return 0.0

    def get_summary(self) -> dict[str, Any]:
        """Generate summary of all results."""
        passed = sum(1 for r in self.results if r.success)
        failed = sum(1 for r in self.results if not r.success and not r.skipped_reason)
        skipped = sum(1 for r in self.results if r.skipped_reason)

        # Top-line metrics from e2e benchmarks
        e2e_results = [r for r in self.results if r.category == "e2e_core_model" and r.success]
        e2e_p50 = np.mean([r.latency.p50_ms for r in e2e_results if r.latency]) if e2e_results else 0
        e2e_p95 = np.mean([r.latency.p95_ms for r in e2e_results if r.latency]) if e2e_results else 0
        e2e_throughput = np.mean([r.throughput.items_per_sec for r in e2e_results if r.throughput]) if e2e_results else 0

        # Peak memory across all
        peak_rss = max((r.memory.rss_mb_peak for r in self.results if r.memory), default=0)
        peak_vram = max((r.memory.vram_mb_peak or 0 for r in self.results if r.memory), default=None)

        return {
            "total_benchmarks": len(self.results),
            "passed": passed,
            "failed": failed,
            "skipped": skipped,
            "top_line_metrics": {
                "e2e_latency_p50_ms": round(e2e_p50, 3),
                "e2e_latency_p95_ms": round(e2e_p95, 3),
                "e2e_throughput_items_per_sec": round(e2e_throughput, 3),
                "peak_memory_rss_mb": round(peak_rss, 2),
                "peak_memory_vram_mb": round(peak_vram, 2) if peak_vram else None,
            },
            "regression_status": {
                "baseline_file": None,
                "has_regressions": False,
                "regression_count": 0,
                "improvement_count": 0,
                "regressions": [],
            },
        }

    def compare_baseline(
        self,
        baseline_path: Path,
        latency_threshold: float = 0.10,
        memory_threshold: float = 0.20,
    ) -> dict[str, Any]:
        """Compare results against baseline.

        Args:
            baseline_path: Path to baseline JSON
            latency_threshold: Regression threshold for latency (default 10%)
            memory_threshold: Regression threshold for memory (default 20%)

        Returns:
            Regression status dict
        """
        if not baseline_path.exists():
            return {"baseline_file": None, "has_regressions": False}

        with open(baseline_path) as f:
            baseline = json.load(f)

        baseline_map = {b["id"]: b for b in baseline.get("benchmarks", [])}
        regressions = []
        improvements = []

        for result in self.results:
            if not result.success or result.id not in baseline_map:
                continue

            baseline_result = baseline_map[result.id]

            # Compare latency p95
            if result.latency and "latency_ms" in baseline_result:
                current = result.latency.p95_ms
                base = baseline_result["latency_ms"].get("p95", 0)
                if base > 0:
                    change = (current - base) / base
                    if change > latency_threshold:
                        regressions.append({
                            "benchmark_id": result.id,
                            "metric": "latency_p95",
                            "baseline_value": base,
                            "current_value": current,
                            "change_percent": round(change * 100, 1),
                        })
                    elif change < -latency_threshold:
                        improvements.append(result.id)

            # Compare memory
            if result.memory and "memory" in baseline_result:
                current = result.memory.rss_mb_peak
                base = baseline_result["memory"].get("rss_mb_peak", 0)
                if base > 0:
                    change = (current - base) / base
                    if change > memory_threshold:
                        regressions.append({
                            "benchmark_id": result.id,
                            "metric": "memory_rss_peak",
                            "baseline_value": base,
                            "current_value": current,
                            "change_percent": round(change * 100, 1),
                        })

        return {
            "baseline_file": str(baseline_path),
            "has_regressions": len(regressions) > 0,
            "regression_count": len(regressions),
            "improvement_count": len(improvements),
            "regressions": regressions,
        }

    def to_report(self) -> dict[str, Any]:
        """Generate full report."""
        return {
            "schema_version": "1.0.0",
            "timestamp": datetime.utcnow().isoformat(),
            "environment": collect_environment(),
            "model": collect_model_info(),
            "benchmarks": [r.to_dict() for r in self.results],
            "summary": self.get_summary(),
        }

    def save_report(self, path: Path) -> None:
        """Save report to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_report(), f, indent=2)


def collect_environment() -> dict[str, Any]:
    """Collect environment information."""
    env = {
        "python": platform.python_version(),
        "os": platform.system(),
        "os_version": platform.release(),
        "cpu": platform.processor() or "Unknown",
        "cpu_count": os.cpu_count() or 1,
        "ram_gb": 0,
        "torch": None,
        "cuda": None,
        "cudnn": None,
        "gpu": None,
        "gpu_count": 0,
        "vram_gb": None,
    }

    # RAM
    if HAS_PSUTIL:
        env["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)

    # PyTorch and CUDA
    if HAS_TORCH:
        env["torch"] = torch.__version__
        if torch.cuda.is_available():
            env["cuda"] = torch.version.cuda
            env["cudnn"] = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else None
            env["gpu_count"] = torch.cuda.device_count()
            if env["gpu_count"] > 0:
                env["gpu"] = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                env["vram_gb"] = round(props.total_memory / (1024**3), 1)

    return env


def collect_model_info() -> dict[str, Any]:
    """Collect model/version information."""
    info = {
        "version": "1.0.0",
        "core_model_id": "RheniumCoreModel",
        "git_sha": None,
        "git_branch": None,
        "git_dirty": False,
        "config_hash": "",
    }

    # Try to get version
    try:
        from rhenium import __version__
        info["version"] = __version__
    except ImportError:
        pass

    # Try to get git info
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info["git_sha"] = result.stdout.strip()[:12]

        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            info["git_branch"] = result.stdout.strip()

        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        info["git_dirty"] = len(result.stdout.strip()) > 0

    except Exception:
        pass

    # Config hash
    try:
        from rhenium.models import RheniumCoreModelConfig
        config = RheniumCoreModelConfig()
        config_str = str(config.__dict__)
        info["config_hash"] = hashlib.sha256(config_str.encode()).hexdigest()[:16]
    except Exception:
        pass

    return info


def run_benchmark(
    func: Callable[[], Any],
    benchmark_id: str,
    category: str,
    task: str,
    config: PerfConfig | None = None,
    **kwargs: Any,
) -> BenchmarkResult:
    """Convenience function to run a single benchmark.

    Args:
        func: Function to benchmark
        benchmark_id: Unique identifier
        category: Category name
        task: Task description
        config: Optional PerfConfig
        **kwargs: Additional arguments for run_benchmark

    Returns:
        BenchmarkResult
    """
    harness = PerfHarness(config)
    return harness.run_benchmark(func, benchmark_id, category, task, **kwargs)
