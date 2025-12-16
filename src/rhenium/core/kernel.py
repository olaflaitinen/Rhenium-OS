"""Rhenium OS Core Kernel.

The central orchestrator that manages the lifecycle of the entire system,
including resource management, component loading, and unified pipeline execution.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import structlog
import torch

from rhenium.core.config import get_settings
from rhenium.core.registry import registry
from rhenium.pipelines.runner import PipelineRunner, JobStatus
from rhenium.data.volume import ImageVolume
from rhenium.xai import EvidenceDossier

logger = structlog.get_logger()


@dataclass
class KernelStats:
    """System statistics."""
    uptime_seconds: float = 0.0
    jobs_processed: int = 0
    jobs_failed: int = 0
    active_pipelines: int = 0
    gpu_memory_used: float = 0.0


class RheniumKernel:
    """The master controller of Rhenium OS."""

    def __init__(self):
        self._settings = get_settings()
        self._pipeline_runner = PipelineRunner()
        self._active = False
        self._stats = KernelStats()
        self._logger = logger.bind(component="kernel")

    def initialize(self) -> None:
        """Initialize the kernel and all subsystems."""
        self._logger.info("system.initializing")
        
        # 1. Verify compute resources
        if self._settings.device == "cuda" and not torch.cuda.is_available():
            self._logger.warning("system.cuda_not_available_falling_back_to_cpu")
            self._settings.device = "cpu"

        # 2. Ensure directories
        self._settings.ensure_directories()
        
        # 3. Load registry components (lazy loading is default, but we can pre-warm)
        self._verify_components()

        self._active = True
        self._logger.info("system.ready", device=self._settings.device)

    def _verify_components(self) -> None:
        """Verify that essential components are registered."""
        pipelines = registry.list_pipelines()
        self._logger.info("components.verified", count=len(pipelines))

    def process_study(
        self,
        study_uid: str,
        volume: ImageVolume,
        pipelines: list[str],
        options: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Process a study through multiple unified pipelines."""
        if not self._active:
            raise RuntimeError("Kernel not initialized")

        options = options or {}
        results = {}
        
        self._logger.info("processing.start", study_uid=study_uid, pipelines=pipelines)

        for pipeline_name in pipelines:
            try:
                # Dispatch to pipeline runner
                job_id = self._pipeline_runner.run(
                    pipeline_name=pipeline_name,
                    volume=volume,
                    config=options.get(pipeline_name, {}),
                )
                
                # Check immediate result (synchronous for this core kernel method)
                job = self._pipeline_runner.get_job(job_id)
                if job and job.status == JobStatus.COMPLETED:
                    results[pipeline_name] = job.result
                    self._stats.jobs_processed += 1
                else:
                    results[pipeline_name] = {"error": job.error if job else "Unknown error"}
                    self._stats.jobs_failed += 1
                    
            except Exception as e:
                self._logger.exception("processing.failed", pipeline=pipeline_name, error=str(e))
                results[pipeline_name] = {"error": str(e)}
                self._stats.jobs_failed += 1

        self._update_stats()
        return results

    def _update_stats(self) -> None:
        """Update system statistics."""
        if torch.cuda.is_available():
            self._stats.gpu_memory_used = torch.cuda.memory_allocated() / 1024**2
    
    def shutdown(self) -> None:
        """Graceful shutdown."""
        self._logger.info("system.shutdown")
        self._active = False
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    @property
    def stats(self) -> KernelStats:
        return self._stats


# Global Kernel Instance
_kernel: RheniumKernel | None = None

def get_kernel() -> RheniumKernel:
    global _kernel
    if _kernel is None:
        _kernel = RheniumKernel()
    return _kernel
