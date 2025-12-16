"""Pipeline runner and job management."""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
import uuid
from rhenium.core.registry import registry
from rhenium.pipelines.base import Pipeline, PipelineConfig
from rhenium.data.volume import ImageVolume


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Job:
    """Pipeline execution job."""
    job_id: str
    pipeline_name: str
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class PipelineRunner:
    """Runs registered pipelines."""

    def __init__(self):
        self._jobs: dict[str, Job] = {}

    def run(
        self,
        pipeline_name: str,
        volume: ImageVolume,
        **kwargs: Any,
    ) -> str:
        """Run pipeline and return job ID."""
        job_id = str(uuid.uuid4())
        job = Job(job_id=job_id, pipeline_name=pipeline_name)
        self._jobs[job_id] = job

        try:
            job.status = JobStatus.RUNNING
            pipeline_cls = registry.get("pipeline", pipeline_name)
            config = PipelineConfig(name=pipeline_name, **kwargs.get("config", {}))
            pipeline = pipeline_cls(config)

            processed = pipeline.preprocess(volume)
            result = pipeline.run(processed, **kwargs)
            job.result = pipeline.postprocess(result)
            job.status = JobStatus.COMPLETED
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
        finally:
            job.completed_at = datetime.utcnow()

        return job_id

    def get_job(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    def list_jobs(self) -> list[Job]:
        return list(self._jobs.values())
