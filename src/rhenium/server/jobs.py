"""Background job management for server."""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any
import uuid
import threading


class JobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    """Background job."""
    job_id: str
    pipeline_name: str
    study_uid: str
    status: JobStatus = JobStatus.PENDING
    progress: float = 0.0
    created_at: datetime = field(default_factory=datetime.utcnow)
    started_at: datetime | None = None
    completed_at: datetime | None = None
    result: dict[str, Any] | None = None
    error: str | None = None


class JobManager:
    """Thread-safe job manager."""

    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._lock = threading.Lock()

    def create_job(self, pipeline_name: str, study_uid: str) -> Job:
        """Create a new job."""
        job = Job(
            job_id=str(uuid.uuid4()),
            pipeline_name=pipeline_name,
            study_uid=study_uid,
        )
        with self._lock:
            self._jobs[job.job_id] = job
        return job

    def get_job(self, job_id: str) -> Job | None:
        """Get job by ID."""
        return self._jobs.get(job_id)

    def update_job(
        self,
        job_id: str,
        status: JobStatus | None = None,
        progress: float | None = None,
        result: dict | None = None,
        error: str | None = None,
    ) -> None:
        """Update job status."""
        with self._lock:
            job = self._jobs.get(job_id)
            if job:
                if status:
                    job.status = status
                    if status == JobStatus.RUNNING:
                        job.started_at = datetime.utcnow()
                    elif status in (JobStatus.COMPLETED, JobStatus.FAILED):
                        job.completed_at = datetime.utcnow()
                if progress is not None:
                    job.progress = progress
                if result is not None:
                    job.result = result
                if error is not None:
                    job.error = error

    def list_jobs(self, status: JobStatus | None = None) -> list[Job]:
        """List jobs, optionally filtered by status."""
        jobs = list(self._jobs.values())
        if status:
            jobs = [j for j in jobs if j.status == status]
        return jobs


# Global instance
_job_manager: JobManager | None = None


def get_job_manager() -> JobManager:
    global _job_manager
    if _job_manager is None:
        _job_manager = JobManager()
    return _job_manager
