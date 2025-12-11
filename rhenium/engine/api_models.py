# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Dashboard Engine API Models
===========================

Pydantic-style request/response models for dashboard integration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class EngineStatus(str, Enum):
    """Dashboard engine operational status."""
    INITIALIZING = "initializing"
    READY = "ready"
    BUSY = "busy"
    ERROR = "error"
    SHUTDOWN = "shutdown"


@dataclass
class StudyInfo:
    """Information about an imaging study."""
    study_id: str
    patient_id: str = ""
    modality: str = ""
    body_part: str = ""
    study_date: Optional[str] = None
    series_count: int = 0
    instance_count: int = 0
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_id": self.study_id,
            "patient_id": self.patient_id,
            "modality": self.modality,
            "body_part": self.body_part,
            "study_date": self.study_date,
            "series_count": self.series_count,
            "instance_count": self.instance_count,
            "description": self.description,
        }


@dataclass
class PipelineInfo:
    """Information about an available pipeline."""
    pipeline_id: str
    name: str
    version: str
    modality: str
    body_part: str
    description: str = ""
    enabled: bool = True
    models: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "pipeline_id": self.pipeline_id,
            "name": self.name,
            "version": self.version,
            "modality": self.modality,
            "body_part": self.body_part,
            "description": self.description,
            "enabled": self.enabled,
            "models": self.models,
        }


@dataclass
class AnalysisRequest:
    """Request for study analysis."""
    study_id: str
    pipeline_id: str = ""
    priority: str = "normal"
    options: dict[str, Any] = field(default_factory=dict)
    clinical_context: dict[str, Any] = field(default_factory=dict)
    callback_url: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "study_id": self.study_id,
            "pipeline_id": self.pipeline_id,
            "priority": self.priority,
            "options": self.options,
            "clinical_context": self.clinical_context,
            "callback_url": self.callback_url,
        }


@dataclass
class AnalysisResponse:
    """Response from study analysis."""
    request_id: str
    study_id: str
    status: str
    pipeline_id: str
    findings_count: int = 0
    has_critical_findings: bool = False
    processing_time_ms: float = 0.0
    disease_output: Optional[dict[str, Any]] = None
    evidence_dossier_id: Optional[str] = None
    report_draft_id: Optional[str] = None
    error_message: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "study_id": self.study_id,
            "status": self.status,
            "pipeline_id": self.pipeline_id,
            "findings_count": self.findings_count,
            "has_critical_findings": self.has_critical_findings,
            "processing_time_ms": self.processing_time_ms,
            "disease_output": self.disease_output,
            "evidence_dossier_id": self.evidence_dossier_id,
            "report_draft_id": self.report_draft_id,
            "error_message": self.error_message,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class HealthStatus:
    """System health status."""
    status: EngineStatus
    version: str
    uptime_seconds: float = 0.0
    pipelines_loaded: int = 0
    models_loaded: int = 0
    gpu_available: bool = False
    gpu_memory_used_mb: float = 0.0
    gpu_memory_total_mb: float = 0.0
    pending_requests: int = 0
    completed_requests: int = 0
    error_count: int = 0
    last_error: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "version": self.version,
            "uptime_seconds": self.uptime_seconds,
            "pipelines_loaded": self.pipelines_loaded,
            "models_loaded": self.models_loaded,
            "gpu_available": self.gpu_available,
            "gpu_memory_used_mb": self.gpu_memory_used_mb,
            "gpu_memory_total_mb": self.gpu_memory_total_mb,
            "pending_requests": self.pending_requests,
            "completed_requests": self.completed_requests,
            "error_count": self.error_count,
            "last_error": self.last_error,
        }
