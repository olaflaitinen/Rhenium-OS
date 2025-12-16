"""Pydantic schemas for API endpoints."""

from __future__ import annotations
from datetime import datetime
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "healthy"
    timestamp: datetime
    version: str


class IngestRequest(BaseModel):
    study_uid: str = Field(..., description="DICOM Study Instance UID")
    modality: str = Field(..., description="Imaging modality")


class IngestResponse(BaseModel):
    task_id: str
    status: str
    study_uid: str
    series_count: int


class PipelineRunRequest(BaseModel):
    pipeline_name: str = Field(..., description="Registered pipeline name")
    study_uid: str = Field(..., description="Study to process")
    config: dict = Field(default_factory=dict)


class PipelineRunResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    progress: float = 0.0
    result: dict | None = None
    error: str | None = None


class FindingSchema(BaseModel):
    finding_id: str
    finding_type: str
    description: str
    confidence: float = Field(..., ge=0, le=1)
    location: str = ""


class EvidenceDossierSchema(BaseModel):
    dossier_id: str
    study_uid: str
    finding: FindingSchema
    created_at: datetime


class ModelInfoSchema(BaseModel):
    name: str
    version: str
    model_type: str
    description: str = ""


class RegistryListResponse(BaseModel):
    models: list[ModelInfoSchema]
    pipelines: list[str]
