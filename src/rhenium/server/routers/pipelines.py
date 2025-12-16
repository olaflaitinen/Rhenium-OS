"""Pipeline execution router."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class RunRequest(BaseModel):
    pipeline_name: str
    study_uid: str


class RunResponse(BaseModel):
    job_id: str
    status: str


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: dict | None = None


@router.post("/run", response_model=RunResponse)
async def run_pipeline(req: RunRequest) -> RunResponse:
    import uuid
    return RunResponse(job_id=str(uuid.uuid4()), status="accepted")


@router.get("/job/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str) -> JobStatusResponse:
    return JobStatusResponse(job_id=job_id, status="completed", result={})
