"""Ingest router."""

from fastapi import APIRouter, UploadFile
from pydantic import BaseModel

router = APIRouter()


class IngestResponse(BaseModel):
    status: str
    study_uid: str
    series_count: int


@router.post("/dicom", response_model=IngestResponse)
async def ingest_dicom(file: UploadFile) -> IngestResponse:
    return IngestResponse(status="accepted", study_uid="1.2.3.4.5", series_count=1)
