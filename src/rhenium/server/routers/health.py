"""Health check router."""

from datetime import datetime
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str


@router.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    import rhenium
    return HealthResponse(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        version=rhenium.__version__,
    )
