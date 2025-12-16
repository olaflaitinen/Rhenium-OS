"""Registry router."""

from fastapi import APIRouter
from rhenium.core.registry import get_registry

router = APIRouter()


@router.get("/pipelines")
async def list_pipelines() -> list[dict]:
    return get_registry().list_pipelines()


@router.get("/models")
async def list_models() -> list[dict]:
    return get_registry().list_models()
