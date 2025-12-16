"""FastAPI application for Rhenium OS."""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from rhenium.server.routers import health, ingest, pipelines, registry

app = FastAPI(
    title="Rhenium OS API",
    description="Multi-Modality AI Platform for Medical Imaging Research",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["Health"])
app.include_router(ingest.router, prefix="/v1/ingest", tags=["Ingest"])
app.include_router(pipelines.router, prefix="/v1/pipeline", tags=["Pipeline"])
app.include_router(registry.router, prefix="/v1/registry", tags=["Registry"])


@app.get("/")
async def root() -> dict:
    return {"message": "Rhenium OS API", "version": "0.1.0"}
