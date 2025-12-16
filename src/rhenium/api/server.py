"""
Rhenium OS Core API Server.

This FastAPI application exposes the RheniumCoreModel to external clients
(e.g., the Java backend).
"""

import logging
import time
import uuid
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware

from rhenium.models import RheniumCoreModel, RheniumCoreModelConfig, TaskType
from rhenium.testing.synthetic import SyntheticDataGenerator
from rhenium.api.schema import InferenceRequest, InferenceResponse

# Global Model Instance
model: RheniumCoreModel | None = None

logger = logging.getLogger("rhenium.api")
logging.basicConfig(level=logging.INFO)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize model on startup."""
    global model
    logger.info("Initializing Rhenium Core Model...")
    
    config = RheniumCoreModelConfig(
        device="cpu", # Force CPU for demo stability
        clinical_mode=True,
        seed=42
    )
    model = RheniumCoreModel(config)
    
    # Try initialization
    try:
        model.initialize()
        logger.info("Model initialized successfully.")
    except Exception as e:
        logger.error(f"Model initialization failed: {e}")
        # We don't raise here to allow API to start and report health status
        
    yield
    
    logger.info("Shutting down model...")
    if model:
        model.shutdown()

app = FastAPI(title="Rhenium OS Core API", version="1.0.0", lifespan=lifespan)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    if model and model.is_initialized:
        return {"status": "healthy", "model_version": model.VERSION}
    return {"status": "unhealthy", "reason": "Model not initialized"}

@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    """
    Run inference on the core model.
    The Java backend calls this endpoint.
    """
    global model
    if not model or not model.is_initialized:
        raise HTTPException(status_code=503, detail="Model not ready")
        
    try:
        # For this demo, we use synthetic data if requested
        # In production, we would load DICOM from study_uid/series_uid via PACS
        if request.use_synthetic:
            gen = SyntheticDataGenerator(seed=int(time.time()))
            volume = gen.generate_volume(shape=(32, 32, 32), modality="MRI")
            
            # Enrich metadata for clinical validation
            volume.metadata.study_uid = request.study_uid
            volume.metadata.series_uid = request.series_uid
            volume.metadata.patient_id = request.parameters.get("patient_id", "ANON")
            volume.metadata.patient_name = request.parameters.get("patient_name", "Anonymous")
            volume.metadata.sop_uid = str(uuid.uuid4())
            volume.metadata.modality = "MR"
            
        else:
            raise HTTPException(status_code=501, detail="Only synthetic data supported in demo mode")

        # Run model
        result = model.run(volume, request.task, **request.parameters)
        
        # Check for audit log entry in private attribute (hack for demo)
        audit_id = None
        # In a real impl, we'd return audit_id from run() or inspect logger
            
        return InferenceResponse(
            status="SUCCESS",
            task_id=str(uuid.uuid4()),
            output_shape=list(result.output.shape),
            metrics=result.metrics,
            evidence_dossier=result.evidence_dossier,
            audit_entry_id=audit_id
        )

    except ValueError as e:
        # Clinical validation error
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("Inference failed")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
