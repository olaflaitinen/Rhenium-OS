"""API Schemas for Rhenium OS."""

from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class TaskType(str, Enum):
    SEGMENTATION = "segmentation"
    CLASSIFICATION = "classification"
    DETECTION = "detection"
    RECONSTRUCTION = "reconstruction"
    SUPER_RESOLUTION = "super_resolution"
    DENOISE = "denoise"
    FULL_PIPELINE = "full_pipeline"

class InferenceRequest(BaseModel):
    """Request for model inference."""
    study_uid: str = Field(..., description="DICOM Study Instance UID")
    series_uid: str = Field(..., description="DICOM Series Instance UID")
    task: TaskType = Field(..., description="Task to perform")
    
    # Input data can be passed as bas64 encoded string or referenced by ID
    # For this demo, we assume the data is uploaded separately or we use synthetic
    use_synthetic: bool = Field(False, description="Use synthetic data for demo")
    
    # Additional parameters
    parameters: Dict[str, Any] = Field(default_factory=dict)

class InferenceResponse(BaseModel):
    """Response from model inference."""
    status: str
    task_id: str
    output_shape: List[int]
    metrics: Dict[str, float]
    evidence_dossier: Optional[Dict[str, Any]] = None
    audit_entry_id: Optional[str] = None
    error: Optional[str] = None
