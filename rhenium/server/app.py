# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Rhenium OS Web Server
=====================

FastAPI connector that allows external websites/systems to talk to RheniumOS.
"""

from typing import Optional
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import shutil
import os
import uuid
import json

from rhenium.core.engine import RheniumOS

app = FastAPI(
    title="Rhenium OS AI API",
    description="Unified Medical Imaging Intelligence Node",
    version="2025.12"
)

# CORS: Allow connectivity from the website frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust for production security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize Engine (Singleton-like)
engine = RheniumOS()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.get("/")
def read_root():
    return {"status": "Rhenium OS Online", "version": engine.version}


@app.get("/health")
def health_check():
    return {"status": "operational", "engine": "ready"}


@app.post("/analyze")
async def analyze_scan(
    file: UploadFile = File(...),
    modality: str = Form(...),
    patient_id: Optional[str] = Form(None),
    metadata: Optional[str] = Form(None)  # JSON string
):
    """
    Main entry point for image analysis.
    
    1. Uploads file to local storage.
    2. Calls RheniumOS core engine.
    3. Returns full Evidence Dossier.
    """
    try:
        # 1. Save File
        file_ext = file.filename.split('.')[-1] if file.filename else "tmp"
        file_id = f"{uuid.uuid4()}.{file_ext}"
        file_path = os.path.join(UPLOAD_DIR, file_id)
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 2. Parse Metadata
        meta_dict = {}
        if metadata:
            try:
                meta_dict = json.loads(metadata)
            except json.JSONDecodeError:
                pass # Or raise HTTP exception
        
        # 3. Invoke Core Engine
        result = engine.analyze(
            file_path=file_path,
            modality=modality,
            patient_id=patient_id,
            metadata=meta_dict
        )
        
        if "error" in result:
             raise HTTPException(status_code=500, detail=result["message"])
             
        # 4. Return result
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
