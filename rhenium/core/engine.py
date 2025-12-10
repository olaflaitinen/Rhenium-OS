# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Rhenium OS Core Engine
======================

The Unified Facade Class `RheniumOS`.
This is the single entry point for all high-level system interactions,
connecting the data layer, pipelines, and output manager.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional
from pathlib import Path
import uuid
import time

from rhenium.pipelines.runner import run_pipeline, PipelineRegistry
# Import pipelines to ensure registration
import rhenium.pipelines.xray_pipeline
import rhenium.pipelines.ct_pipeline
import rhenium.pipelines.us_pipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("RheniumOS")

class RheniumOS:
    """
    The Core Intelligence Engine.
    
    Usage:
        engine = RheniumOS()
        result = engine.analyze(file_path="scan.dcm", modality="CT")
    """
    
    def __init__(self, storage_root: str = "outputs"):
        self.storage_root = Path(storage_root)
        self.storage_root.mkdir(parents=True, exist_ok=True)
        self.version = "2025.12.10-RC1"
        logger.info(f"Rhenium OS Engine v{self.version} Initialized.")
        
    def analyze(
        self,
        file_path: str,
        modality: str,
        patient_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze a medical case using the appropriate SoTA pipeline.
        
        Args:
            file_path: Path to the input image/volume.
            modality: 'X-ray', 'CT', 'Ultrasound', or 'MRI'.
            patient_id: ID of the patient (optional, generated if None).
            metadata: Additional config/clinical context (e.g. {'region': 'chest'}).
            
        Returns:
            Dict: The serialized Evidence Dossier containing findings, metrics, and map paths.
        """
        start_time = time.time()
        pid = patient_id or f"anon-{uuid.uuid4().hex[:8]}"
        meta = metadata or {}
        
        logger.info(f"Received request: Modality={modality}, Patient={pid}, File={file_path}")
        
        # 1. Pipeline Routing
        pipeline_key = self._resolve_pipeline(modality, meta)
        if not pipeline_key:
            raise ValueError(f"Unsupported modality or configuration: {modality} / {meta}")
            
        # 2. Execution
        try:
            # We mock the loading of 'file_path' inside the pipelines for now (demo mode)
            # In production, file_path would be passed to 'IngestStage'
            logger.info(f"Routing to pipeline: {pipeline_key}")
            
            context = run_pipeline(
                pipeline_key,
                patient_id=pid,
                study_uid=f"study-{uuid.uuid4().hex[:8]}",
                metadata=meta
            )
            
            # 3. Output Retrieval
            if context.output_manager:
                dossier = context.output_manager.dossier.to_dict()
                duration = time.time() - start_time
                dossier['system_meta'] = {
                    'engine_version': self.version,
                    'execution_duration_sec': round(duration, 3),
                    'pipeline_used': pipeline_key
                }
                logger.info(f"Analysis complete in {duration:.2f}s. Dossier ID: {dossier.get('case_id')}")
                return dossier
            else:
                raise RuntimeError("Pipeline completed but produced no output manager context.")
                
        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            return {
                "error": True,
                "message": str(e),
                "modality": modality,
                "timestamp": time.time()
            }

    def _resolve_pipeline(self, modality: str, metadata: dict) -> str | None:
        """Map modality string to registered pipeline key."""
        m = modality.lower()
        if "x-ray" in m or "xray" in m:
            return "cxr_pipeline"
        elif "ct" in m or "computed tomography" in m:
            return "ct_pipeline" # supports region='abdomen' via metadata
        elif "ultrasound" in m or "us" in m:
            return "us_pipeline"
        elif "mri" in m:
            # Assuming an mri_pipeline exists or will exist. 
            # If not yet registered in demo, default to None or throw.
            # We haven't fully refactored MRI pipeline for new output manager in previous turn,
            # but let's assume 'us_pipeline' logic for now or add 'mri_pipeline' if registry has it.
            # Safe fallback: None if not ready.
            return None 
            
        return None
