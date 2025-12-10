# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Rhenium OS X-ray Pipeline
=========================

Synchronous pipeline for Chest X-ray analysis.
1. Ingest DICOM
2. Preprocess (Bone Suppression)
3. Inference (Triage + Detection)
4. Explanation
"""

from __future__ import annotations

import numpy as np

from rhenium.pipelines.base import SynchronousPipeline, PipelineStage, PipelineContext
from rhenium.pipelines.runner import PipelineRegistry

# Import our modules
from rhenium.data.modality_xray import XRayImageInfo, XRayModalityType
from rhenium.enhancement.xray.preprocessing import XRayPreprocessor
from rhenium.enhancement.xray.bone_suppression import DLBoneSuppressor
from rhenium.perception.organ.chest_xray.models import CXRTriageModel, CXRPathologyDetector


class XRayIngestStage(PipelineStage):
    def execute(self, context: PipelineContext) -> PipelineContext:
        # Mock ingestion
        context.log("Ingest", "Loading DICOM data...")
        # In a real system, we'd read from context.metadata['file_path']
        # Here we mock an image
        context.images['raw'] = np.random.rand(1024, 1024).astype(np.float32)
        context.metadata['modality'] = XRayModalityType.DX
        return context

class XRayEnhancementStage(PipelineStage):
    def execute(self, context: PipelineContext) -> PipelineContext:
        preprocessor = XRayPreprocessor()
        suppressor = DLBoneSuppressor()
        
        raw_img = context.images['raw']
        
        # 1. Preprocess
        proc_img = preprocessor.process(raw_img)
        context.images['processed'] = proc_img
        context.log("Enhance", "Preprocessing complete.")
        
        # 2. Bone Suppression (Optional)
        bs_img = suppressor.suppress(proc_img)
        context.images['bone_suppressed'] = bs_img
        context.log("Enhance", "Bone suppression complete.")
        
        return context

class CXRPerceptionStage(PipelineStage):
    def execute(self, context: PipelineContext) -> PipelineContext:
        triage_model = CXRTriageModel()
        detector = CXRPathologyDetector()
        
        img = context.images['processed']
        
        # 1. Triage
        result = triage_model.predict(img)
        context.findings.append({
            "type": "Triage",
            "is_normal": result.is_normal,
            "abnormality_score": result.abnormality_score
        })
        context.log("Perception", f"Triage Result: Normal={result.is_normal}")
        
        # 2. Detection if abnormal
        if not result.is_normal:
            findings = detector.detect(img)
            context.findings.extend([{"type": "Pathology", "label": f.label} for f in findings])
            context.log("Perception", f"Detected {len(findings)} pathologies.")
            
        return context

class CXRPipeline(SynchronousPipeline):
    def __init__(self):
        super().__init__("Chest X-ray Analysis")
        self.add_stage(XRayIngestStage())
        self.add_stage(XRayEnhancementStage())
        self.add_stage(CXRPerceptionStage())

# Register
PipelineRegistry.register("cxr_pipeline", CXRPipeline)
