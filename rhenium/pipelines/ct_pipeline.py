# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Rhenium OS CT Pipeline
======================

Synchronous pipeline for CT analysis.
1. Ingest Data
2. Reconstruct (DL Recon)
3. Perception (Lung/Liver etc.)
"""

from __future__ import annotations

import numpy as np

from rhenium.pipelines.base import SynchronousPipeline, PipelineStage, PipelineContext
from rhenium.pipelines.runner import PipelineRegistry

from rhenium.reconstruction.ct.dl_reconstruction import SparseViewCTReconstructor
from rhenium.perception.organ.lung.models import LungNoduleDetector
from rhenium.perception.organ.liver.models import LiverLesionDetector


class CTIngestStage(PipelineStage):
    def execute(self, context: PipelineContext) -> PipelineContext:
        context.log("Ingest", "Loading CT Sinogram/Volume...")
        # Mock 512x512 volume
        context.images['raw_sinogram'] = np.random.rand(360, 512).astype(np.float32)
        return context

class CTReconStage(PipelineStage):
    def execute(self, context: PipelineContext) -> PipelineContext:
        recon = SparseViewCTReconstructor()
        sino = context.images['raw_sinogram']
        
        vol = recon.reconstruct(sino)
        context.images['reconstructed_volume'] = vol
        context.log("Recon", "DL Reconstruction complete.")
        return context

class CTPerceptionStage(PipelineStage):
    def execute(self, context: PipelineContext) -> PipelineContext:
        # Mock perception based on requested region
        region = context.metadata.get('region', 'chest')
        
        vol = context.images['reconstructed_volume']
        
        if region == 'chest':
            detector = LungNoduleDetector()
            # Mock behavior
            context.log("Perception", "Running Lung Nodule Detection...")
            # findings = detector.detect(vol)
            context.findings.append({"type": "Nodule", "count": 0})
            
        elif region == 'abdomen':
            detector = LiverLesionDetector()
            context.log("Perception", "Running Liver Lesion Detection...")
            # findings = detector.detect(vol)
            context.findings.append({"type": "LiverLesion", "count": 0})
            
        return context

class CTPipeline(SynchronousPipeline):
    def __init__(self):
        super().__init__("CT Analysis Suite")
        self.add_stage(CTIngestStage())
        self.add_stage(CTReconStage())
        self.add_stage(CTPerceptionStage())

PipelineRegistry.register("ct_pipeline", CTPipeline)
