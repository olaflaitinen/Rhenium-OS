# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Rhenium OS Ultrasound Pipeline
==============================

Synchronous pipeline for Ultrasound.
1. Beamforming / Ingest
2. Enhancement (Speckle Reduction)
3. Perception (Echo/Ob)
"""

from __future__ import annotations

import numpy as np

from rhenium.pipelines.base import SynchronousPipeline, PipelineStage, PipelineContext
from rhenium.pipelines.runner import PipelineRegistry

from rhenium.reconstruction.ultrasound.enhancement import SpeckleReducer
from rhenium.perception.organ.cardiac.echo_models import AutoEFCalculator
from rhenium.perception.organ.obstetric.models import FetalBiometryEstimator


class USIngestStage(PipelineStage):
    def execute(self, context: PipelineContext) -> PipelineContext:
        context.log("Ingest", "Loading Ultrasound frames...")
        # Mock B-mode frame
        context.images['bmode'] = np.random.rand(600, 800).astype(np.float32)
        return context

class USEnhancementStage(PipelineStage):
    def execute(self, context: PipelineContext) -> PipelineContext:
        reducer = SpeckleReducer()
        img = context.images['bmode']
        
        enhanced = reducer.filter(img)
        context.images['enhanced'] = enhanced
        context.log("Enhance", "Speckle reduction complete.")
        return context

class USPerceptionStage(PipelineStage):
    def execute(self, context: PipelineContext) -> PipelineContext:
        app = context.metadata.get('application', 'cardiac')
        img = context.images['enhanced']
        
        if app == 'cardiac':
            ef_calc = AutoEFCalculator()
            # ef = ef_calc.calculate_ef(...)
            context.log("Perception", "Calculated EF (Mock)")
            context.findings.append({"type": "EF", "value": 55.0})
            
        elif app == 'obstetric':
            bio = FetalBiometryEstimator()
            context.log("Perception", "Estimated Biometry (Mock)")
            context.findings.append({"type": "Biometry", "hc_mm": 120.0})
            
        return context

class USPipeline(SynchronousPipeline):
    def __init__(self):
        super().__init__("Ultrasound Analysis Suite")
        self.add_stage(USIngestStage())
        self.add_stage(USEnhancementStage())
        self.add_stage(USPerceptionStage())

PipelineRegistry.register("us_pipeline", USPipeline)
