# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Rhenium OS Ultrasound Pipeline
==============================

Synchronous pipeline for Ultrasound.
Refactored for High-Volume Output Generation.
"""

from __future__ import annotations

import numpy as np
import random 

from rhenium.pipelines.base import SynchronousPipeline, PipelineStage, PipelineContext
from rhenium.pipelines.runner import PipelineRegistry

from rhenium.reconstruction.ultrasound.enhancement import SpeckleReducer
from rhenium.perception.organ.cardiac.echo_models import AutoEFCalculator
from rhenium.perception.organ.obstetric.models import FetalBiometryEstimator
from rhenium.outputs.types import Measurement, Finding, OrganOutput, PixelMapOutput, RegionOutput


class USIngestStage(PipelineStage):
    def execute(self, context: PipelineContext) -> PipelineContext:
        context.log("Ingest", "Loading Ultrasound frames...")
        # Mock B-mode frame
        context.images['bmode'] = np.random.rand(600, 800).astype(np.float32)
        
        if context.output_manager:
            context.output_manager.add_measurement(Measurement(name="FrameRate", value=60, unit="Hz"))
            context.output_manager.add_measurement(Measurement(name="MechanicalIndex", value=0.9, unit="MI"))
            
        return context

class USEnhancementStage(PipelineStage):
    def execute(self, context: PipelineContext) -> PipelineContext:
        reducer = SpeckleReducer()
        img = context.images['bmode']
        
        enhanced = reducer.filter(img)
        context.images['enhanced'] = enhanced
        
        if context.output_manager:
              context.output_manager.add_map(PixelMapOutput(
                id="us_enhanced", type="EnhancedBMode", description="Despeckled Ultrasound",
                file_path="us_enhanced.png", dimensions=[800, 600], resolution=[0.2, 0.2]
            ))
            
        context.log("Enhance", "Speckle reduction complete.")
        return context

class USPerceptionStage(PipelineStage):
    def execute(self, context: PipelineContext) -> PipelineContext:
        app = context.metadata.get('application', 'cardiac')
        
        if app == 'cardiac':
            # Simulated Echo Analysis
            heart = OrganOutput(organ_name="Heart", status="Abnormal")
            
            # LV Metrics
            ef = 55.0 + random.random() * 5.0
            heart.metrics.append(Measurement(name="LVEF_Auto", value=ef, unit="percent"))
            heart.metrics.append(Measurement(name="LVEDV", value=120.0, unit="ml"))
            heart.metrics.append(Measurement(name="LVESV", value=55.0, unit="ml"))
            heart.metrics.append(Measurement(name="GlobalLongitudinalStrain", value=-18.5, unit="percent"))
            
            # Doppler simulation
            heart.metrics.append(Measurement(name="E_Velocity", value=0.8, unit="m/s"))
            heart.metrics.append(Measurement(name="A_Velocity", value=0.6, unit="m/s"))
            heart.metrics.append(Measurement(name="EA_Ratio", value=1.33, unit="ratio"))
            
            if context.output_manager:
                context.output_manager.add_organ_output(heart)
                context.output_manager.add_global_finding(Finding(label="Normal LV Function", probability=0.98))
            
        elif app == 'obstetric':
            fetus = OrganOutput(organ_name="Fetus", status="Normal")
            fetus.metrics.append(Measurement(name="BPD", value=45.0, unit="mm"))
            fetus.metrics.append(Measurement(name="HC", value=160.0, unit="mm"))
            fetus.metrics.append(Measurement(name="AC", value=140.0, unit="mm"))
            fetus.metrics.append(Measurement(name="FL", value=30.0, unit="mm"))
            fetus.metrics.append(Measurement(name="EstimatedFetalWeight", value=800.0, unit="g"))
            fetus.metrics.append(Measurement(name="GestationalAge", value=24.0, unit="weeks"))
            
            if context.output_manager:
                context.output_manager.add_organ_output(fetus)

        return context

class USPipeline(SynchronousPipeline):
    def __init__(self):
        super().__init__("Ultrasound Analysis Suite", modality="Ultrasound")
        self.add_stage(USIngestStage())
        self.add_stage(USEnhancementStage())
        self.add_stage(USPerceptionStage())

PipelineRegistry.register("us_pipeline", USPipeline)
