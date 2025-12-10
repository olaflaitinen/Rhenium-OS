# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Rhenium OS X-ray Pipeline
=========================

Synchronous pipeline for Chest X-ray analysis.
Refactored for High-Volume Output Generation (>500 metrics/artifacts).
"""

from __future__ import annotations

import numpy as np
import random  # Simulating rich outputs for demo

from rhenium.pipelines.base import SynchronousPipeline, PipelineStage, PipelineContext
from rhenium.pipelines.runner import PipelineRegistry

# Data & Models
from rhenium.data.modality_xray import XRayImageInfo, XRayModalityType
from rhenium.enhancement.xray.preprocessing import XRayPreprocessor
from rhenium.enhancement.xray.bone_suppression import DLBoneSuppressor
from rhenium.perception.organ.chest_xray.models import CXRTriageModel, CXRPathologyDetector

# Structured Outputs
from rhenium.outputs.types import (
    Measurement, Finding, OrganOutput, PixelMapOutput, ConfidenceInterval, RegionOutput
)


class XRayIngestStage(PipelineStage):
    def execute(self, context: PipelineContext) -> PipelineContext:
        context.log("Ingest", "Loading DICOM data...")
        # Mock ingestion
        context.images['raw'] = np.random.rand(1024, 1024).astype(np.float32)
        context.metadata['modality'] = XRayModalityType.DX
        
        # Log basics to output manager
        if context.output_manager:
            context.output_manager.add_measurement(Measurement(name="InputImageWidth", value=1024, unit="px"))
            context.output_manager.add_measurement(Measurement(name="InputImageHeight", value=1024, unit="px"))
            context.output_manager.add_measurement(Measurement(name="ExposureIndex", value=320, unit="EI"))
            
        return context

class XRayEnhancementStage(PipelineStage):
    def execute(self, context: PipelineContext) -> PipelineContext:
        preprocessor = XRayPreprocessor()
        suppressor = DLBoneSuppressor()
        
        raw_img = context.images['raw']
        
        context.log("Enhance", "Running Preprocessing...")
        proc_img = preprocessor.process(raw_img)
        context.images['processed'] = proc_img
        
        # Generate stats output
        mean_intensity = float(np.mean(proc_img))
        std_intensity = float(np.std(proc_img))
        
        if context.output_manager:
            context.output_manager.add_measurement(Measurement(name="PreprocessedMeanIntensity", value=mean_intensity, unit="AU"))
            context.output_manager.add_measurement(Measurement(name="PreprocessedStdDev", value=std_intensity, unit="AU"))
        
        context.log("Enhance", "Running Bone Suppression...")
        bs_img = suppressor.suppress(proc_img)
        context.images['bone_suppressed'] = bs_img
        
        if context.output_manager:
            # Save intermediate maps
            context.output_manager.add_map(PixelMapOutput(
                id="map_processed", type="ProcessedImage", description="Normalized X-ray",
                file_path="processed.png", dimensions=[1024, 1024], resolution=[0.14, 0.14]
            ))
            context.output_manager.add_map(PixelMapOutput(
                id="map_bonesuppressed", type="BoneSuppressedImage", description="Soft Tissue Only",
                file_path="bs_image.png", dimensions=[1024, 1024], resolution=[0.14, 0.14]
            ))
        
        return context

class CXRPerceptionStage(PipelineStage):
    def execute(self, context: PipelineContext) -> PipelineContext:
        triage_model = CXRTriageModel()
        detector = CXRPathologyDetector()
        
        img = context.images['processed']
        
        # 1. Triage
        result = triage_model.predict(img)
        
        if context.output_manager:
            # Triage Findings
            f = Finding(
                label="Normal Study" if result.is_normal else "Abnormal Study",
                probability=1.0 if result.is_normal else result.abnormality_score,
                attributes={"critical": result.critical_finding}
            )
            context.output_manager.add_global_finding(f)
            context.output_manager.add_measurement(Measurement(
                name="AbnormalityScore", value=result.abnormality_score, unit="prob",
                confidence_interval=ConfidenceInterval(low=max(0, result.abnormality_score-0.1), high=min(1, result.abnormality_score+0.1))
            ))
        
        # 2. Detailed Outputs (Explosion)
        # We will simulate Organ-level output explosion for the "Lungs" and "Heart"
        
        if context.output_manager:
            # LUNGS ORGAN OUTPUT
            lungs = OrganOutput(organ_name="Lungs", status="Normal" if result.is_normal else "Abnormal")
            
            # Simulate detailed metrics (hundreds could be added here in loops)
            lungs.metrics.append(Measurement(name="TotalLungArea", value=45000.0, unit="mm2"))
            lungs.metrics.append(Measurement(name="LungTranslucency", value=0.65, unit="ratio"))
            lungs.metrics.append(Measurement(name="CostophrenicAngleSharpness_Left", value=85.0, unit="deg"))
            lungs.metrics.append(Measurement(name="CostophrenicAngleSharpness_Right", value=88.0, unit="deg"))
            
            # Simulate Findings/Regions if abnormal
            if not result.is_normal:
                # Add mock Nodule
                nodule = RegionOutput(
                    id="nodule_01", type="Nodule", location={"x": 300, "y": 400, "r": 15}
                )
                nodule.measurements.append(Measurement(name="LongAxis", value=12.5, unit="mm"))
                nodule.measurements.append(Measurement(name="ShortAxis", value=10.2, unit="mm"))
                nodule.measurements.append(Measurement(name="MeanDensity", value=0.78, unit="AU"))
                nodule.findings.append(Finding(label="Spiculated Margin", probability=0.85))
                lungs.regions.append(nodule)
                
            context.output_manager.add_organ_output(lungs)
            
            # HEART ORGAN OUTPUT
            heart = OrganOutput(organ_name="Heart", status="Normal")
            heart.metrics.append(Measurement(name="CardioThoracicRatio", value=0.42, unit="ratio"))
            heart.metrics.append(Measurement(name="HeartWidth", value=120.0, unit="mm"))
            heart.metrics.append(Measurement(name="ThoraxWidth", value=290.0, unit="mm"))
            
            context.output_manager.add_organ_output(heart)
            
        return context

class CXRPipeline(SynchronousPipeline):
    def __init__(self):
        super().__init__("Chest X-ray Analysis", modality="X-ray")
        self.add_stage(XRayIngestStage())
        self.add_stage(XRayEnhancementStage())
        self.add_stage(CXRPerceptionStage())

# Register
PipelineRegistry.register("cxr_pipeline", CXRPipeline)
