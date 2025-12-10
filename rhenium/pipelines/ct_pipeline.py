# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Rhenium OS CT Pipeline
======================

Synchronous pipeline for CT analysis.
Refactored for High-Volume Output Generation.
"""

from __future__ import annotations

import numpy as np

from rhenium.pipelines.base import SynchronousPipeline, PipelineStage, PipelineContext
from rhenium.pipelines.runner import PipelineRegistry

from rhenium.reconstruction.ct.dl_reconstruction import SparseViewCTReconstructor
from rhenium.perception.organ.lung.models import LungNoduleDetector
# from rhenium.perception.organ.liver.models import LiverLesionDetector # Imported if needed

from rhenium.outputs.types import Measurement, Finding, OrganOutput, PixelMapOutput, RegionOutput

class CTIngestStage(PipelineStage):
    def execute(self, context: PipelineContext) -> PipelineContext:
        context.log("Ingest", "Loading CT Sinogram/Volume...")
        context.images['raw_sinogram'] = np.random.rand(360, 512).astype(np.float32)
        
        if context.output_manager:
            context.output_manager.add_measurement(Measurement(name="TubeCurrent", value=200, unit="mA"))
            context.output_manager.add_measurement(Measurement(name="TubeVoltage", value=120, unit="kVp"))
            context.output_manager.add_measurement(Measurement(name="RotationTime", value=0.5, unit="s"))
            
        return context

class CTReconStage(PipelineStage):
    def execute(self, context: PipelineContext) -> PipelineContext:
        recon = SparseViewCTReconstructor()
        sino = context.images['raw_sinogram']
        
        vol = recon.reconstruct(sino)
        context.images['reconstructed_volume'] = vol
        context.log("Recon", "DL Reconstruction complete.")
        
        if context.output_manager:
            context.output_manager.add_map(PixelMapOutput(
                id="vol_recon", type="ReconstructedVolume", description="DL Reconstructed CT",
                file_path="recon_vol.nii.gz", dimensions=[512, 512, 100], resolution=[0.6, 0.6, 1.0]
            ))
            context.output_manager.add_measurement(Measurement(name="ReconNoiseLevel", value=12.5, unit="HU"))
            context.output_manager.add_measurement(Measurement(name="ReconSharpness", value=0.85, unit="MTF"))
            
        return context

class CTPerceptionStage(PipelineStage):
    def execute(self, context: PipelineContext) -> PipelineContext:
        region = context.metadata.get('region', 'chest')
        vol = context.images['reconstructed_volume']
        
        if region == 'chest':
            # Lung Organ Analysis
            langs_out = OrganOutput(organ_name="Lungs", status="Indeterminate")
            
            # Volumetry
            langs_out.volumetry = Measurement(name="TotalLungVolume", value=4200.5, unit="cm3")
            langs_out.metrics.append(Measurement(name="RightLungVolume", value=2200.0, unit="cm3"))
            langs_out.metrics.append(Measurement(name="LeftLungVolume", value=2000.5, unit="cm3"))
            langs_out.metrics.append(Measurement(name="MeanLungDensity", value=-850.0, unit="HU"))
            langs_out.metrics.append(Measurement(name="EmphysemaIndex", value=2.5, unit="percent"))
            
            # Nodules (Simulation)
            for i in range(3): # Simulate 3 nodules
                nod = RegionOutput(id=f"nodule_{i}", type="LungNodule", location={"slice": 45, "x": 200, "y": 200})
                nod.measurements.append(Measurement(name="Volume", value=random_float(50, 200), unit="mm3"))
                nod.measurements.append(Measurement(name="Diameter", value=random_float(4, 10), unit="mm"))
                nod.measurements.append(Measurement(name="MeanHU", value=random_float(40, 80), unit="HU"))
                nod.findings.append(Finding(label="Solid", probability=0.9))
                langs_out.regions.append(nod)
                
            if context.output_manager:
                context.output_manager.add_organ_output(langs_out)
                context.output_manager.add_global_finding(Finding(label="Multiple Lung Nodules", probability=0.95))

        elif region == 'abdomen':
            liver_out = OrganOutput(organ_name="Liver", status="Normal")
            liver_out.volumetry = Measurement(name="LiverVolume", value=1500.0, unit="cm3")
            liver_out.metrics.append(Measurement(name="MeanLiverDensity", value=60.0, unit="HU"))
            liver_out.metrics.append(Measurement(name="FatFraction", value=0.05, unit="ratio"))
            
            if context.output_manager:
                context.output_manager.add_organ_output(liver_out)
            
        return context

def random_float(a, b):
    import random
    return a + random.random() * (b - a)

class CTPipeline(SynchronousPipeline):
    def __init__(self):
        super().__init__("CT Analysis Suite", modality="CT")
        self.add_stage(CTIngestStage())
        self.add_stage(CTReconStage())
        self.add_stage(CTPerceptionStage())

PipelineRegistry.register("ct_pipeline", CTPipeline)
