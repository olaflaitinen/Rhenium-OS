"""Unified pipelines for full-stack processing workflows.

These pipelines integrate reconstruction, perception, and XAI into single workflows.
"""

from __future__ import annotations
from typing import Any
import numpy as np
import structlog

from rhenium.core.registry import registry, ComponentType
from rhenium.pipelines.base import Pipeline, PipelineConfig
from rhenium.data.volume import ImageVolume
from rhenium.data.modality.mri import apply_bias_field_correction
from rhenium.xai import EvidenceDossier, Finding, QuantitativeEvidence

logger = structlog.get_logger()


@registry.register(ComponentType.PIPELINE, "mri_full_chain", version="1.0.0")
class MRIFullChainPipeline(Pipeline):
    """
    Unified MRI Pipeline:
    1. Preprocessing (N4 Bias Correction)
    2. Segmentation (UNet3D)
    3. Measurements
    4. XAI Dossier Generation
    """

    def run(self, volume: ImageVolume, **kwargs: Any) -> dict[str, Any]:
        logger.info("pipeline.mri.start")
        
        # 1. Preprocessing
        logger.info("pipeline.mri.step", step="preprocessing")
        processed = apply_bias_field_correction(volume)
        
        # 2. Segmentation
        # For this example we use the registered model (or a mock if simplified)
        logger.info("pipeline.mri.step", step="segmentation")
        # In a real system, we'd load the model weights here
        # For the kernel demo, we'll simulate the inference output using image intensity
        mask = (processed.array > 0.5).astype(np.uint8)  # Placeholder logic
        
        # 3. Measurements
        logger.info("pipeline.mri.step", step="measurements")
        lesion_vol = float(np.sum(mask) * np.prod(processed.spacing))
        
        # 4. Evidence Dossier
        logger.info("pipeline.mri.step", step="xai")
        # Find lesion center
        coords = np.argwhere(mask > 0)
        center = coords.mean(axis=0).tolist() if len(coords) > 0 else [0, 0, 0]
        
        finding = Finding(
            finding_id="auto_001",
            finding_type="lesion",
            description="Automated segmentation finding",
            confidence=0.88,
            location=f"Coordinates {center}",
            quantitative_evidence=[
                QuantitativeEvidence("volume", lesion_vol, "mm³")
            ]
        )
        
        dossier = EvidenceDossier(
            dossier_id="doc_auto",
            finding=finding,
            study_uid=kwargs.get("study_uid", "unknown"),
        )
        
        return {
            "mask": mask,
            "dossier": dossier.to_dict(),
            "metrics": {"volume_mm3": lesion_vol}
        }


@registry.register(ComponentType.PIPELINE, "ct_lung_analysis", version="1.0.0")
class CTLungAnalysisPipeline(Pipeline):
    """
    Unified CT Pipeline:
    1. Lung Segmentation
    2. Nodule Detection (Simulated)
    3. Reporting
    """
    
    def run(self, volume: ImageVolume, **kwargs: Any) -> dict[str, Any]:
        logger.info("pipeline.ct.start")
        
        # 1. Lung Segmentation
        from rhenium.data.modality.ct import segment_lungs
        lung_mask = segment_lungs(volume)
        
        # 2. Nodule Detection (Simulated)
        # Using simple thresholding inside lung mask for "nodules"
        nodule_candidates = (volume.array > -500) & (lung_mask > 0)
        
        # 3. Report
        num_nodules = int(nodule_candidates.sum() / 100) # Dummy count
        
        return {
            "lung_mask": lung_mask,
            "findings": {
                "nodule_count": num_nodules,
                "lung_volume_ml": lung_mask.sum() * np.prod(volume.spacing) / 1000
            }
        }
