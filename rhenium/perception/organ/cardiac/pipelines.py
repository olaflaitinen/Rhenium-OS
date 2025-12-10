# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Rhenium OS Cardiac MRI Pipelines
=================================

Configuration-driven pipelines for cardiac MRI analysis.



"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import (
    CardiacCineSegmenter,
    CardiacFlowAnalyzer,
    CardiacSegmentationResult,
    CardiacFlowResult,
    LGEAnalyzer,
    LGEAnalysisResult,
)


@dataclass
class CardiacPipelineConfig:
    """Configuration for cardiac MRI pipeline."""
    # Input
    cine_series_description: str = "cine"
    lge_series_description: str = "lge"
    flow_series_description: str = "flow"
    
    # Segmentation
    segment_cine: bool = True
    segment_lge: bool = True
    
    # Analysis
    analyze_flow: bool = False
    calculate_function: bool = True
    
    # LGE settings
    lge_sd_threshold: float = 5.0
    
    # Model paths
    cine_model_path: str | None = None
    lge_model_path: str | None = None


@dataclass
class CardiacPipelineResult:
    """Results from cardiac MRI pipeline."""
    cine_segmentation: CardiacSegmentationResult | None = None
    lge_analysis: LGEAnalysisResult | None = None
    flow_analysis: CardiacFlowResult | None = None
    
    # Summary metrics
    lvef: float | None = None
    rvef: float | None = None
    lv_edv_ml: float | None = None
    lv_esv_ml: float | None = None
    stroke_volume_ml: float | None = None
    scar_percentage: float | None = None


class CardiacMRIPipeline:
    """
    End-to-end cardiac MRI analysis pipeline.
    
    Workflow:
    1. Cine segmentation (LV, RV, myocardium)
    2. Functional parameter calculation (EF, volumes)
    3. LGE scar quantification (if available)
    4. Flow analysis (if phase-contrast available)
    5. XAI evidence generation
    
    Clinical Applications:
    - Heart failure assessment
    - Cardiomyopathy characterization
    - Post-infarct viability
    - Valvular disease quantification
    """
    
    def __init__(self, config: CardiacPipelineConfig | None = None):
        """Initialize pipeline."""
        self.config = config or CardiacPipelineConfig()
        
        self.cine_segmenter = CardiacCineSegmenter(
            model_path=self.config.cine_model_path
        )
        self.lge_analyzer = LGEAnalyzer(
            sd_threshold=self.config.lge_sd_threshold
        )
        self.flow_analyzer = CardiacFlowAnalyzer()
    
    def run(
        self,
        cine_volume: Any | None = None,
        lge_volume: Any | None = None,
        flow_data: dict[str, Any] | None = None,
        pixel_spacing_mm: tuple[float, float] = (1.5, 1.5),
        slice_thickness_mm: float = 8.0,
        heart_rate_bpm: float | None = None,
    ) -> CardiacPipelineResult:
        """
        Run cardiac MRI pipeline.
        
        Args:
            cine_volume: 4D cine volume (slices, phases, h, w)
            lge_volume: 3D LGE volume
            flow_data: Dict with magnitude, phase, masks
            pixel_spacing_mm: In-plane pixel spacing
            slice_thickness_mm: Slice thickness
            heart_rate_bpm: Heart rate for CO calculation
        
        Returns:
            Pipeline results with all analyses
        """
        result = CardiacPipelineResult()
        
        # Cine segmentation and function
        if cine_volume is not None and self.config.segment_cine:
            seg_result = self.cine_segmenter.segment(
                cine_volume,
                pixel_spacing_mm=pixel_spacing_mm,
                slice_thickness_mm=slice_thickness_mm,
            )
            
            if self.config.calculate_function:
                seg_result = self.cine_segmenter.calculate_function(
                    seg_result,
                    pixel_spacing_mm=pixel_spacing_mm,
                    slice_thickness_mm=slice_thickness_mm,
                    heart_rate_bpm=heart_rate_bpm,
                )
            
            result.cine_segmentation = seg_result
            result.lvef = seg_result.lv_ejection_fraction
            result.rvef = seg_result.rv_ejection_fraction
            result.lv_edv_ml = seg_result.lv_end_diastolic_volume
            result.lv_esv_ml = seg_result.lv_end_systolic_volume
            result.stroke_volume_ml = seg_result.stroke_volume
        
        # LGE analysis
        if lge_volume is not None and self.config.segment_lge:
            # Would use myocardium mask from cine if available
            myo_mask = None
            if result.cine_segmentation:
                # Extract myocardium mask from segmentation
                pass
            
            lge_result = self.lge_analyzer.analyze(
                lge_volume,
                myocardium_mask=myo_mask,
            )
            result.lge_analysis = lge_result
            result.scar_percentage = lge_result.scar_percentage
        
        return result
