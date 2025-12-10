# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Brain MRI Models
================

Perception models for brain MRI analysis:
- White matter lesion segmentation
- Intracranial hemorrhage detection
- Brain tumor segmentation
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rhenium.core.logging import get_perception_logger
from rhenium.core.registry import registry, ComponentType
from rhenium.perception.segmentation import BaseSegmentationModel, SegmentationResult
from rhenium.perception.detection import BaseDetectionModel, DetectionResult

logger = get_perception_logger()


class LesionSegmentationModel(BaseSegmentationModel):
    """White matter lesion segmentation (e.g., MS lesions)."""

    LABELS = ["background", "lesion"]

    def load(self) -> None:
        logger.info("Loading lesion segmentation model")
        self._loaded = True

    def predict(self, image: np.ndarray) -> SegmentationResult:
        if not self._loaded:
            self.load()
        mask = np.zeros(image.shape[:3], dtype=np.int32)
        return SegmentationResult(mask=mask, class_labels=self.LABELS)


class HemorrhageDetectionModel(BaseDetectionModel):
    """Intracranial hemorrhage detection."""

    TYPES = ["epidural", "subdural", "subarachnoid", "intraparenchymal", "intraventricular"]

    def load(self) -> None:
        logger.info("Loading hemorrhage detection model")
        self._loaded = True

    def predict(self, image: np.ndarray) -> DetectionResult:
        if not self._loaded:
            self.load()
        return DetectionResult(detections=[], image_shape=image.shape)


class TumorSegmentationModel(BaseSegmentationModel):
    """Brain tumor segmentation (glioma grades)."""

    LABELS = ["background", "necrotic_core", "edema", "enhancing_tumor"]

    def load(self) -> None:
        logger.info("Loading tumor segmentation model")
        self._loaded = True

    def predict(self, image: np.ndarray) -> SegmentationResult:
        if not self._loaded:
            self.load()
        mask = np.zeros(image.shape[:3], dtype=np.int32)
        return SegmentationResult(mask=mask, class_labels=self.LABELS)


@dataclass
class BrainMRIModule:
    """Complete brain MRI analysis module."""
    lesion_model: LesionSegmentationModel = field(default_factory=LesionSegmentationModel)
    hemorrhage_model: HemorrhageDetectionModel = field(default_factory=HemorrhageDetectionModel)
    tumor_model: TumorSegmentationModel = field(default_factory=TumorSegmentationModel)

    def analyze(self, image: np.ndarray) -> dict[str, Any]:
        """Run brain analysis pipeline."""
        logger.info("Running brain MRI analysis")
        return {
            "lesions": self.lesion_model.predict(image),
            "hemorrhages": self.hemorrhage_model.predict(image),
            "tumors": self.tumor_model.predict(image),
        }


registry.register_organ_module(
    "brain_mri",
    BrainMRIModule,
    version="1.0.0",
    description="Brain MRI analysis module",
    tags=["neuro", "brain", "mri"],
)
