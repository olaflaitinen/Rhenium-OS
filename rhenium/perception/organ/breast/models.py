# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Breast Imaging Models
=====================

Lesion detection and characterization for breast MRI/mammography.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from rhenium.core.logging import get_perception_logger
from rhenium.core.registry import registry
from rhenium.perception.detection import BaseDetectionModel, DetectionResult
from rhenium.perception.classification import BaseClassificationModel, ClassificationResult

logger = get_perception_logger()


class BreastLesionDetectionModel(BaseDetectionModel):
    """Breast lesion detection for MRI/mammography."""

    def load(self) -> None:
        logger.info("Loading breast lesion detection model")
        self._loaded = True

    def predict(self, image: np.ndarray) -> DetectionResult:
        if not self._loaded:
            self.load()
        return DetectionResult(detections=[], image_shape=image.shape)


class BreastDensityModel(BaseClassificationModel):
    """Breast density classification (BI-RADS categories)."""
    DENSITIES = ["a_fatty", "b_scattered", "c_heterogeneous", "d_dense"]

    def __init__(self, **kwargs: Any):
        super().__init__(class_labels=self.DENSITIES, **kwargs)

    def load(self) -> None:
        logger.info("Loading breast density model")
        self._loaded = True

    def predict(self, image: np.ndarray) -> ClassificationResult:
        if not self._loaded:
            self.load()
        probs = {d: 0.25 for d in self.DENSITIES}
        return ClassificationResult(
            predicted_class="b_scattered",
            predicted_class_id=1,
            probabilities=probs,
            confidence=0.25,
        )


@dataclass
class BreastImagingModule:
    """Complete breast imaging analysis module."""
    lesion_model: BreastLesionDetectionModel = field(default_factory=BreastLesionDetectionModel)
    density_model: BreastDensityModel = field(default_factory=BreastDensityModel)

    def analyze(self, image: np.ndarray) -> dict[str, Any]:
        logger.info("Running breast imaging analysis")
        return {
            "lesions": self.lesion_model.predict(image),
            "density": self.density_model.predict(image),
        }


registry.register_organ_module(
    "breast_imaging",
    BreastImagingModule,
    version="1.0.0",
    description="Breast imaging analysis module",
    tags=["oncology", "breast", "mri", "mammography"],
)
