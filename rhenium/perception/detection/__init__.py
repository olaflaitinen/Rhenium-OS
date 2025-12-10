# Copyright (c) 2025 Skolyn LLC. All rights reserved.
# SPDX-License-Identifier: EUPL-1.1

"""
Detection Module
================

Base classes and interfaces for medical image detection models.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from rhenium.core.logging import get_perception_logger

logger = get_perception_logger()


@dataclass
class BoundingBox:
    """3D bounding box for detection."""
    x_min: float
    y_min: float
    z_min: float
    x_max: float
    y_max: float
    z_max: float

    @property
    def center(self) -> tuple[float, float, float]:
        return (
            (self.x_min + self.x_max) / 2,
            (self.y_min + self.y_max) / 2,
            (self.z_min + self.z_max) / 2,
        )

    @property
    def size(self) -> tuple[float, float, float]:
        return (
            self.x_max - self.x_min,
            self.y_max - self.y_min,
            self.z_max - self.z_min,
        )


@dataclass
class Detection:
    """Single detection result."""
    box: BoundingBox
    class_label: str
    confidence: float
    class_id: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DetectionResult:
    """Result from a detection model."""
    detections: list[Detection]
    image_shape: tuple[int, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    def filter_by_confidence(self, threshold: float) -> "DetectionResult":
        """Filter detections by confidence threshold."""
        filtered = [d for d in self.detections if d.confidence >= threshold]
        return DetectionResult(
            detections=filtered,
            image_shape=self.image_shape,
            metadata=self.metadata,
        )


class BaseDetectionModel(ABC):
    """Abstract base class for detection models."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = "cpu",
        confidence_threshold: float = 0.5,
    ):
        self.model_path = Path(model_path) if model_path else None
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.model = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @abstractmethod
    def load(self) -> None:
        """Load model weights."""
        pass

    @abstractmethod
    def predict(self, image: np.ndarray) -> DetectionResult:
        """Run detection inference."""
        pass

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        return image.astype(np.float32)

    def postprocess(self, detections: list[Detection]) -> list[Detection]:
        """Apply NMS and filtering."""
        return [d for d in detections if d.confidence >= self.confidence_threshold]


class DummyDetectionModel(BaseDetectionModel):
    """Placeholder detection model for testing."""

    def __init__(self, class_labels: list[str] | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.class_labels = class_labels or ["lesion"]

    def load(self) -> None:
        logger.info("Loading dummy detection model")
        self._loaded = True

    def predict(self, image: np.ndarray) -> DetectionResult:
        if not self._loaded:
            self.load()

        return DetectionResult(detections=[], image_shape=image.shape)


__all__ = [
    "BaseDetectionModel",
    "BoundingBox",
    "Detection",
    "DetectionResult",
    "DummyDetectionModel",
]
