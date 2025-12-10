# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Segmentation Module
===================

Base classes and interfaces for medical image segmentation models.
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
class SegmentationResult:
    """Result from a segmentation model."""
    mask: np.ndarray
    class_labels: list[str]
    probabilities: np.ndarray | None = None
    uncertainty: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_classes(self) -> int:
        return len(self.class_labels)


class BaseSegmentationModel(ABC):
    """
    Abstract base class for segmentation models.

    All segmentation models in Rhenium OS should inherit from this class.
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = "cpu",
    ):
        self.model_path = Path(model_path) if model_path else None
        self.device = device
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
    def predict(self, image: np.ndarray) -> SegmentationResult:
        """Run segmentation inference."""
        pass

    def predict_with_uncertainty(
        self,
        image: np.ndarray,
        num_samples: int = 10,
    ) -> SegmentationResult:
        """
        Predict with Monte Carlo dropout for uncertainty estimation.

        Override in subclasses that support MC dropout.
        """
        return self.predict(image)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess input image. Override as needed."""
        return image.astype(np.float32)

    def postprocess(self, output: np.ndarray) -> np.ndarray:
        """Postprocess model output. Override as needed."""
        return output


class DummySegmentationModel(BaseSegmentationModel):
    """Placeholder segmentation model for testing."""

    def __init__(self, class_labels: list[str] | None = None, **kwargs: Any):
        super().__init__(**kwargs)
        self.class_labels = class_labels or ["background", "foreground"]

    def load(self) -> None:
        logger.info("Loading dummy segmentation model")
        self._loaded = True

    def predict(self, image: np.ndarray) -> SegmentationResult:
        if not self._loaded:
            self.load()

        # Generate random mask
        mask = np.zeros(image.shape[:3] if image.ndim >= 3 else image.shape, dtype=np.int32)
        return SegmentationResult(
            mask=mask,
            class_labels=self.class_labels,
        )


__all__ = [
    "BaseSegmentationModel",
    "SegmentationResult",
    "DummySegmentationModel",
]
