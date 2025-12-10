# Copyright (c) 2025 Skolyn LLC. All rights reserved.
# SPDX-License-Identifier: EUPL-1.1

"""
Classification Module
=====================

Base classes and interfaces for medical image classification models.
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
class ClassificationResult:
    """Result from a classification model."""
    predicted_class: str
    predicted_class_id: int
    probabilities: dict[str, float]
    confidence: float
    uncertainty: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def top_k(self) -> list[tuple[str, float]]:
        """Get top classes sorted by probability."""
        return sorted(self.probabilities.items(), key=lambda x: x[1], reverse=True)


class BaseClassificationModel(ABC):
    """Abstract base class for classification models."""

    def __init__(
        self,
        model_path: str | Path | None = None,
        device: str = "cpu",
        class_labels: list[str] | None = None,
    ):
        self.model_path = Path(model_path) if model_path else None
        self.device = device
        self.class_labels = class_labels or []
        self.model = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    @property
    def num_classes(self) -> int:
        return len(self.class_labels)

    @abstractmethod
    def load(self) -> None:
        """Load model weights."""
        pass

    @abstractmethod
    def predict(self, image: np.ndarray) -> ClassificationResult:
        """Run classification inference."""
        pass

    def predict_with_uncertainty(
        self,
        image: np.ndarray,
        num_samples: int = 10,
    ) -> ClassificationResult:
        """Predict with MC dropout for uncertainty."""
        return self.predict(image)

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        return image.astype(np.float32)


class DummyClassificationModel(BaseClassificationModel):
    """Placeholder classification model for testing."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        if not self.class_labels:
            self.class_labels = ["normal", "abnormal"]

    def load(self) -> None:
        logger.info("Loading dummy classification model")
        self._loaded = True

    def predict(self, image: np.ndarray) -> ClassificationResult:
        if not self._loaded:
            self.load()

        probs = {label: 1.0 / len(self.class_labels) for label in self.class_labels}
        return ClassificationResult(
            predicted_class=self.class_labels[0],
            predicted_class_id=0,
            probabilities=probs,
            confidence=probs[self.class_labels[0]],
        )


__all__ = [
    "BaseClassificationModel",
    "ClassificationResult",
    "DummyClassificationModel",
]
