"""Pipeline base classes."""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any
from rhenium.data.volume import ImageVolume


@dataclass
class PipelineConfig:
    """Pipeline configuration."""
    name: str
    version: str = "1.0.0"
    modality: str | None = None
    device: str = "cuda"
    params: dict[str, Any] = field(default_factory=dict)


class Pipeline(ABC):
    """Base class for processing pipelines."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.name = config.name
        self.version = config.version

    @abstractmethod
    def run(self, volume: ImageVolume, **kwargs: Any) -> dict[str, Any]:
        """Execute pipeline on input volume."""
        pass

    def preprocess(self, volume: ImageVolume) -> ImageVolume:
        """Preprocessing hook."""
        return volume

    def postprocess(self, result: dict[str, Any]) -> dict[str, Any]:
        """Postprocessing hook."""
        return result
