"""Dataset Card for dataset documentation."""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml


@dataclass
class DatasetCard:
    """Structured dataset documentation."""
    name: str
    version: str
    modality: str = ""
    description: str = ""
    num_samples: int = 0
    sources: list[str] = field(default_factory=list)
    demographics: dict[str, Any] = field(default_factory=dict)
    annotation_type: str = ""
    splits: dict[str, int] = field(default_factory=dict)
    known_biases: list[str] = field(default_factory=list)
    privacy: str = "De-identified per DICOM PS3.15"

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_details": {"name": self.name, "version": self.version, "modality": self.modality},
            "description": self.description,
            "collection": {"sources": self.sources, "num_samples": self.num_samples},
            "demographics": self.demographics,
            "annotations": {"type": self.annotation_type},
            "splits": self.splits,
            "known_biases": self.known_biases,
            "privacy": self.privacy,
        }

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
