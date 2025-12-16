"""Model Card for ML model documentation."""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any
import yaml


@dataclass
class ModelCard:
    """Structured model documentation."""
    name: str
    version: str
    description: str = ""
    task: str = ""
    architecture: str = ""
    intended_use: list[str] = field(default_factory=list)
    out_of_scope: list[str] = field(default_factory=list)
    training_data: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    limitations: list[str] = field(default_factory=list)
    ethical_considerations: str = ""
    regulatory_status: str = "Research use only"

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_details": {"name": self.name, "version": self.version, "architecture": self.architecture},
            "description": self.description,
            "intended_use": {"primary": self.intended_use, "out_of_scope": self.out_of_scope},
            "training_data": self.training_data,
            "performance": self.metrics,
            "limitations": self.limitations,
            "regulatory": {"status": self.regulatory_status},
        }

    def save(self, path: Path) -> None:
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    @classmethod
    def load(cls, path: Path) -> "ModelCard":
        with open(path) as f:
            data = yaml.safe_load(f)
        return cls(
            name=data.get("model_details", {}).get("name", ""),
            version=data.get("model_details", {}).get("version", ""),
            description=data.get("description", ""),
            architecture=data.get("model_details", {}).get("architecture", ""),
            metrics=data.get("performance", {}),
            limitations=data.get("limitations", []),
            regulatory_status=data.get("regulatory", {}).get("status", ""),
        )
