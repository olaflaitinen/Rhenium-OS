# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Model Cards
===========

Standardized documentation for perception models.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from typing import Any
import json


@dataclass
class ModelCard:
    """Model card documentation."""
    model_name: str
    model_version: str
    task: str  # segmentation, detection, classification
    modality: str
    body_part: str
    intended_use: str = ""
    intended_users: list[str] = field(default_factory=list)
    out_of_scope_uses: list[str] = field(default_factory=list)
    training_data: str = ""
    training_data_size: int = 0
    evaluation_data: str = ""
    metrics: dict[str, float] = field(default_factory=dict)
    known_limitations: list[str] = field(default_factory=list)
    ethical_considerations: list[str] = field(default_factory=list)
    caveats: list[str] = field(default_factory=list)
    last_updated: date = field(default_factory=date.today)
    authors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_version": self.model_version,
            "task": self.task,
            "modality": self.modality,
            "body_part": self.body_part,
            "intended_use": self.intended_use,
            "intended_users": self.intended_users,
            "training_data": self.training_data,
            "metrics": self.metrics,
            "known_limitations": self.known_limitations,
            "last_updated": self.last_updated.isoformat(),
        }

    def to_markdown(self) -> str:
        """Generate markdown model card."""
        lines = [
            f"# Model Card: {self.model_name}",
            f"\n**Version**: {self.model_version}",
            f"\n## Intended Use\n{self.intended_use}",
            f"\n## Training Data\n{self.training_data}",
            "\n## Performance Metrics\n",
        ]
        for metric, value in self.metrics.items():
            lines.append(f"- {metric}: {value}")
        lines.append("\n## Known Limitations\n")
        for lim in self.known_limitations:
            lines.append(f"- {lim}")
        return "\n".join(lines)
