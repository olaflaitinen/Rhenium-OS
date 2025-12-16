"""Evidence Dossier for transparent AI findings."""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
import json


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class VisualEvidence:
    """Visual explanation artifact."""
    evidence_id: str
    evidence_type: str  # saliency, attention, contour
    image_path: Path | None = None
    slice_index: int | None = None
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.evidence_id, "type": self.evidence_type,
            "path": str(self.image_path) if self.image_path else None,
            "slice": self.slice_index, "description": self.description,
        }


@dataclass
class QuantitativeEvidence:
    """Numerical evidence."""
    evidence_id: str
    evidence_type: str  # volume, diameter, confidence
    value: float
    unit: str | None = None
    reference_range: tuple[float, float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.evidence_id, "type": self.evidence_type,
            "value": self.value, "unit": self.unit,
        }


@dataclass
class NarrativeEvidence:
    """Natural language explanation."""
    evidence_id: str
    explanation: str
    limitations: list[str] = field(default_factory=list)
    confidence_statement: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.evidence_id, "explanation": self.explanation,
            "limitations": self.limitations,
        }


@dataclass
class Finding:
    """A detected AI finding."""
    finding_id: str
    finding_type: str
    description: str
    confidence: float
    severity: Severity = Severity.LOW
    location: str = ""
    visual_evidence: list[VisualEvidence] = field(default_factory=list)
    quantitative_evidence: list[QuantitativeEvidence] = field(default_factory=list)
    narrative_evidence: list[NarrativeEvidence] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.finding_id, "type": self.finding_type,
            "description": self.description, "confidence": self.confidence,
            "severity": self.severity.value, "location": self.location,
            "visual": [v.to_dict() for v in self.visual_evidence],
            "quantitative": [q.to_dict() for q in self.quantitative_evidence],
            "narrative": [n.to_dict() for n in self.narrative_evidence],
        }


@dataclass
class EvidenceDossier:
    """Complete evidence package for clinical findings."""
    dossier_id: str
    finding: Finding
    study_uid: str
    series_uid: str
    pipeline_name: str
    pipeline_version: str
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict[str, Any]:
        return {
            "dossier_id": self.dossier_id,
            "finding": self.finding.to_dict(),
            "study_uid": self.study_uid,
            "pipeline": f"{self.pipeline_name}:{self.pipeline_version}",
            "created_at": self.created_at.isoformat(),
        }

    def save(self, output_dir: Path) -> Path:
        """Save dossier to directory."""
        dossier_dir = output_dir / f"dossier_{self.dossier_id}"
        dossier_dir.mkdir(parents=True, exist_ok=True)
        with open(dossier_dir / "dossier.json", "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        return dossier_dir
