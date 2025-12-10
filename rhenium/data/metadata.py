# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Metadata Management Module
==========================

Structured metadata for studies and series, PHI handling, and data lineage.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any
from uuid import uuid4


class DeidentificationStatus(str, Enum):
    """PHI de-identification status."""
    NOT_DEIDENTIFIED = "not_deidentified"
    PARTIALLY_DEIDENTIFIED = "partially_deidentified"
    FULLY_DEIDENTIFIED = "fully_deidentified"


@dataclass
class DataLineage:
    """
    Tracks data provenance and transformations.

    Attributes:
        source_id: Original data source identifier.
        transformations: List of transformations applied.
        created_at: When lineage tracking started.
    """
    source_id: str = ""
    transformations: list[dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def add_transformation(
        self,
        name: str,
        parameters: dict[str, Any] | None = None,
    ) -> None:
        """Record a data transformation."""
        self.transformations.append({
            "name": name,
            "parameters": parameters or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })

    def to_dict(self) -> dict[str, Any]:
        """Serialize lineage for storage."""
        return {
            "source_id": self.source_id,
            "transformations": self.transformations,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class StudyMetadata:
    """
    Study-level metadata with PHI controls.

    Attributes:
        study_uid: Study Instance UID.
        study_description: Description of the study.
        modalities: List of modalities in the study.
        body_parts: Body parts examined.
        deidentification_status: PHI status.
        lineage: Data provenance information.
    """
    study_uid: str = ""
    study_description: str = ""
    modalities: list[str] = field(default_factory=list)
    body_parts: list[str] = field(default_factory=list)
    institution: str = ""
    deidentification_status: DeidentificationStatus = DeidentificationStatus.NOT_DEIDENTIFIED
    lineage: DataLineage = field(default_factory=DataLineage)
    custom_fields: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata (PHI-safe)."""
        return {
            "study_uid": self.study_uid,
            "study_description": self.study_description,
            "modalities": self.modalities,
            "body_parts": self.body_parts,
            "deidentification_status": self.deidentification_status.value,
            "lineage": self.lineage.to_dict(),
        }


@dataclass
class SeriesMetadata:
    """Series-level metadata."""
    series_uid: str = ""
    series_description: str = ""
    modality: str = ""
    body_part: str = ""
    num_instances: int = 0
    manufacturer: str = ""
    protocol_name: str = ""
    lineage: DataLineage = field(default_factory=DataLineage)

    def to_dict(self) -> dict[str, Any]:
        """Serialize metadata."""
        return {
            "series_uid": self.series_uid,
            "series_description": self.series_description,
            "modality": self.modality,
            "body_part": self.body_part,
            "num_instances": self.num_instances,
            "lineage": self.lineage.to_dict(),
        }


def generate_pseudonym() -> str:
    """Generate a pseudonymous identifier for data tracking."""
    return f"RHEN-{uuid4().hex[:12].upper()}"
