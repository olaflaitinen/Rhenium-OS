# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Base Pipeline
=============

Abstract pipeline class for end-to-end medical imaging analysis.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from rhenium.core.logging import get_pipeline_logger
from rhenium.core.disease_types import DiseaseReasoningOutput
from rhenium.data.dicom_io import DICOMStudy, DICOMSeries, ImageVolume
from rhenium.xai.explanation_schema import Finding
from rhenium.xai.evidence_dossier import EvidenceDossier

logger = get_pipeline_logger()


@dataclass
class PipelineResult:
    """Result from a pipeline execution."""
    run_id: str = field(default_factory=lambda: uuid4().hex[:12])
    pipeline_name: str = ""
    pipeline_version: str = ""
    status: str = "success"  # success, failed, partial
    findings: list[Finding] = field(default_factory=list)
    dossiers: list[EvidenceDossier] = field(default_factory=list)
    disease_output: Optional[DiseaseReasoningOutput] = None
    report_draft: Any = None
    execution_time_seconds: float = 0.0
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: datetime | None = None
    errors: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize result."""
        return {
            "run_id": self.run_id,
            "pipeline_name": self.pipeline_name,
            "pipeline_version": self.pipeline_version,
            "status": self.status,
            "num_findings": len(self.findings),
            "findings": [f.to_dict() for f in self.findings],
            "disease_output": self.disease_output.to_dict() if self.disease_output else None,
            "execution_time_seconds": self.execution_time_seconds,
            "started_at": self.started_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "errors": self.errors,
        }

    @property
    def has_disease(self) -> bool:
        """Check if disease was detected."""
        return self.disease_output is not None and self.disease_output.has_disease

    @property
    def has_critical_findings(self) -> bool:
        """Check for critical findings requiring escalation."""
        return self.disease_output is not None and self.disease_output.has_critical_flags


class BasePipeline(ABC):
    """
    Abstract base class for analysis pipelines.

    Pipelines orchestrate the full workflow:
    1. Load input data
    2. Run reconstruction (if applicable)
    3. Run perception models
    4. Run disease reasoning
    5. Generate XAI artifacts
    6. Generate MedGemma explanations
    7. Assemble results
    """

    name: str = "base_pipeline"
    version: str = "1.0.0"

    def __init__(self, config: dict[str, Any] | None = None):
        self.config = config or {}
        self._result: PipelineResult | None = None

    @abstractmethod
    def load_input(self, source: DICOMStudy | DICOMSeries | ImageVolume | Path) -> None:
        """Load input data."""
        pass

    @abstractmethod
    def run_reconstruction(self) -> None:
        """Run reconstruction step (if applicable)."""
        pass

    @abstractmethod
    def run_analysis(self) -> None:
        """Run perception models."""
        pass

    def run_disease_reasoning(self) -> Optional[DiseaseReasoningOutput]:
        """Run disease reasoning (optional, override in subclasses)."""
        return None

    @abstractmethod
    def run_xai(self) -> None:
        """Generate XAI artifacts."""
        pass

    @abstractmethod
    def run_medgemma_explanation(self) -> None:
        """Generate MedGemma explanations."""
        pass

    @abstractmethod
    def assemble_results(self) -> PipelineResult:
        """Assemble final results."""
        pass

    def run(self, source: DICOMStudy | DICOMSeries | ImageVolume | Path) -> PipelineResult:
        """Execute full pipeline."""
        import time
        start_time = time.time()

        logger.info("Starting pipeline", name=self.name, version=self.version)

        result = PipelineResult(
            pipeline_name=self.name,
            pipeline_version=self.version,
        )

        try:
            self.load_input(source)
            self.run_reconstruction()
            self.run_analysis()
            self._disease_output = self.run_disease_reasoning()
            self.run_xai()
            self.run_medgemma_explanation()
            result = self.assemble_results()
            result.status = "success"

        except Exception as e:
            logger.error("Pipeline failed", error=str(e))
            result.status = "failed"
            result.errors.append(str(e))

        result.completed_at = datetime.now(timezone.utc)
        result.execution_time_seconds = time.time() - start_time

        logger.info(
            "Pipeline complete",
            status=result.status,
            duration=f"{result.execution_time_seconds:.2f}s",
            num_findings=len(result.findings),
        )

        return result
