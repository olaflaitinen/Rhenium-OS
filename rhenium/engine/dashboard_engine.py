# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Rhenium OS Dashboard Engine
===========================

Core engine class providing unified API for dashboard integration.
This is the primary interface for external applications to interact
with Rhenium OS capabilities.

Architecture:
    Dashboard → DashboardEngine → Pipelines → Models → Results
                     ↓
              Evidence Dossiers
                     ↓
              MedGemma Reports
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

from rhenium.engine.api_models import (
    AnalysisRequest,
    AnalysisResponse,
    EngineStatus,
    HealthStatus,
    PipelineInfo,
    StudyInfo,
)
from rhenium.core.disease_types import DiseaseReasoningOutput


@dataclass
class EngineConfig:
    """Configuration for dashboard engine."""
    data_dir: Path = field(default_factory=lambda: Path("./data"))
    models_dir: Path = field(default_factory=lambda: Path("./models"))
    cache_dir: Path = field(default_factory=lambda: Path("./cache"))
    log_level: str = "INFO"
    max_concurrent_requests: int = 4
    enable_gpu: bool = True
    medgemma_backend: str = "stub"
    default_pipeline: str = ""
    enable_audit_logging: bool = True


class DashboardEngine:
    """
    Rhenium OS Dashboard Engine.
    
    Provides a unified API for dashboard applications to access
    all Rhenium OS capabilities including:
    
    - Study analysis with perception models
    - Disease reasoning and hypothesis generation
    - XAI evidence generation
    - MedGemma report drafting
    - Pipeline management
    - System health monitoring
    
    Usage:
        engine = DashboardEngine(config)
        engine.initialize()
        
        # Analyze a study
        response = engine.analyze_study(request)
        
        # Get disease assessment
        disease = engine.get_disease_assessment(study_id)
        
        # Get report draft
        report = engine.get_report_draft(study_id)
    """
    
    def __init__(self, config: Optional[EngineConfig] = None):
        """Initialize dashboard engine with configuration."""
        self.config = config or EngineConfig()
        self._status = EngineStatus.INITIALIZING
        self._start_time = datetime.now(timezone.utc)
        self._pipelines: dict[str, PipelineInfo] = {}
        self._results_cache: dict[str, AnalysisResponse] = {}
        self._disease_cache: dict[str, DiseaseReasoningOutput] = {}
        self._request_count = 0
        self._error_count = 0
        self._last_error: Optional[str] = None
        self._version = "1.0.0"
    
    def initialize(self) -> None:
        """
        Initialize the engine and load pipelines.
        
        Loads pipeline configurations and prepares models for inference.
        """
        try:
            self._load_pipelines()
            self._status = EngineStatus.READY
        except Exception as e:
            self._status = EngineStatus.ERROR
            self._last_error = str(e)
            raise
    
    def shutdown(self) -> None:
        """Shutdown the engine and release resources."""
        self._status = EngineStatus.SHUTDOWN
        self._pipelines.clear()
        self._results_cache.clear()
        self._disease_cache.clear()
    
    def _load_pipelines(self) -> None:
        """Load available pipeline configurations."""
        # Load from configs directory
        configs_dir = Path(__file__).parent.parent / "pipelines" / "configs"
        if configs_dir.exists():
            for config_file in configs_dir.glob("*.yaml"):
                pipeline_id = config_file.stem
                self._pipelines[pipeline_id] = PipelineInfo(
                    pipeline_id=pipeline_id,
                    name=pipeline_id.replace("_", " ").title(),
                    version="1.0.0",
                    modality=self._extract_modality(pipeline_id),
                    body_part=self._extract_body_part(pipeline_id),
                    description=f"Pipeline configuration from {config_file.name}",
                )
    
    def _extract_modality(self, pipeline_id: str) -> str:
        """Extract modality from pipeline ID."""
        if "mri" in pipeline_id.lower():
            return "MRI"
        elif "ct" in pipeline_id.lower():
            return "CT"
        elif "xray" in pipeline_id.lower() or "mammo" in pipeline_id.lower():
            return "X-ray"
        elif "us" in pipeline_id.lower():
            return "Ultrasound"
        return "Unknown"
    
    def _extract_body_part(self, pipeline_id: str) -> str:
        """Extract body part from pipeline ID."""
        parts = ["knee", "brain", "chest", "liver", "prostate", "breast", "head"]
        for part in parts:
            if part in pipeline_id.lower():
                return part.upper()
        return "Unknown"
    
    # =========================================================================
    # Core Analysis Methods
    # =========================================================================
    
    def analyze_study(self, request: AnalysisRequest) -> AnalysisResponse:
        """
        Analyze an imaging study using the specified pipeline.
        
        Args:
            request: Analysis request with study ID and options.
            
        Returns:
            Analysis response with findings and results.
        """
        request_id = uuid4().hex[:12]
        start_time = time.time()
        
        try:
            self._request_count += 1
            
            # Select pipeline
            pipeline_id = request.pipeline_id or self.config.default_pipeline
            if not pipeline_id and self._pipelines:
                pipeline_id = list(self._pipelines.keys())[0]
            
            # Create response (stub implementation)
            response = AnalysisResponse(
                request_id=request_id,
                study_id=request.study_id,
                status="completed",
                pipeline_id=pipeline_id,
                findings_count=0,
                has_critical_findings=False,
                processing_time_ms=(time.time() - start_time) * 1000,
            )
            
            # Cache result
            self._results_cache[request.study_id] = response
            
            return response
            
        except Exception as e:
            self._error_count += 1
            self._last_error = str(e)
            return AnalysisResponse(
                request_id=request_id,
                study_id=request.study_id,
                status="error",
                pipeline_id=request.pipeline_id,
                error_message=str(e),
            )
    
    def get_disease_assessment(
        self, study_id: str
    ) -> Optional[DiseaseReasoningOutput]:
        """
        Get disease reasoning output for a study.
        
        Args:
            study_id: Study identifier.
            
        Returns:
            Disease reasoning output or None if not available.
        """
        return self._disease_cache.get(study_id)
    
    def get_findings(self, study_id: str) -> list[dict[str, Any]]:
        """
        Get perception findings for a study.
        
        Args:
            study_id: Study identifier.
            
        Returns:
            List of findings as dictionaries.
        """
        # Return cached findings or empty list
        result = self._results_cache.get(study_id)
        if result:
            return []  # Placeholder
        return []
    
    def get_evidence_dossier(self, study_id: str) -> Optional[dict[str, Any]]:
        """
        Get XAI evidence dossier for a study.
        
        Args:
            study_id: Study identifier.
            
        Returns:
            Evidence dossier as dictionary or None.
        """
        result = self._results_cache.get(study_id)
        if result and result.evidence_dossier_id:
            return {"dossier_id": result.evidence_dossier_id}
        return None
    
    def get_report_draft(self, study_id: str) -> Optional[dict[str, Any]]:
        """
        Get MedGemma report draft for a study.
        
        Args:
            study_id: Study identifier.
            
        Returns:
            Report draft as dictionary or None.
        """
        result = self._results_cache.get(study_id)
        if result and result.report_draft_id:
            return {"report_id": result.report_draft_id}
        return None
    
    # =========================================================================
    # Pipeline Management
    # =========================================================================
    
    def list_pipelines(self) -> list[PipelineInfo]:
        """
        List available pipelines.
        
        Returns:
            List of pipeline information objects.
        """
        return list(self._pipelines.values())
    
    def get_pipeline(self, pipeline_id: str) -> Optional[PipelineInfo]:
        """
        Get information about a specific pipeline.
        
        Args:
            pipeline_id: Pipeline identifier.
            
        Returns:
            Pipeline information or None if not found.
        """
        return self._pipelines.get(pipeline_id)
    
    # =========================================================================
    # System Status
    # =========================================================================
    
    def get_status(self) -> HealthStatus:
        """
        Get current engine health status.
        
        Returns:
            Health status object with system metrics.
        """
        uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds()
        
        return HealthStatus(
            status=self._status,
            version=self._version,
            uptime_seconds=uptime,
            pipelines_loaded=len(self._pipelines),
            models_loaded=0,  # Placeholder
            gpu_available=self.config.enable_gpu,
            gpu_memory_used_mb=0.0,
            gpu_memory_total_mb=0.0,
            pending_requests=0,
            completed_requests=self._request_count,
            error_count=self._error_count,
            last_error=self._last_error,
        )
    
    @property
    def is_ready(self) -> bool:
        """Check if engine is ready for requests."""
        return self._status == EngineStatus.READY
    
    @property
    def status(self) -> EngineStatus:
        """Get current engine status."""
        return self._status
