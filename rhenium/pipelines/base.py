# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Rhenium OS Core Pipeline Interfaces
===================================

Defines the synchronous execution contract for all modality pipelines.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List
import time
import uuid

@dataclass
class PipelineContext:
    """Context object passed through the pipeline stages."""
    pipeline_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    patient_id: str | None = None
    study_uid: str | None = None
    start_time: float = field(default_factory=time.time)
    
    # Shared state
    metadata: Dict[str, Any] = field(default_factory=dict)
    images: Dict[str, Any] = field(default_factory=dict)
    findings: List[Dict[str, Any]] = field(default_factory=list)
    reports: List[str] = field(default_factory=list)
    
    execution_log: List[str] = field(default_factory=list)

    def log(self, step: str, message: str):
        entry = f"[{time.time() - self.start_time:.3f}s] [{step}] {message}"
        self.execution_log.append(entry)
        print(entry)


class PipelineStage(ABC):
    """A single synchronous stage in the pipeline."""
    
    @abstractmethod
    def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Process the context and return it.
        Modifies context in place is allowed/expected.
        """
        pass


class SynchronousPipeline(ABC):
    """
    Orchestrator for a sequence of stages.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.stages: List[PipelineStage] = []
        
    def add_stage(self, stage: PipelineStage):
        self.stages.append(stage)
        
    def run(self, context: PipelineContext) -> PipelineContext:
        context.log("Orchestrator", f"Starting pipeline: {self.name}")
        
        for stage in self.stages:
            stage_name = stage.__class__.__name__
            context.log("Orchestrator", f"Executing stage: {stage_name}")
            try:
                context = stage.execute(context)
            except Exception as e:
                context.log("Error", f"Stage {stage_name} failed: {str(e)}")
                raise
                
        context.log("Orchestrator", f"Finished pipeline: {self.name}")
        return context
