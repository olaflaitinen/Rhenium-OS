# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Rhenium OS Core Pipeline Interfaces
===================================

Defines the synchronous execution contract for all modality pipelines.
Now integrated with OutputManager for Evidence Dossier generation.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time
import uuid

from rhenium.outputs.manager import OutputManager

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
    
    # Output Management (Orchestrator)
    output_manager: Optional[OutputManager] = None
    
    # Legacy execution log (kept for compatibility, though manager handles logs too)
    execution_log: List[str] = field(default_factory=list)

    def log(self, step: str, message: str):
        entry = f"[{time.time() - self.start_time:.3f}s] [{step}] {message}"
        self.execution_log.append(entry)
        print(entry)
        if self.output_manager:
            self.output_manager.add_log(entry)


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
    
    def __init__(self, name: str, modality: str = "Unknown"):
        self.name = name
        self.modality = modality
        self.stages: List[PipelineStage] = []
        
    def add_stage(self, stage: PipelineStage):
        self.stages.append(stage)
        
    def run(self, context: PipelineContext) -> PipelineContext:
        # Initialize Output Manager if not present
        if context.output_manager is None:
            case_id = context.pipeline_id
            context.output_manager = OutputManager(case_id=case_id, modality=self.modality)
            if context.patient_id:
                context.output_manager.set_patient_info(
                    patient_id=context.patient_id,
                    study_date=time.strftime("%Y-%m-%d"),
                    protocol=self.name
                )

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
        
        # Save output dossier
        if context.output_manager:
            saved_path = context.output_manager.save_dossier()
            context.log("Orchestrator", f"Evidence Dossier saved to: {saved_path}")
            
        return context
