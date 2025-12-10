# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Rhenium OS Pipeline Orchestrator (Runner)
=========================================

Central entry point to run modality-specific pipelines synchronously.
"""

from __future__ import annotations

from typing import Dict, Type
from .base import SynchronousPipeline, PipelineContext

class PipelineRegistry:
    """Registry of available pipelines."""
    _pipelines: Dict[str, Type[SynchronousPipeline]] = {}
    
    @classmethod
    def register(cls, name: str, pipeline_cls: Type[SynchronousPipeline]):
        cls._pipelines[name] = pipeline_cls
        
    @classmethod
    def get(cls, name: str) -> Type[SynchronousPipeline]:
        if name not in cls._pipelines:
            raise ValueError(f"Pipeline '{name}' not found. Available: {list(cls._pipelines.keys())}")
        return cls._pipelines[name]


def run_pipeline(pipeline_name: str, **context_kwargs) -> PipelineContext:
    """
    Helper function to instantiate and run a pipeline.
    
    Args:
        pipeline_name: Name of registered pipeline
        **context_kwargs: Initial data for PipelineContext (patient_id, etc.)
        
    Returns:
        Final PipelineContext
    """
    pipeline_cls = PipelineRegistry.get(pipeline_name)
    pipeline = pipeline_cls()
    
    context = PipelineContext(**context_kwargs)
    return pipeline.run(context)
