# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Rhenium OS Dashboard Engine
===========================

Backend engine for dashboard API integration.
Provides unified access to all Rhenium OS capabilities.
"""

from rhenium.engine.dashboard_engine import (
    DashboardEngine,
    EngineConfig,
    EngineStatus,
)
from rhenium.engine.api_models import (
    AnalysisRequest,
    AnalysisResponse,
    StudyInfo,
    PipelineInfo,
    HealthStatus,
)

__all__ = [
    "DashboardEngine",
    "EngineConfig",
    "EngineStatus",
    "AnalysisRequest",
    "AnalysisResponse",
    "StudyInfo",
    "PipelineInfo",
    "HealthStatus",
]
