# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
MedGemma Integration Module
===========================

Adapter layer for MedGemma multi-modal reasoning and explanation.
"""

from rhenium.medgemma.adapter import (
    MedGemmaClient,
    LocalMedGemmaClient,
    RemoteMedGemmaClient,
    StubMedGemmaClient,
    get_medgemma_client,
)

__all__ = [
    "MedGemmaClient",
    "LocalMedGemmaClient",
    "RemoteMedGemmaClient",
    "StubMedGemmaClient",
    "get_medgemma_client",
]
