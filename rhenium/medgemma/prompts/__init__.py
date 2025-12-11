# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
MedGemma Prompt Templates
========================

Prompt template modules for MedGemma integration.
"""

from rhenium.medgemma.prompts.disease_explanation import (
    DiseaseExplanationTemplate,
    DISEASE_EXPLANATION_TEMPLATES,
    get_template,
    format_prompt,
)

__all__ = [
    "DiseaseExplanationTemplate",
    "DISEASE_EXPLANATION_TEMPLATES",
    "get_template",
    "format_prompt",
]
