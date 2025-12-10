# Copyright (c) 2025 Skolyn LLC. All rights reserved.
# SPDX-License-Identifier: EUPL-1.1

"""
Reporting Templates
===================

Prompt templates for radiology report generation with MedGemma.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


STRUCTURED_REPORT_SYSTEM_PROMPT = """You are an expert radiologist assistant helping to generate 
structured radiology reports. Your reports must be:
- Clinically accurate and evidence-based
- Professionally worded using standard radiology terminology
- Clear about uncertainty and limitations
- Suitable for clinical decision-making support

Always emphasize that AI findings require human radiologist verification."""


@dataclass
class ReportTemplate:
    """Template for report generation."""
    name: str
    system_prompt: str
    user_prompt_template: str
    sections: list[str]


GENERAL_REPORT_TEMPLATE = ReportTemplate(
    name="general_radiology",
    system_prompt=STRUCTURED_REPORT_SYSTEM_PROMPT,
    user_prompt_template="""Generate a structured radiology report based on the following:

CLINICAL INDICATION:
{indication}

TECHNIQUE:
{technique}

AI-DETECTED FINDINGS:
{findings}

MEASUREMENTS:
{measurements}

Generate a report with the following sections:
1. INDICATION
2. TECHNIQUE
3. COMPARISON
4. FINDINGS
5. IMPRESSION

Important: Clearly state that these are AI-assisted findings requiring radiologist review.""",
    sections=["indication", "technique", "comparison", "findings", "impression"],
)


MRI_KNEE_TEMPLATE = ReportTemplate(
    name="mri_knee",
    system_prompt=STRUCTURED_REPORT_SYSTEM_PROMPT,
    user_prompt_template="""Generate a knee MRI report based on:

INDICATION: {indication}
TECHNIQUE: MRI of the knee without contrast

AI FINDINGS:
- Menisci: {meniscus_findings}
- Ligaments: {ligament_findings}
- Cartilage: {cartilage_findings}
- Other: {other_findings}

MEASUREMENTS:
{measurements}

Structure the report with standard sections. Note any areas of uncertainty.""",
    sections=["indication", "technique", "findings", "impression"],
)


BRAIN_MRI_TEMPLATE = ReportTemplate(
    name="mri_brain",
    system_prompt=STRUCTURED_REPORT_SYSTEM_PROMPT,
    user_prompt_template="""Generate a brain MRI report based on:

INDICATION: {indication}
TECHNIQUE: {technique}

AI FINDINGS:
- White matter: {white_matter_findings}
- Lesions: {lesion_findings}
- Other: {other_findings}

Volume measurements:
{volume_measurements}

Structure with standard sections. Emphasize clinical significance.""",
    sections=["indication", "technique", "findings", "impression"],
)


def format_report_prompt(
    template: ReportTemplate,
    context: dict[str, Any],
) -> str:
    """Format prompt template with context."""
    # Fill in template with available context, use default for missing
    filled_context = {k: context.get(k, "Not provided") for k in _extract_placeholders(template.user_prompt_template)}
    return template.user_prompt_template.format(**filled_context)


def _extract_placeholders(template: str) -> list[str]:
    """Extract placeholder names from template."""
    import re
    return re.findall(r'\{(\w+)\}', template)
