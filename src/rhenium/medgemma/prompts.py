"""MedGemma prompts for medical imaging tasks."""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any


@dataclass
class PromptTemplate:
    """Template for MedGemma prompts."""
    name: str
    system_prompt: str
    user_template: str


SYSTEM_PROMPTS = {
    "radiologist": (
        "You are an expert radiologist AI assistant. Provide clear, accurate, "
        "and clinically relevant interpretations. Always note limitations and "
        "recommend verification by a qualified physician."
    ),
    "researcher": (
        "You are a medical imaging research assistant. Provide detailed technical "
        "analysis of imaging data, including quantitative measurements and "
        "potential methodological considerations."
    ),
}


FINDING_NARRATIVE = PromptTemplate(
    name="finding_narrative",
    system_prompt=SYSTEM_PROMPTS["radiologist"],
    user_template="""
Based on the AI analysis results, generate a clinical narrative for the following finding:

**Finding Type**: {finding_type}
**Description**: {description}
**Confidence**: {confidence:.0%}
**Location**: {location}

Quantitative Measurements:
{measurements}

Please provide:
1. A clear description of the finding
2. Clinical significance
3. Recommended follow-up actions
4. Important limitations to note
""",
)


COMPARISON_NARRATIVE = PromptTemplate(
    name="comparison_narrative",
    system_prompt=SYSTEM_PROMPTS["radiologist"],
    user_template="""
Compare the current study with prior imaging:

**Current Study Date**: {current_date}
**Prior Study Date**: {prior_date}
**Modality**: {modality}

Current Findings:
{current_findings}

Prior Findings:
{prior_findings}

Please describe any changes, stability, or progression of findings.
""",
)


QUALITY_ASSESSMENT = PromptTemplate(
    name="quality_assessment",
    system_prompt=SYSTEM_PROMPTS["researcher"],
    user_template="""
Assess the quality of this medical image:

**Modality**: {modality}
**Body Region**: {body_region}

Image Quality Metrics:
- Signal-to-Noise Ratio: {snr}
- Contrast: {contrast}
- Spatial Resolution: {resolution}

Provide a quality assessment and any recommendations for image acquisition improvement.
""",
)


def format_prompt(
    template: PromptTemplate,
    **kwargs: Any,
) -> tuple[str, str]:
    """Format a prompt template with provided values."""
    user_prompt = template.user_template.format(**kwargs)
    return template.system_prompt, user_prompt


def get_template(name: str) -> PromptTemplate:
    """Get prompt template by name."""
    templates = {
        "finding_narrative": FINDING_NARRATIVE,
        "comparison": COMPARISON_NARRATIVE,
        "quality": QUALITY_ASSESSMENT,
    }
    return templates.get(name, FINDING_NARRATIVE)
