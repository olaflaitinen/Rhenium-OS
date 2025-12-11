# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
MedGemma Disease Explanation Prompts
====================================

Prompt templates for generating disease-level explanations via MedGemma.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class DiseaseExplanationTemplate:
    """Template for disease-level explanation generation."""
    name: str
    system_prompt: str
    user_prompt_template: str
    required_inputs: list[str]
    output_sections: list[str]


# Disease Presence Explanation Template
PRESENCE_EXPLANATION_TEMPLATE = DiseaseExplanationTemplate(
    name="disease_presence_explanation",
    system_prompt="""You are a radiologist assistant explaining AI-generated disease 
presence assessments. Provide clear, clinically grounded explanations that:
1. Describe what was evaluated
2. Explain the presence/absence finding
3. Summarize supporting evidence
4. Acknowledge limitations and uncertainty

Never provide definitive diagnoses. Always recommend clinical correlation.""",
    user_prompt_template="""
DISEASE PRESENCE ASSESSMENT:
- Status: {presence_status}
- Confidence: {confidence:.0%}
- Modality: {modality}
- Organ Systems Evaluated: {organ_systems}

SUPPORTING EVIDENCE:
{supporting_evidence}

RATIONALE:
{rationale}

LIMITATIONS:
{limitations}

Generate a narrative explanation of this disease presence assessment suitable 
for inclusion in a clinical report. Include any caveats or recommendations.""",
    required_inputs=["presence_status", "confidence", "modality", "organ_systems", 
                     "supporting_evidence", "rationale", "limitations"],
    output_sections=["explanation", "caveats"],
)


# Disease Hypothesis Explanation Template
HYPOTHESIS_EXPLANATION_TEMPLATE = DiseaseExplanationTemplate(
    name="disease_hypothesis_explanation",
    system_prompt="""You are a radiologist assistant explaining AI-generated disease 
hypotheses. Provide educational, clinically relevant explanations that:
1. Describe the disease hypothesis
2. Explain supporting imaging features
3. Note contradicting features
4. Discuss differential considerations
5. Suggest appropriate clinical correlation

Use probabilistic language. Never claim diagnostic certainty.""",
    user_prompt_template="""
DISEASE HYPOTHESIS:
- Disease: {disease_name}
- Probability: {probability:.0%}
- Confidence Level: {confidence_level}

SUPPORTING FEATURES:
{supporting_features}

CONTRADICTING FEATURES:
{contradicting_features}

IMAGING EVIDENCE:
{imaging_evidence}

Generate a clinical explanation of why this disease hypothesis is being 
considered, including the imaging rationale and any limitations.""",
    required_inputs=["disease_name", "probability", "confidence_level",
                     "supporting_features", "contradicting_features", "imaging_evidence"],
    output_sections=["clinical_context", "imaging_rationale", "limitations"],
)


# Differential Diagnosis Explanation Template
DIFFERENTIAL_EXPLANATION_TEMPLATE = DiseaseExplanationTemplate(
    name="differential_diagnosis_explanation",
    system_prompt="""You are a radiologist assistant explaining AI-generated differential 
diagnoses. Provide clear, ranked explanations that:
1. Explain each diagnosis in the differential
2. Distinguish between alternatives
3. Highlight key discriminating features
4. Recommend additional workup when appropriate

Use imaging-appropriate language. Emphasize that these are imaging surrogates.""",
    user_prompt_template="""
DIFFERENTIAL DIAGNOSIS (IMAGING-BASED):
{differential_list}

PRIMARY FINDINGS:
{primary_findings}

KEY IMAGING FEATURES:
{key_features}

Generate a narrative differential diagnosis explanation suitable for 
a clinical report, including any recommendations for further workup.""",
    required_inputs=["differential_list", "primary_findings", "key_features"],
    output_sections=["differential_narrative", "recommendations"],
)


# Safety Flag Explanation Template
SAFETY_FLAG_EXPLANATION_TEMPLATE = DiseaseExplanationTemplate(
    name="safety_flag_explanation",
    system_prompt="""You are a radiologist assistant explaining urgent AI-detected 
findings that require clinical escalation. Provide clear, actionable explanations:
1. Describe the critical finding
2. Explain the clinical urgency
3. Provide recommended actions
4. Maintain appropriate urgency without causing alarm

These findings require immediate attention.""",
    user_prompt_template="""
CRITICAL FINDING:
- Type: {flag_type}
- Severity: {severity}
- Description: {description}

RECOMMENDED ACTION:
{recommended_action}

TIME SENSITIVITY: {time_sensitivity}

Generate a clear, urgent explanation of this finding suitable for 
immediate clinical communication.""",
    required_inputs=["flag_type", "severity", "description", 
                     "recommended_action", "time_sensitivity"],
    output_sections=["urgent_notification", "recommended_actions"],
)


# Trajectory Explanation Template
TRAJECTORY_EXPLANATION_TEMPLATE = DiseaseExplanationTemplate(
    name="trajectory_explanation",
    system_prompt="""You are a radiologist assistant explaining disease trajectory 
based on comparison with prior imaging. Provide interval change explanations:
1. Summarize changes from prior
2. Quantify changes when possible
3. Interpret clinical significance
4. Note any limitations in comparison

Use standard radiology comparison terminology.""",
    user_prompt_template="""
TRAJECTORY ASSESSMENT:
- Current Study: {current_study_date}
- Prior Study: {prior_study_date}
- Interval: {interval_days} days
- Trajectory: {trajectory_label}

QUANTITATIVE CHANGES:
{quantitative_changes}

NEW FINDINGS:
{new_findings}

RESOLVED FINDINGS:
{resolved_findings}

RESPONSE CATEGORY: {response_category}

Generate a comparison narrative suitable for a clinical report.""",
    required_inputs=["current_study_date", "prior_study_date", "interval_days",
                     "trajectory_label", "quantitative_changes", "new_findings",
                     "resolved_findings", "response_category"],
    output_sections=["comparison_narrative", "clinical_interpretation"],
)


# Stage Assessment Explanation Template  
STAGING_EXPLANATION_TEMPLATE = DiseaseExplanationTemplate(
    name="staging_explanation",
    system_prompt="""You are a radiologist assistant explaining imaging-based 
disease staging. Clearly communicate that:
1. Staging is an imaging surrogate only
2. Definitive staging requires clinical and pathologic correlation
3. Explain the stage assignment rationale
4. Note what cannot be assessed by imaging

Use appropriate medical terminology and standard staging systems.""",
    user_prompt_template="""
IMAGING-BASED STAGING:
- Disease: {disease_name}
- Staging System: {staging_system}
- Stage: {stage_label}
- T Component: {t_component}
- N Component: {n_component}
- M Component: {m_component}

ASSUMPTIONS:
{assumptions}

LIMITATIONS:
{limitations}

Generate a staging explanation emphasizing that this is an imaging 
surrogate requiring clinical and pathologic correlation.""",
    required_inputs=["disease_name", "staging_system", "stage_label",
                     "t_component", "n_component", "m_component",
                     "assumptions", "limitations"],
    output_sections=["staging_narrative", "disclaimer"],
)


# Registry of all templates
DISEASE_EXPLANATION_TEMPLATES = {
    "presence": PRESENCE_EXPLANATION_TEMPLATE,
    "hypothesis": HYPOTHESIS_EXPLANATION_TEMPLATE,
    "differential": DIFFERENTIAL_EXPLANATION_TEMPLATE,
    "safety_flag": SAFETY_FLAG_EXPLANATION_TEMPLATE,
    "trajectory": TRAJECTORY_EXPLANATION_TEMPLATE,
    "staging": STAGING_EXPLANATION_TEMPLATE,
}


def get_template(template_name: str) -> DiseaseExplanationTemplate:
    """Get a disease explanation template by name."""
    if template_name not in DISEASE_EXPLANATION_TEMPLATES:
        raise ValueError(f"Unknown template: {template_name}")
    return DISEASE_EXPLANATION_TEMPLATES[template_name]


def format_prompt(
    template: DiseaseExplanationTemplate,
    inputs: dict[str, Any],
) -> str:
    """Format a template with provided inputs."""
    return template.user_prompt_template.format(**inputs)
