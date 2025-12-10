# MedGemma Integration Architecture

**Last Updated: December 2025**

---

## Overview

The MedGemma module provides clinical reasoning and explanation capabilities within Rhenium OS. As one component of the broader model ecosystem—alongside PINNs, GANs, U-Net, Vision Transformers, and 3D CNNs—MedGemma specializes in generating structured reports, providing finding explanations, and ensuring narrative consistency. This document describes the integration architecture, design patterns, and best practices.

---

## Role of MedGemma

| Function | Description |
|----------|-------------|
| Report Drafting | Generate structured radiology report sections |
| Finding Explanation | Provide clinical reasoning for AI-detected findings |
| Consistency Validation | Check logical consistency across findings |
| Question Answering | Interactive queries about imaging cases |
| Guideline Alignment | Map findings to clinical guidelines |

---

## Adapter Architecture

```
+------------------------------------------------------------------+
|                     MedGemma Adapter Layer                        |
|  +-------------------------+  +-------------------------------+   |
|  |   Abstract Client       |  |     Tool-Use Framework        |   |
|  |   - generate_report()   |  |     - GuidelineChecker        |   |
|  |   - explain_finding()   |  |     - RuleBasedValidator      |   |
|  |   - validate_consistency|  |     - MeasurementVerifier     |   |
|  |   - answer_question()   |  |     - KnowledgeGraphQuery     |   |
|  +-------------------------+  +-------------------------------+   |
|                 |                            |                    |
|  +----------------------------------------------------------+    |
|  |                 Prompt Template Engine                    |    |
|  |   - Organ-specific templates                             |    |
|  |   - Modality-specific templates                          |    |
|  |   - Uncertainty-aware templates                          |    |
|  +----------------------------------------------------------+    |
+------------------------------------------------------------------+
                               |
            +------------------+------------------+
            |                  |                  |
     +------v------+    +------v------+    +------v------+
     | StubClient  |    | LocalClient |    | RemoteClient|
     | (Testing)   |    | (On-Prem)   |    | (Cloud API) |
     +-------------+    +-------------+    +-------------+
```

---

## Client Implementations

### StubMedGemmaClient

- Used for testing without actual model inference
- Returns deterministic, template-based responses
- Enables CI/CD testing without GPU resources

### LocalMedGemmaClient

- Loads MedGemma weights from local storage
- Runs inference on local GPU
- Full control over model version and configuration

### RemoteMedGemmaClient

- Connects to remote MedGemma API endpoint
- HTTP-based communication with retry logic
- Suitable for cloud deployments

---

## Core Methods

### generate_report()

```python
def generate_report(
    self,
    findings: list[Finding],
    clinical_context: dict[str, Any] | None = None,
    template: ReportTemplate | None = None,
) -> ReportDraft:
    """
    Generate structured radiology report from findings.
    
    Args:
        findings: List of AI-detected findings with evidence
        clinical_context: Patient history, indication, prior studies
        template: Report template (organ/modality specific)
    
    Returns:
        ReportDraft with sections: indication, technique, 
        comparison, findings, impression, recommendations
    """
```

### explain_finding()

```python
def explain_finding(
    self,
    finding: Finding,
    dossier: EvidenceDossier,
    context: dict[str, Any] | None = None,
) -> NarrativeEvidence:
    """
    Generate explanation for a specific finding.
    
    The explanation includes:
    - Description of what was detected
    - Clinical significance
    - Differential diagnosis considerations
    - Limitations and uncertainty
    - Recommendations
    """
```

### validate_consistency()

```python
def validate_consistency(
    self,
    findings: list[Finding],
    metadata: dict[str, Any] | None = None,
) -> ConsistencyReport:
    """
    Check logical consistency across findings.
    
    Validates:
    - Laterality consistency
    - Measurement plausibility
    - Anatomical consistency
    - Temporal consistency (if prior available)
    """
```

---

## Tool-Use Framework

MedGemma can invoke external tools for enhanced reasoning:

### GuidelineChecker

Verifies findings against clinical guidelines (ACR, ESR, etc.):

```python
class GuidelineChecker:
    def check_alignment(
        self,
        finding: Finding,
        guideline_set: str,
    ) -> GuidelineAlignment:
        """
        Check finding against relevant guidelines.
        Returns alignment status and recommendations.
        """
```

### RuleBasedValidator

Applies deterministic validation rules:

```python
class RuleBasedValidator:
    def validate(self, findings: list[Finding]) -> list[ValidationIssue]:
        """
        Apply rule-based validation:
        - Left/right consistency
        - Measurement range plausibility
        - Required measurements present
        """
```

### MeasurementVerifier

Cross-checks measurements for consistency:

```python
class MeasurementVerifier:
    def verify(
        self,
        measurements: dict[str, float],
        expected_ranges: dict[str, tuple[float, float]],
    ) -> list[MeasurementIssue]:
        """
        Verify measurements against expected ranges.
        """
```

---

## Prompt Engineering

### Template Structure

```python
@dataclass
class ReportTemplate:
    name: str                      # e.g., "knee_mri"
    system_prompt: str             # Role and constraints
    user_prompt_template: str      # Formatted with findings/context
    sections: list[str]            # Required report sections
    output_format: str             # structured, free-text
```

### Best Practices

1. **Explicit Role Definition**: System prompts clearly define MedGemma's role as a radiologist assistant

2. **Structured Input**: Findings and measurements provided in consistent format

3. **Uncertainty Elicitation**: Prompts designed to surface uncertainty

4. **Limitation Acknowledgment**: Templates include space for caveats

5. **No Overstatement**: Prompts discourage definitive diagnostic claims

### Example Template

```python
KNEE_MRI_TEMPLATE = ReportTemplate(
    name="knee_mri",
    system_prompt="""You are an expert radiologist assistant 
    generating structured reports for knee MRI studies. 
    Always acknowledge uncertainty and recommend human review.""",
    user_prompt_template="""
    CLINICAL INDICATION: {indication}
    TECHNIQUE: {technique}
    
    AI-DETECTED FINDINGS:
    {findings}
    
    MEASUREMENTS:
    {measurements}
    
    Generate a structured report with sections:
    TECHNIQUE, FINDINGS, IMPRESSION
    
    Acknowledge any limitations or uncertainty.
    """,
    sections=["technique", "findings", "impression"],
)
```

---

## Logging and Audit

All MedGemma interactions are logged:

```python
@dataclass
class MedGemmaAuditEntry:
    timestamp: datetime
    method: str                    # generate_report, explain_finding
    input_hash: str                # Hash of input for reproducibility
    output_hash: str               # Hash of output
    latency_ms: float
    token_count: int
    model_version: str
```

---

## Safety Considerations

### Hallucination Mitigation

- Cross-reference with quantitative evidence
- Rule-based validation of claims
- Confidence calibration

### Overconfidence Prevention

- Prompts designed to elicit uncertainty
- Post-processing to add disclaimers
- Rejection of definitive diagnostic claims

### PHI Protection

- No patient identifiers in prompts
- Input sanitization before logging
- Output review for inadvertent PHI

---

## Conclusion

MedGemma integration in Rhenium OS follows a robust, auditable architecture that enables powerful clinical reasoning while maintaining safety, transparency, and regulatory compliance.

---

**Copyright (c) 2025 Skolyn LLC. All rights reserved.**

**SPDX-License-Identifier: EUPL-1.1**
