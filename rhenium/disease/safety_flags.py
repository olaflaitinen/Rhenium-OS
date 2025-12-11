# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Clinical Safety Flag Detection
==============================

Detection of urgent patterns requiring clinical escalation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from rhenium.core.disease_types import (
    CaseEvidenceBundle,
    DiseaseHypothesis,
    ClinicalSafetyFlag,
    SafetyFlagType,
    SafetyFlagSeverity,
)
from rhenium.core.logging import get_pipeline_logger

logger = get_pipeline_logger()


@dataclass
class SafetyPattern:
    """Definition of a clinical safety pattern."""
    name: str
    flag_type: SafetyFlagType
    severity: SafetyFlagSeverity
    description: str
    recommended_action: str
    time_sensitivity: str = ""
    escalation_required: bool = False


class SafetyFlagDetector:
    """
    Detect clinical safety patterns requiring urgent attention.
    
    Identifies red flag patterns across organ systems that should
    trigger clinical escalation or urgent review.
    """
    
    # Registry of safety patterns by organ/modality
    SAFETY_PATTERNS: dict[str, list[SafetyPattern]] = {
        "chest_ct": [
            SafetyPattern(
                name="massive_pe",
                flag_type=SafetyFlagType.URGENT_FINDING,
                severity=SafetyFlagSeverity.CRITICAL,
                description="Findings suggestive of massive pulmonary embolism with saddle thrombus or central vessel involvement",
                recommended_action="Immediate clinical notification; consider thrombolysis evaluation",
                time_sensitivity="immediate",
                escalation_required=True,
            ),
            SafetyPattern(
                name="tension_pneumothorax",
                flag_type=SafetyFlagType.URGENT_FINDING,
                severity=SafetyFlagSeverity.CRITICAL,
                description="Findings suggestive of tension pneumothorax with mediastinal shift",
                recommended_action="Immediate clinical notification; emergent decompression may be required",
                time_sensitivity="immediate",
                escalation_required=True,
            ),
            SafetyPattern(
                name="aortic_dissection",
                flag_type=SafetyFlagType.URGENT_FINDING,
                severity=SafetyFlagSeverity.CRITICAL,
                description="Findings suggestive of aortic dissection",
                recommended_action="Immediate clinical notification; vascular surgery consultation",
                time_sensitivity="immediate",
                escalation_required=True,
            ),
        ],
        "head_ct": [
            SafetyPattern(
                name="large_ich",
                flag_type=SafetyFlagType.URGENT_FINDING,
                severity=SafetyFlagSeverity.CRITICAL,
                description="Findings suggestive of large intracranial hemorrhage",
                recommended_action="Immediate clinical notification; neurosurgical evaluation",
                time_sensitivity="immediate",
                escalation_required=True,
            ),
            SafetyPattern(
                name="midline_shift",
                flag_type=SafetyFlagType.URGENT_FINDING,
                severity=SafetyFlagSeverity.CRITICAL,
                description="Significant midline shift indicating mass effect",
                recommended_action="Immediate clinical notification; neurosurgical evaluation",
                time_sensitivity="immediate",
                escalation_required=True,
            ),
            SafetyPattern(
                name="herniation",
                flag_type=SafetyFlagType.URGENT_FINDING,
                severity=SafetyFlagSeverity.CRITICAL,
                description="Findings suggestive of brain herniation",
                recommended_action="Immediate clinical notification; emergent intervention required",
                time_sensitivity="immediate",
                escalation_required=True,
            ),
        ],
        "abdomen_ct": [
            SafetyPattern(
                name="free_air",
                flag_type=SafetyFlagType.URGENT_FINDING,
                severity=SafetyFlagSeverity.CRITICAL,
                description="Free intraperitoneal air suggestive of bowel perforation",
                recommended_action="Immediate clinical notification; surgical evaluation",
                time_sensitivity="immediate",
                escalation_required=True,
            ),
            SafetyPattern(
                name="aaa_rupture",
                flag_type=SafetyFlagType.URGENT_FINDING,
                severity=SafetyFlagSeverity.CRITICAL,
                description="Findings suggestive of ruptured abdominal aortic aneurysm",
                recommended_action="Immediate clinical notification; vascular surgery consultation",
                time_sensitivity="immediate",
                escalation_required=True,
            ),
        ],
        "image_quality": [
            SafetyPattern(
                name="severe_motion",
                flag_type=SafetyFlagType.LIMITED_IMAGE_QUALITY,
                severity=SafetyFlagSeverity.MODERATE,
                description="Severe motion artifact limiting diagnostic quality",
                recommended_action="Consider repeat imaging if clinically indicated",
                time_sensitivity="routine",
                escalation_required=False,
            ),
            SafetyPattern(
                name="incomplete_coverage",
                flag_type=SafetyFlagType.LIMITED_IMAGE_QUALITY,
                severity=SafetyFlagSeverity.MODERATE,
                description="Incomplete anatomical coverage; some regions not evaluated",
                recommended_action="Clinical correlation recommended; consider additional imaging",
                time_sensitivity="routine",
                escalation_required=False,
            ),
        ],
    }
    
    def __init__(self, enabled_patterns: list[str] | None = None):
        """Initialize detector with optional pattern filtering."""
        self.enabled_patterns = enabled_patterns
    
    def detect(
        self,
        evidence: CaseEvidenceBundle,
        hypotheses: list[DiseaseHypothesis],
        context: str = "",
    ) -> list[ClinicalSafetyFlag]:
        """
        Detect clinical safety flags from evidence and hypotheses.
        
        Args:
            evidence: Aggregated perception evidence.
            hypotheses: Disease hypotheses.
            context: Clinical context (e.g., "chest_ct", "head_ct").
        
        Returns:
            List of clinical safety flags.
        """
        flags = []
        
        # Check for pattern-based flags
        flags.extend(self._check_pattern_flags(evidence, context))
        
        # Check for hypothesis-based flags
        flags.extend(self._check_hypothesis_flags(hypotheses))
        
        # Check for image quality flags
        flags.extend(self._check_quality_flags(evidence))
        
        # Sort by severity (critical first)
        severity_order = {
            SafetyFlagSeverity.CRITICAL: 0,
            SafetyFlagSeverity.HIGH: 1,
            SafetyFlagSeverity.MODERATE: 2,
            SafetyFlagSeverity.LOW: 3,
        }
        flags.sort(key=lambda f: severity_order.get(f.severity, 4))
        
        return flags
    
    def _check_pattern_flags(
        self,
        evidence: CaseEvidenceBundle,
        context: str,
    ) -> list[ClinicalSafetyFlag]:
        """Check for pattern-based safety flags."""
        flags = []
        patterns = self.SAFETY_PATTERNS.get(context, [])
        
        for pattern in patterns:
            if self._pattern_matches(pattern, evidence):
                flag = ClinicalSafetyFlag(
                    flag_type=pattern.flag_type,
                    severity=pattern.severity,
                    description=pattern.description,
                    finding_pattern=pattern.name,
                    recommended_action=pattern.recommended_action,
                    escalation_required=pattern.escalation_required,
                    time_sensitivity=pattern.time_sensitivity,
                )
                flags.append(flag)
        
        return flags
    
    def _pattern_matches(
        self,
        pattern: SafetyPattern,
        evidence: CaseEvidenceBundle,
    ) -> bool:
        """Check if a safety pattern matches the evidence."""
        # Check global flags
        global_flags = evidence.global_features.get("safety_indicators", {})
        if pattern.name in global_flags:
            return global_flags[pattern.name]
        
        # Check lesion-level flags
        for lesion in evidence.lesion_features:
            if lesion.get("safety_pattern") == pattern.name:
                return True
        
        return False
    
    def _check_hypothesis_flags(
        self,
        hypotheses: list[DiseaseHypothesis],
    ) -> list[ClinicalSafetyFlag]:
        """Check for flags based on high-confidence urgent diagnoses."""
        flags = []
        
        urgent_diseases = {
            "massive_pe": "Pulmonary embolism",
            "aortic_dissection": "Aortic dissection",
            "intracranial_hemorrhage": "Intracranial hemorrhage",
            "bowel_perforation": "Bowel perforation",
        }
        
        for hyp in hypotheses:
            if hyp.probability >= 0.7 and hyp.disease_code in urgent_diseases:
                flag = ClinicalSafetyFlag(
                    flag_type=SafetyFlagType.URGENT_FINDING,
                    severity=SafetyFlagSeverity.CRITICAL,
                    description=f"High probability ({hyp.probability:.0%}) of {urgent_diseases[hyp.disease_code]}",
                    finding_pattern=hyp.disease_code,
                    evidence_ids=hyp.evidence_ids,
                    recommended_action="Immediate clinical notification recommended",
                    escalation_required=True,
                    time_sensitivity="immediate",
                )
                flags.append(flag)
        
        return flags
    
    def _check_quality_flags(
        self,
        evidence: CaseEvidenceBundle,
    ) -> list[ClinicalSafetyFlag]:
        """Check for image quality-related flags."""
        flags = []
        quality_patterns = self.SAFETY_PATTERNS.get("image_quality", [])
        
        # Check image quality
        quality = evidence.metadata_features.get("image_quality", 1.0)
        motion = evidence.metadata_features.get("motion_artifact_severity", 0.0)
        incomplete = evidence.metadata_features.get("incomplete_coverage", False)
        
        if quality < 0.5 or motion > 0.7:
            for pattern in quality_patterns:
                if pattern.name == "severe_motion":
                    flag = ClinicalSafetyFlag(
                        flag_type=pattern.flag_type,
                        severity=pattern.severity,
                        description=pattern.description,
                        finding_pattern=pattern.name,
                        recommended_action=pattern.recommended_action,
                    )
                    flags.append(flag)
                    break
        
        if incomplete:
            for pattern in quality_patterns:
                if pattern.name == "incomplete_coverage":
                    flag = ClinicalSafetyFlag(
                        flag_type=pattern.flag_type,
                        severity=pattern.severity,
                        description=pattern.description,
                        finding_pattern=pattern.name,
                        recommended_action=pattern.recommended_action,
                    )
                    flags.append(flag)
                    break
        
        return flags
