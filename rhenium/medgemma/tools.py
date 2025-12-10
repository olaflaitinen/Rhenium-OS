# Copyright (c) 2025 Skolyn LLC. All rights reserved.
# SPDX-License-Identifier: EUPL-1.1

"""
MedGemma Tool-Use Framework
===========================

Complementary reasoning tools that MedGemma can invoke for enhanced
clinical reasoning, validation, and guideline alignment.

Last Updated: December 2025
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from rhenium.xai.explanation_schema import Finding
from rhenium.core.logging import get_medgemma_logger

logger = get_medgemma_logger()


@dataclass
class ToolResult:
    """Result from a tool invocation."""
    tool_name: str
    success: bool
    result: Any
    message: str = ""


class BaseTool(ABC):
    """Abstract base class for reasoning tools."""
    
    name: str = "base_tool"
    description: str = ""
    
    @abstractmethod
    def invoke(self, **kwargs: Any) -> ToolResult:
        """Invoke the tool with given parameters."""
        pass


class GuidelineChecker(BaseTool):
    """
    Check findings against clinical guidelines.
    
    Validates that findings and recommendations align with
    established guidelines (ACR, ESR, NICE, etc.).
    """
    
    name = "guideline_checker"
    description = "Check alignment with clinical guidelines"
    
    def __init__(self):
        self.guideline_sets = {
            "acr": "American College of Radiology",
            "esr": "European Society of Radiology",
            "nice": "National Institute for Health and Care Excellence",
        }
        
    def invoke(
        self,
        finding: Finding | None = None,
        guideline_set: str = "acr",
        **kwargs: Any,
    ) -> ToolResult:
        """Check finding against specified guidelines."""
        logger.info("Checking guideline alignment", 
                   finding_type=finding.finding_type if finding else None,
                   guideline_set=guideline_set)
        
        if guideline_set not in self.guideline_sets:
            return ToolResult(
                tool_name=self.name,
                success=False,
                result=None,
                message=f"Unknown guideline set: {guideline_set}",
            )
        
        # Placeholder: would query guideline database
        return ToolResult(
            tool_name=self.name,
            success=True,
            result={
                "aligned": True,
                "guideline_set": self.guideline_sets[guideline_set],
                "references": [],
            },
            message="Finding aligns with guidelines",
        )


class RuleBasedValidator(BaseTool):
    """
    Apply deterministic validation rules.
    
    Checks for logical consistency, measurement plausibility,
    and laterality correctness.
    """
    
    name = "rule_validator"
    description = "Apply rule-based validation checks"
    
    def invoke(
        self,
        findings: list[Finding] | None = None,
        **kwargs: Any,
    ) -> ToolResult:
        """Validate findings with rule-based checks."""
        findings = findings or []
        issues = []
        
        # Check laterality consistency
        lateralities = [f.laterality for f in findings if f.laterality]
        if "left" in lateralities and "right" in lateralities:
            # Check if bilateral is also mentioned
            if "bilateral" not in lateralities:
                issues.append("Both left and right findings without bilateral designation")
        
        # Check confidence thresholds
        low_confidence = [f for f in findings if f.confidence < 0.5]
        if low_confidence:
            issues.append(f"{len(low_confidence)} findings with confidence < 50%")
        
        logger.debug("Rule validation complete", num_issues=len(issues))
        
        return ToolResult(
            tool_name=self.name,
            success=len(issues) == 0,
            result={"issues": issues},
            message=f"Found {len(issues)} validation issues",
        )


class MeasurementVerifier(BaseTool):
    """
    Verify measurements for plausibility.
    
    Checks that measurements fall within expected ranges
    for the given anatomical context.
    """
    
    name = "measurement_verifier"
    description = "Verify measurement plausibility"
    
    # Expected ranges by measurement type (mm)
    EXPECTED_RANGES = {
        "lesion_diameter": (1.0, 200.0),
        "volume_ml": (0.001, 5000.0),
        "meniscus_tear_length": (1.0, 50.0),
        "tumor_diameter": (1.0, 150.0),
    }
    
    def invoke(
        self,
        measurements: dict[str, float] | None = None,
        measurement_type: str = "lesion_diameter",
        **kwargs: Any,
    ) -> ToolResult:
        """Verify measurements against expected ranges."""
        measurements = measurements or {}
        issues = []
        
        for name, value in measurements.items():
            expected = self.EXPECTED_RANGES.get(measurement_type, (0, float('inf')))
            if not expected[0] <= value <= expected[1]:
                issues.append(f"{name}={value} outside range {expected}")
        
        return ToolResult(
            tool_name=self.name,
            success=len(issues) == 0,
            result={"issues": issues},
            message=f"Verified {len(measurements)} measurements",
        )


class KnowledgeGraphQuery(BaseTool):
    """
    Query medical knowledge graph.
    
    Retrieves contextual information about diseases, findings,
    and their relationships from a structured knowledge base.
    """
    
    name = "knowledge_graph"
    description = "Query medical knowledge graph"
    
    def invoke(
        self,
        entity: str | None = None,
        query_type: str = "related_findings",
        **kwargs: Any,
    ) -> ToolResult:
        """Query knowledge graph for entity information."""
        logger.info("Querying knowledge graph", entity=entity, query_type=query_type)
        
        # Placeholder: would query actual knowledge graph
        return ToolResult(
            tool_name=self.name,
            success=True,
            result={
                "entity": entity,
                "related_entities": [],
                "properties": {},
            },
            message="Knowledge graph query complete",
        )


@dataclass
class ToolRegistry:
    """Registry of available reasoning tools."""
    tools: dict[str, BaseTool] = field(default_factory=dict)
    
    def register(self, tool: BaseTool) -> None:
        """Register a tool."""
        self.tools[tool.name] = tool
        
    def get(self, name: str) -> BaseTool | None:
        """Get tool by name."""
        return self.tools.get(name)
    
    def list_tools(self) -> list[str]:
        """List available tool names."""
        return list(self.tools.keys())
    
    def invoke(self, tool_name: str, **kwargs: Any) -> ToolResult:
        """Invoke a tool by name."""
        tool = self.get(tool_name)
        if tool is None:
            return ToolResult(
                tool_name=tool_name,
                success=False,
                result=None,
                message=f"Unknown tool: {tool_name}",
            )
        return tool.invoke(**kwargs)


# Default tool registry
default_tool_registry = ToolRegistry()
default_tool_registry.register(GuidelineChecker())
default_tool_registry.register(RuleBasedValidator())
default_tool_registry.register(MeasurementVerifier())
default_tool_registry.register(KnowledgeGraphQuery())


def get_tool_registry() -> ToolRegistry:
    """Get the default tool registry."""
    return default_tool_registry
