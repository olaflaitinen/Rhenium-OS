"""Risk Register for tracking and mitigating risks."""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, Literal
import yaml


class RiskCategory(str, Enum):
    TECHNICAL = "technical"
    PRIVACY = "privacy"
    REGULATORY = "regulatory"
    BIAS = "bias"
    SAFETY = "safety"


class RiskStatus(str, Enum):
    OPEN = "open"
    MITIGATED = "mitigated"
    MONITORING = "monitoring"
    CLOSED = "closed"


@dataclass
class Risk:
    """Individual risk entry."""
    id: str
    title: str
    category: RiskCategory
    description: str = ""
    likelihood: Literal["low", "medium", "high"] = "medium"
    impact: Literal["low", "medium", "high", "critical"] = "medium"
    mitigation: list[str] = field(default_factory=list)
    status: RiskStatus = RiskStatus.OPEN
    owner: str = ""

    def risk_score(self) -> int:
        scores = {"low": 1, "medium": 2, "high": 3, "critical": 4}
        return scores[self.likelihood] * scores.get(self.impact, 2)


@dataclass
class RiskRegister:
    """Collection of risks."""
    risks: list[Risk] = field(default_factory=list)

    def add(self, risk: Risk) -> None:
        self.risks.append(risk)

    def get_high_risks(self) -> list[Risk]:
        return [r for r in self.risks if r.risk_score() >= 6]

    def save(self, path: Path) -> None:
        data = {"risks": [
            {"id": r.id, "title": r.title, "category": r.category.value,
             "likelihood": r.likelihood, "impact": r.impact,
             "mitigation": r.mitigation, "status": r.status.value}
            for r in self.risks
        ]}
        with open(path, "w") as f:
            yaml.dump(data, f)
