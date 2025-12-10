# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""Risk tracking for regulatory compliance."""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


class RiskSeverity(str, Enum):
    """Risk severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class RiskEntry:
    """Recorded risk or failure mode."""
    risk_id: str
    title: str
    description: str
    severity: RiskSeverity
    component: str
    mitigation: str = ""
    status: str = "open"  # open, mitigated, accepted
    identified_date: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class RiskTracker:
    """Track known risks and failure modes."""

    def __init__(self) -> None:
        self.risks: list[RiskEntry] = []

    def add_risk(self, risk: RiskEntry) -> None:
        self.risks.append(risk)

    def get_open_risks(self) -> list[RiskEntry]:
        return [r for r in self.risks if r.status == "open"]

    def get_by_severity(self, severity: RiskSeverity) -> list[RiskEntry]:
        return [r for r in self.risks if r.severity == severity]
