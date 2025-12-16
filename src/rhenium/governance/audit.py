"""Audit logging for governance and compliance."""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
import json


class AuditEventType(str, Enum):
    PIPELINE_START = "pipeline_start"
    PIPELINE_COMPLETE = "pipeline_complete"
    PIPELINE_ERROR = "pipeline_error"
    MODEL_LOAD = "model_load"
    PREDICTION = "prediction"
    DATA_ACCESS = "data_access"


@dataclass
class AuditEvent:
    """Single audit event."""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime = field(default_factory=datetime.utcnow)
    study_uid: str | None = None
    user_id: str | None = None
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.event_id, "type": self.event_type.value,
            "timestamp": self.timestamp.isoformat(),
            "study_uid": self.study_uid, "details": self.details,
        }


class AuditLogger:
    """Immutable audit log."""

    def __init__(self, log_dir: Path | None = None):
        self.log_dir = log_dir
        self.events: list[AuditEvent] = []

    def log(self, event: AuditEvent) -> None:
        self.events.append(event)
        if self.log_dir:
            self._persist(event)

    def _persist(self, event: AuditEvent) -> None:
        log_file = self.log_dir / f"audit_{event.timestamp.strftime('%Y%m%d')}.jsonl"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with open(log_file, "a") as f:
            f.write(json.dumps(event.to_dict()) + "\n")

    def query(
        self,
        event_type: AuditEventType | None = None,
        study_uid: str | None = None,
    ) -> list[AuditEvent]:
        results = self.events
        if event_type:
            results = [e for e in results if e.event_type == event_type]
        if study_uid:
            results = [e for e in results if e.study_uid == study_uid]
        return results
