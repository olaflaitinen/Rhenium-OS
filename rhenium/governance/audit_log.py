# Copyright (c) 2025 Skolyn LLC. All rights reserved.
# SPDX-License-Identifier: EUPL-1.1

"""
Audit Logging
=============

Regulatory-compliant audit logging for pipeline runs and model inferences.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from uuid import uuid4

from rhenium.core.config import get_settings
from rhenium.core.logging import get_governance_logger

logger = get_governance_logger()


@dataclass
class AuditEntry:
    """Single audit log entry."""
    entry_id: str = field(default_factory=lambda: uuid4().hex)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    event_type: str = ""  # pipeline_run, model_inference, data_access
    pipeline_name: str = ""
    pipeline_version: str = ""
    model_names: list[str] = field(default_factory=list)
    model_versions: list[str] = field(default_factory=list)
    data_id: str = ""  # Pseudonymized
    config_hash: str = ""
    outcome: str = ""  # success, failure, partial
    error_message: str = ""
    duration_seconds: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["timestamp"] = self.timestamp.isoformat()
        return result


class AuditLogger:
    """Centralized audit logger."""

    def __init__(self, log_dir: Path | None = None):
        settings = get_settings()
        self.log_dir = log_dir or settings.audit_log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def log(self, entry: AuditEntry) -> None:
        """Write audit entry."""
        log_file = self.log_dir / f"audit_{datetime.now().strftime('%Y%m%d')}.jsonl"
        with open(log_file, "a") as f:
            f.write(json.dumps(entry.to_dict()) + "\n")
        logger.info("Audit entry logged", entry_id=entry.entry_id)

    def query(self, start_date: datetime, end_date: datetime) -> list[AuditEntry]:
        """Query audit entries by date range."""
        # Simplified - would implement full query logic
        return []


def log_pipeline_run(
    pipeline_name: str,
    pipeline_version: str,
    data_id: str,
    outcome: str,
    duration: float,
    config: dict[str, Any] | None = None,
) -> None:
    """Convenience function to log pipeline run."""
    config_hash = hashlib.sha256(json.dumps(config or {}, sort_keys=True).encode()).hexdigest()[:16]

    entry = AuditEntry(
        event_type="pipeline_run",
        pipeline_name=pipeline_name,
        pipeline_version=pipeline_version,
        data_id=data_id,
        config_hash=config_hash,
        outcome=outcome,
        duration_seconds=duration,
    )

    AuditLogger().log(entry)
