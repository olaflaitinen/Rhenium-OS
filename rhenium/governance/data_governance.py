# Copyright (c) 2025 Skolyn LLC. All rights reserved.
# SPDX-License-Identifier: EUPL-1.1

"""Data governance utilities."""

from __future__ import annotations
from dataclasses import dataclass
from enum import Enum


class AccessLevel(str, Enum):
    """Data access levels."""
    PUBLIC = "public"
    INTERNAL = "internal"
    RESTRICTED = "restricted"
    PHI = "phi"


@dataclass
class DataAccessPolicy:
    """Policy for data access."""
    resource_type: str
    access_level: AccessLevel
    allowed_roles: list[str]
    requires_audit: bool = True


def anonymize_patient_id(patient_id: str) -> str:
    """Generate anonymized patient identifier."""
    import hashlib
    return hashlib.sha256(patient_id.encode()).hexdigest()[:16]
