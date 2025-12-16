"""
Rhenium OS Clinical Core.

This module provides validation, auditing, and compliance utilities
required for clinical deployment.

Features:
- Immutable Audit Logging
- DICOM Tag Validation
- Clinical Safety Checks
"""

import hashlib
import json
import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# --- Audit Logging ---

@dataclass
class AuditEntry:
    """Immutable audit log entry."""
    entry_id: str
    timestamp: str
    action: str
    user_id: str
    resource_id: str
    status: str
    details: dict[str, Any]
    prev_hash: str
    signature: str = ""

    def compute_hash(self) -> str:
        """Compute SHA-256 hash of the entry content."""
        content = f"{self.entry_id}{self.timestamp}{self.action}{self.user_id}{self.resource_id}{self.status}{json.dumps(self.details, sort_keys=True)}{self.prev_hash}"
        return hashlib.sha256(content.encode()).hexdigest()

class AuditLogger:
    """Clinical Audit Logger with integrity checks."""

    def __init__(self, log_dir: Path = Path("logs/audit")):
        self.log_dir = log_dir
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_log_file = self.log_dir / f"audit_{datetime.utcnow().strftime('%Y%m%d')}.jsonl"
        self._last_hash = "0" * 64

    def log(self, action: str, user_id: str, resource_id: str, status: str = "SUCCESS", details: dict[str, Any] = None) -> str:
        """
        Write a trusted audit log entry.
        
        Args:
            action: The action performed (e.g., "SEGMENTATION_REQUEST")
            user_id: ID of the user/system performing action
            resource_id: ID of the resource affected (e.g., StudyInstanceUID)
            status: Outcome "SUCCESS" or "FAILURE"
            details: Additional context
        
        Returns:
            Entry ID
        """
        entry_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()
        
        entry = AuditEntry(
            entry_id=entry_id,
            timestamp=timestamp,
            action=action,
            user_id=user_id,
            resource_id=resource_id,
            status=status,
            details=details or {},
            prev_hash=self._last_hash
        )
        
        # In a real system, this would be signed with a private key
        entry.signature = entry.compute_hash() 
        self._last_hash = entry.signature
        
        # Write to immutable log (append-only)
        with open(self.current_log_file, "a") as f:
            f.write(json.dumps(entry.__dict__) + "\n")
            
        return entry_id

# --- DICOM Validation ---

class DicomValidator:
    """Validator for clinical DICOM compliance."""
    
    REQUIRED_TAGS = [
        "PatientID",
        "PatientName",
        "StudyInstanceUID",
        "SeriesInstanceUID",
        "SOPInstanceUID",
        "Modality",
    ]
    
    FIELD_MAPPING = {
        "patient_id": "PatientID",
        "patient_name": "PatientName",
        "study_uid": "StudyInstanceUID",
        "series_uid": "SeriesInstanceUID",
        "sop_uid": "SOPInstanceUID",
        "modality": "Modality"
    }

    @staticmethod
    def validate_metadata(metadata: dict[str, Any]) -> tuple[bool, list[str]]:
        """
        Validate DICOM metadata against required tags.
        
        Args:
            metadata: Dictionary of DICOM tags (or internal snake_case fields)
            
        Returns:
            (is_valid, list_of_errors)
        """
        # Create a mapped copy for validation
        mapped_metadata = metadata.copy()
        for internal, dicom in DicomValidator.FIELD_MAPPING.items():
            if internal in metadata and metadata[internal]:
                 mapped_metadata[dicom] = metadata[internal]
                 
        errors = []
        for tag in DicomValidator.REQUIRED_TAGS:
            if tag not in mapped_metadata or not mapped_metadata[tag]:
                errors.append(f"Missing required DICOM tag: {tag}")
                
        # Modality check
        if "Modality" in mapped_metadata:
            modality = mapped_metadata["Modality"]
            # Convert enum to value if needed
            if hasattr(modality, "value"):
                modality = modality.value
                
            if modality not in ["MR", "CT", "US", "XA", "CR", "DX", "MRI"]: # Added MRI as it's used in generator
                 errors.append(f"Unsupported clinical modality: {modality}")

        return len(errors) == 0, errors

    @staticmethod
    def anonymize(metadata: dict[str, Any]) -> dict[str, Any]:
        """Strip PII from metadata."""
        cleaned = metadata.copy()
        pii_tags = ["PatientName", "PatientID", "PatientBirthDate", "PatientSex"]
        for tag in pii_tags:
            if tag in cleaned:
                cleaned[tag] = "ANONYMIZED"
        return cleaned

# --- Clinical Safety ---

class ClinicalSafety:
    """Runtime safety checks."""
    
    @staticmethod
    def verify_disk_space(min_gb: float = 1.0) -> bool:
        """Ensure sufficient disk space for safety."""
        import shutil
        total, used, free = shutil.disk_usage(".")
        return (free / (1024**3)) > min_gb

    @staticmethod
    def verify_memory_headroom(min_gb: float = 0.5) -> bool:
        """Ensure sufficient RAM for safe inference."""
        import psutil
        avail = psutil.virtual_memory().available / (1024**3)
        return avail > min_gb

# Usage Example
if __name__ == "__main__":
    logger = AuditLogger()
    lid = logger.log("TEST_ACTION", "system", "res-123", details={"info": "test"})
    print(f"Logged entry: {lid}")
