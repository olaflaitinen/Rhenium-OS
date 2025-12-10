# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Rhenium OS Output Data Models
=============================

Structured definitions for all system outputs to ensure consistency,
auditability, and rich reporting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import time
import uuid

# =============================================================================
# Core Types
# =============================================================================

class OutputCategory(str, Enum):
    PIXEL_MAP = "pixel_map"
    REGION_ROI = "region_roi"
    ORGAN_METRIC = "organ_metric"
    STUDY_METRIC = "study_metric"
    MODEL_INFO = "model_info"
    SYSTEM_LOG = "system_log"
    XAI_ARTIFACT = "xai_artifact"


@dataclass
class Unit:
    """Measurement unit definition."""
    code: str
    description: str


@dataclass
class ConfidenceInterval:
    low: float
    high: float
    level: float = 0.95


@dataclass
class Finding:
    """Clinical finding (categorical)."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    label: str = "Unknown"
    probability: float = 0.0
    severity: str = "Unknown"
    location: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Measurement:
    """Quantitative measurement."""
    name: str
    value: float
    unit: str
    confidence_interval: Optional[ConfidenceInterval] = None
    reference_range: Optional[str] = None
    source_method: str = "AI Model"


# =============================================================================
# Level 1: Pixel / Voxel Maps
# =============================================================================

@dataclass
class PixelMapOutput:
    """Reference to a generated image map (mask, heatmap, etc.)."""
    id: str
    type: str  # Segmentation, Probability, Attention, Parameter
    description: str
    file_path: str  # Relative path to storage
    dimensions: List[int]
    resolution: List[float]


# =============================================================================
# Level 2: Region / Lesion
# =============================================================================

@dataclass
class RegionOutput:
    """Output for a specific ROI/Lesion."""
    id: str
    type: str  # Nodule, Fracture, Mass
    location: Dict[str, Any]  # BBox, Centroid
    measurements: List[Measurement] = field(default_factory=list)
    findings: List[Finding] = field(default_factory=list)
    associated_maps: List[str] = field(default_factory=list)  # Map IDs


# =============================================================================
# Level 3: Organ System
# =============================================================================

@dataclass
class OrganOutput:
    """Aggregated outputs for an organ."""
    organ_name: str
    status: str  # Normal, Abnormal, Indeterminate
    volumetry: Optional[Measurement] = None
    metrics: List[Measurement] = field(default_factory=list)
    regions: List[RegionOutput] = field(default_factory=list)


# =============================================================================
# Level 4: Study / Patient
# =============================================================================

@dataclass
class StudyOutput:
    """Study-level summary."""
    patient_id: str
    study_date: str
    modality: str
    protocol: str
    
    # Aggregations
    organs: List[OrganOutput] = field(default_factory=list)
    global_findings: List[Finding] = field(default_factory=list)
    global_metrics: List[Measurement] = field(default_factory=list)
    
    # Dossier Metadata
    generated_at: float = field(default_factory=time.time)
    rhenium_version: str = "2025.12"
    
    def add_organ(self, organ: OrganOutput):
        self.organs.append(organ)

    def add_finding(self, finding: Finding):
        self.global_findings.append(finding)


# =============================================================================
# Evidence Dossier (Root Object)
# =============================================================================

@dataclass
class EvidenceDossier:
    """
    Master container for all evidence generated during a pipeline run.
    This is the object that gets serialized to JSON.
    """
    case_id: str
    study_output: StudyOutput
    
    # Artifacts
    maps: List[PixelMapOutput] = field(default_factory=list)
    xai_overlays: List[PixelMapOutput] = field(default_factory=list)
    
    # Operations
    logs: List[str] = field(default_factory=list)
    model_info: List[Dict[str, Any]] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        from dataclasses import asdict
        return asdict(self)
