# Copyright (c) 2025 Skolyn LLC. All rights reserved.

"""
Rhenium OS Output Manager
=========================

Orchestrates the collection and persistence of system outputs.
Attached to the PipelineContext to provide a unified reporting API.
"""

from __future__ import annotations

import logging
from typing import Any, List, Optional
import json
from pathlib import Path

from rhenium.outputs.types import (
    EvidenceDossier, StudyOutput, OrganOutput, RegionOutput,
    PixelMapOutput, Measurement, Finding
)

logger = logging.getLogger(__name__)

class OutputManager:
    """
    Manages the creation and storage of an Evidence Dossier.
    """
    
    def __init__(self, case_id: str, modality: str, output_dir: str = "outputs"):
        self.output_dir = Path(output_dir) / case_id
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize an empty dossier
        self.dossier = EvidenceDossier(
            case_id=case_id,
            study_output=StudyOutput(
                patient_id="Unknown",
                study_date="Unknown",
                modality=modality,
                protocol="Unknown"
            )
        )
        
    def set_patient_info(self, patient_id: str, study_date: str, protocol: str):
        self.dossier.study_output.patient_id = patient_id
        self.dossier.study_output.study_date = study_date
        self.dossier.study_output.protocol = protocol

    def add_organ_output(self, organ: OrganOutput):
        self.dossier.study_output.add_organ(organ)

    def add_global_finding(self, finding: Finding):
        self.dossier.study_output.add_finding(finding)

    def add_measurement(self, metric: Measurement):
        self.dossier.study_output.global_metrics.append(metric)

    def add_map(self, map_output: PixelMapOutput):
        self.dossier.maps.append(map_output)

    def add_xai_overlay(self, overlay: PixelMapOutput):
        self.dossier.xai_overlays.append(overlay)

    def add_log(self, message: str):
        self.dossier.logs.append(message)

    def add_model_info(self, info: dict):
        self.dossier.model_info.append(info)

    def save_dossier(self, filename: str = "evidence_dossier.json"):
        """Save the full dossier to JSON."""
        file_path = self.output_dir / filename
        try:
            with open(file_path, 'w') as f:
                json.dump(self.dossier.to_dict(), f, indent=2, default=str)
            logger.info(f"Saved Evidence Dossier to {file_path}")
            return str(file_path)
        except Exception as e:
            logger.error(f"Failed to save dossier: {e}")
            raise

    def get_summary(self) -> str:
        """Generate a brief text summary."""
        study = self.dossier.study_output
        summary = [
            f"Case ID: {self.dossier.case_id}",
            f"Modality: {study.modality}",
            f"Organs Analyzed: {len(study.organs)}",
            f"Findings: {len(study.global_findings)} global",
            f"Metrics: {len(study.global_metrics)} global"
        ]
        return "\n".join(summary)
