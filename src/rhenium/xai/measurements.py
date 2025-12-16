"""Quantitative measurements from segmentation masks."""

from __future__ import annotations
import numpy as np
from rhenium.xai.evidence_dossier import QuantitativeEvidence


class MeasurementExtractor:
    """Extract quantitative measurements from segmentations."""

    def extract_volume(
        self,
        mask: np.ndarray,
        spacing: tuple[float, float, float],
    ) -> QuantitativeEvidence:
        """Compute volume in cubic millimeters."""
        voxel_volume = np.prod(spacing)
        num_voxels = np.sum(mask > 0)
        volume_mm3 = float(num_voxels * voxel_volume)
        return QuantitativeEvidence(
            evidence_id="vol_001", evidence_type="volume",
            value=volume_mm3, unit="mm³",
        )

    def extract_max_diameter(
        self,
        mask: np.ndarray,
        spacing: tuple[float, float, float],
    ) -> QuantitativeEvidence:
        """Compute maximum diameter."""
        coords = np.argwhere(mask > 0)
        if len(coords) == 0:
            return QuantitativeEvidence(
                evidence_id="diam_001", evidence_type="diameter", value=0.0, unit="mm"
            )
        coords_mm = coords * np.array(spacing)
        dists = np.sqrt(((coords_mm[:, None] - coords_mm[None, :]) ** 2).sum(axis=-1))
        max_diam = float(dists.max())
        return QuantitativeEvidence(
            evidence_id="diam_001", evidence_type="diameter", value=max_diam, unit="mm",
        )

    def extract_all(
        self,
        mask: np.ndarray,
        spacing: tuple[float, float, float],
    ) -> list[QuantitativeEvidence]:
        """Extract all measurements."""
        return [
            self.extract_volume(mask, spacing),
            self.extract_max_diameter(mask, spacing),
        ]
