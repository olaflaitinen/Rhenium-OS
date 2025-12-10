# Copyright (c) 2025 Skolyn LLC. All rights reserved.
# SPDX-License-Identifier: EUPL-1.1

"""Tests for evaluation metrics."""

import pytest
import numpy as np

from rhenium.evaluation.metrics import (
    dice_score,
    iou_score,
    hausdorff_distance,
    psnr,
    compute_sensitivity_specificity,
)


class TestSegmentationMetrics:
    """Test segmentation metrics."""

    def test_dice_perfect(self):
        """Test dice with perfect overlap."""
        mask = np.ones((10, 10), dtype=bool)
        assert dice_score(mask, mask) == pytest.approx(1.0)

    def test_dice_no_overlap(self):
        """Test dice with no overlap."""
        pred = np.zeros((10, 10), dtype=bool)
        pred[:5, :] = True
        target = np.zeros((10, 10), dtype=bool)
        target[5:, :] = True
        assert dice_score(pred, target) == pytest.approx(0.0)

    def test_iou(self, sample_mask):
        """Test IoU computation."""
        score = iou_score(sample_mask, sample_mask)
        assert score == pytest.approx(1.0)


class TestClassificationMetrics:
    """Test classification metrics."""

    def test_sensitivity_specificity(self):
        """Test sensitivity and specificity."""
        y_true = np.array([1, 1, 0, 0, 1, 0])
        y_pred = np.array([1, 0, 0, 1, 1, 0])

        sens, spec = compute_sensitivity_specificity(y_true, y_pred)
        assert 0 <= sens <= 1
        assert 0 <= spec <= 1


class TestImageQualityMetrics:
    """Test image quality metrics."""

    def test_psnr(self, sample_image):
        """Test PSNR computation."""
        noisy = sample_image + np.random.randn(*sample_image.shape) * 0.1
        psnr_val = psnr(sample_image, sample_image)
        assert psnr_val > 30  # Same image, high PSNR
