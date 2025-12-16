"""Evaluation metrics tests."""

import pytest
import numpy as np
from rhenium.evaluation import dice_score, iou_score, psnr, ssim


class TestSegmentationMetrics:
    def test_dice_identical(self, sample_mask):
        score = dice_score(sample_mask, sample_mask)
        assert score == pytest.approx(1.0, abs=1e-5)

    def test_dice_zero_overlap(self):
        pred = np.zeros((10, 10, 10))
        target = np.ones((10, 10, 10))
        score = dice_score(pred, target)
        assert score == pytest.approx(0.0, abs=1e-5)

    def test_iou_identical(self, sample_mask):
        score = iou_score(sample_mask, sample_mask)
        assert score == pytest.approx(1.0, abs=1e-5)


class TestReconstructionMetrics:
    def test_psnr_identical(self, sample_volume):
        score = psnr(sample_volume, sample_volume)
        assert score == float('inf')

    def test_psnr_different(self, sample_volume):
        noisy = sample_volume + np.random.randn(*sample_volume.shape) * 0.1
        score = psnr(sample_volume, noisy)
        assert score > 0

    def test_ssim_identical(self, sample_volume):
        score = ssim(sample_volume, sample_volume)
        assert score == pytest.approx(1.0, abs=1e-5)
