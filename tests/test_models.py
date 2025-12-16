"""Model tests."""

import pytest
import torch
from rhenium.perception.segmentation import UNet3D
from rhenium.perception.detection import CenterNet3D
from rhenium.perception.classification import ResNet3D


class TestUNet3D:
    def test_forward(self, sample_tensor):
        model = UNet3D(in_channels=1, out_channels=2, features=[16, 32])
        model.eval()
        with torch.no_grad():
            output = model(sample_tensor)
        assert output.shape == (1, 2, 32, 64, 64)

    def test_predict(self, sample_tensor):
        model = UNet3D(in_channels=1, out_channels=2, features=[16, 32])
        model.eval()
        pred = model.predict(sample_tensor)
        assert pred.shape == (1, 32, 64, 64)


class TestCenterNet3D:
    def test_forward(self, sample_tensor):
        model = CenterNet3D(in_channels=1, num_classes=1, features=[16, 32])
        model.eval()
        with torch.no_grad():
            output = model(sample_tensor)
        assert "heatmap" in output
        assert "size" in output


class TestResNet3D:
    def test_forward(self, sample_tensor):
        model = ResNet3D(in_channels=1, num_classes=3, base_features=16, layers=[1, 1, 1, 1])
        model.eval()
        with torch.no_grad():
            output = model(sample_tensor)
        assert output.shape == (1, 3)
