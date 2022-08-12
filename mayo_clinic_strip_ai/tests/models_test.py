import pytest
import torch
from torchvision import models

from mayo_clinic_strip_ai import model


def test_normalizer():
    normalizer = model.Normalizer()
    x = torch.testing.make_tensor(
        (2, 3, 512, 512),
        dtype=torch.uint8,
        device="cpu",
        low=0,
        high=255,
    )
    y: torch.Tensor = normalizer(x)
    assert y.dtype is torch.float32
    assert y.min().item() >= 0.0
    assert y.max().item() <= 1.0


@pytest.mark.parametrize("backbone", ["densenet121"])
def test_feature_extractor(backbone: str):
    fe = model.FeatureExtractor(backbone=getattr(models, backbone)())
    fe.train()
    x = torch.testing.make_tensor((2, 3, 512, 512), dtype=torch.float32, device="cpu")
    y: torch.Tensor = fe(x)
    assert y.shape[0] == x.shape[0]
