import pytest
import torch

from mammography.src.data import transforms


@pytest.mark.parametrize(argnames=["height", "width"], argvalues=[(32, 32), (64, 32), (1024, 32), (32, 1024)])
def test_crop_center_right(height: int, width: int) -> None:
    x = torch.randint(low=0, high=255, dtype=torch.uint8, size=(1, 512, 512))
    cropper = transforms.CropCenterRight(height=height, width=width)
    y: torch.Tensor = cropper(x)
    assert y.ndim == x.ndim
    assert y.shape[0] == x.shape[0]
    assert y.shape[1] == min(x.shape[1], cropper.height)
    assert y.shape[2] == min(x.shape[2], cropper.width)
