import numpy as np
from PIL import Image
import pytest
import torch

from mayo_clinic_strip_ai import dataset


@pytest.fixture
def tif_img_path(tmpdir):
    img_path = str(tmpdir / "img.tif")
    arr = np.random.randint(
        low=0,
        high=255,
        dtype=np.uint8,
        size=(512, 512, 3),
    )
    img = Image.fromarray(arr)
    img.save(img_path, format="TIFF")
    return img_path


@pytest.mark.parametrize("training", [True, False])
def test_tifdataset(tif_img_path: str, training: bool):
    aug = dataset.TifDataset(training=training)
    x = torch.testing.make_tensor(
        (3, 1024, 1024),
        low=0,
        high=255,
        dtype=torch.uint8,
        device="cpu",
    )
    y: torch.Tensor = aug(x)
    assert y.dtype == torch.uint8
    assert y.shape == (3, 512, 512)
