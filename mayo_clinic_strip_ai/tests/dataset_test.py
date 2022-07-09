import numpy as np
from PIL import Image
from pytest import fixture
import torch

from mayo_clinic_strip_ai import dataset


@fixture
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


def test_load_tif(tif_img_path):
    img = dataset.load_tif(tif_img_path)
    assert img.ndim == 3
    assert img.dtype == torch.uint8


def test_augmenter():
    aug = dataset.Augmenter()
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


# def test_tif_dataset(tif_img_path):
#     metadata = pd.DataFrame
#     ds = dataset.TifDataset()
