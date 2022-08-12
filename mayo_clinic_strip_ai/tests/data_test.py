from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from mayo_clinic_strip_ai import data


@pytest.fixture
def tif_img_path(tmp_path: Path) -> Path:
    img_path = tmp_path / "img.tif"
    arr = np.random.randint(
        low=0,
        high=255,
        dtype=np.uint8,
        size=(512, 512, 3),
    )
    cv2.imwrite(filename=img_path.as_posix(), img=arr)
    return img_path


@pytest.fixture
def outline_path(tmp_path: Path) -> Path:
    arr = np.random.randint(
        low=0,
        high=30,
        dtype=np.int32,
        size=(10, 10),
    )
    np.save(tmp_path / "0", arr, allow_pickle=False)
    return tmp_path / "0.npy"


@pytest.mark.skip("Requires dataset")
@pytest.mark.parametrize("training", [True, False])
def test_roidataset(training: bool):
    rois = pd.read_csv("mayo_clinic_strip_ai/data/ROIs/train/ROIs.csv")
    dataset = data.ROIDataset(
        training=training,
        metadata=rois,
        tif_dir="mayo_clinic_strip_ai/data/train",
        outline_dir="mayo_clinic_strip_ai/data/ROIs/train",
        crop_size=512,
        final_size=256,
        min_intersect_pct=0.5,
    )
    x = dataset[0]
    assert x.shape == (3, 256, 256)  # type: ignore[union-attr]


def test_stratified_batch_sampler():
    levels = pd.DataFrame({"label": np.random.choice(["a", "b"], size=101)})
    batch_sampler = data.StratifiedBatchSampler(levels=levels[["label"]], batch_size=10, seed=0)
    for batch in batch_sampler:
        assert len(batch) == batch_sampler.batch_size
