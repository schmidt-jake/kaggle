import os
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


@pytest.mark.parametrize("training", [True, False])
def test_roidataset(tif_img_path: Path, training: bool):
    metadata = pd.DataFrame([{"image_id": "img", "label": "LAA", "x": 0, "y": 0, "h": 512, "w": 512}])
    dataset = data.ROIDataset(training=training, metadata=metadata, tif_dir=os.path.dirname(tif_img_path))
    for i in range(len(dataset)):
        print(dataset[i])


def test_stratified_batch_sampler():
    rois = pd.read_csv("mayo_clinic_strip_ai/data/ROIs/train/ROIs.csv")
    meta = pd.read_csv("mayo_clinic_strip_ai/data/train.csv")
    levels = rois.merge(meta, on="image_id", how="inner", validate="m:1")
    batch_sampler = data.StratifiedBatchSampler(levels=levels[["label"]], batch_size=10)
    for batch in batch_sampler:
        assert len(batch) == batch_sampler.batch_size
