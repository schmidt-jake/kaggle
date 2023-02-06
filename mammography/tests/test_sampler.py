import numpy as np
import pandas as pd
import torch
from pytest import MonkeyPatch

from mammography.src.data import MILDataset
from mammography.src.sampler import BreastSampler


def test_sampler(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setattr(
        "cv2.imread", lambda filepath, params: np.random.randint(low=0, high=255, size=(512, 512), dtype=np.uint8)
    )
    meta = pd.read_csv("mammography/data/raw/train.csv")
    meta = meta[meta["view"].isin(["CC", "MLO"])]
    sampler = BreastSampler(meta)
    meta["filepath"] = "foo"
    dataset = MILDataset(
        meta.set_index("image_id"), augmentation=torch.nn.Sequential(), keys={"patient_id", "laterality"}
    )
    for indices in sampler:
        batch = dataset[indices]
        print(batch)
        break
