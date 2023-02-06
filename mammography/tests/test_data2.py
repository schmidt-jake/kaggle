import pandas as pd
import torch
from torchvision.transforms import CenterCrop

from mammography.src.data2 import build_dataloader


def test_build_dataloader() -> None:
    df = pd.read_csv("mammography/data/png/train.csv")
    df["filepath"] = "mammography/data/png/" + df["image_id"].astype(str) + "_0.png"
    dataloader = build_dataloader(df=df, augmentation=torch.nn.Sequential(CenterCrop(32)))
    for batch in dataloader:
        print(batch)
        assert batch["cancer"].shape == (2,)
        assert batch["pixels"].shape == (2, 1, 32, 32)
        break
