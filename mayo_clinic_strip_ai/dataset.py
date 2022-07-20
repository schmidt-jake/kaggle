import os
from typing import Tuple

import cv2
import numpy as np
from openslide import OpenSlide
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import RandomVerticalFlip

POS_CLS = "LAA"
NEG_CLS = "CE"

LABEL_MAP = {POS_CLS: 1, NEG_CLS: 0}


class TifDataset(Dataset):
    def __init__(self, metadata: pd.DataFrame, training: bool, data_dir: str, crop_size: int = 512) -> None:
        super().__init__()
        self.metadata = metadata
        self.random_hflip = RandomHorizontalFlip()
        self.random_vflip = RandomVerticalFlip()
        self.training = training
        self.crop_size = crop_size
        self.data_dir = data_dir
        # self.cache = OpenSlideCache()

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        row = self.metadata.iloc[index]
        img = OpenSlide(os.path.join(self.data_dir, row["image_id"] + ".tif"))
        if img.level_count > 1:
            raise ValueError(f"Got {img.level_count} levels!")
        w, h = img.dimensions
        if self.training:
            x = np.random.randint(low=0, high=w - self.crop_size)
            y = np.random.randint(low=0, high=h - self.crop_size)
        else:
            x = (w - self.crop_size) // 2
            y = (h - self.crop_size) // 2
        img = img.read_region(location=(x, y), size=(self.crop_size, self.crop_size), level=0)
        img = np.array(img)
        if img[:, :, 3].min() != img[:, :, 3].max():
            raise ValueError("The alpha channel has signal?")
        img = img[:, :, :3]  # drop alpha channel
        img = cv2.bitwise_not(img)
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)
        if self.training:
            img = self.random_hflip(img)
            img = self.random_vflip(img)
        label_id = LABEL_MAP[row["label"]]
        return img, label_id
