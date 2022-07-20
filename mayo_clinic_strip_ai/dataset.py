import os
from typing import Tuple

import cv2
import numpy as np
import numpy.typing as npt
from openslide import OpenSlide
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import RandomVerticalFlip

from mayo_clinic_strip_ai.find_ROIs import Rect

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

    @staticmethod
    def _read_region(img: OpenSlide, crop: Rect) -> npt.NDArray[np.uint8]:
        x = img.read_region(
            location=(crop.x, crop.y),
            size=(crop.w, crop.h),
            level=0,
        )
        x = np.array(x)
        x = cv2.cvtColor(x, cv2.COLOR_RGBA2RGB)
        x = cv2.bitwise_not(x)
        return x

    def _random_crop(self, img: OpenSlide, roi: Rect) -> npt.NDArray[np.uint8]:
        x_offset = np.random.randint(low=0, high=min(1, roi.w - self.crop_size))
        y_offset = np.random.randint(low=0, high=min(1, roi.h - self.crop_size))
        return self._read_region(
            img=img,
            crop=Rect(
                x=roi.x + x_offset,
                y=roi.y + y_offset,
                w=self.crop_size,
                h=self.crop_size,
            ),
        )

    def valid_random_crop(self, img: OpenSlide, roi: Rect) -> npt.NDArray[np.uint8]:
        x = self._random_crop(img=img, roi=roi)
        i = 0
        while x.mean() < 20.0:
            if i > 200:
                print("Couldn't find a good crop!")
                break
            x = self._random_crop(img=img, roi=roi)
            i += 1
        return x

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        row = self.metadata.iloc[index]
        img = OpenSlide(os.path.join(self.data_dir, row["image_id"] + ".tif"))
        if img.level_count > 1:
            raise ValueError(f"Got {img.level_count} levels!")
        roi = Rect(x=row["x"], y=row["y"], w=row["w"], h=row["h"])
        # FIXME: what if ROI is smaller than crop size?
        if self.training:
            img = self.valid_random_crop(img=img, roi=roi)
        else:
            x_offset = (roi.w - self.crop_size) // 2
            y_offset = (roi.h - self.crop_size) // 2
            img = self._read_region(
                img=img,
                crop=Rect(
                    x=roi.x + x_offset,
                    y=roi.y + y_offset,
                    w=self.crop_size,
                    h=self.crop_size,
                ),
            )
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)
        if self.training:
            img = self.random_hflip(img)
            img = self.random_vflip(img)
        label_id = LABEL_MAP[row["label"]]
        return img, label_id
