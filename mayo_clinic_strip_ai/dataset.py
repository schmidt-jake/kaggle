import os
from typing import Tuple

import numpy as np
from openslide import OpenSlide
import torch
from torch.utils.data import Dataset
from torchvision.transforms import PILToTensor
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import RandomVerticalFlip

from mayo_clinic_strip_ai.metadata import Metadata


class TifDataset(Dataset):
    def __init__(self, metadata: Metadata, training: bool, crop_size: int = 512) -> None:
        super().__init__()
        self.metadata = metadata
        self.random_hflip = RandomHorizontalFlip()
        self.random_vflip = RandomVerticalFlip()
        self.pil_to_tensor = PILToTensor()
        self.training = training
        self.crop_size = crop_size
        # self.cache = OpenSlideCache()

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img = OpenSlide(os.path.join(*self.metadata.local_img_path(index)))
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
        img = self.pil_to_tensor(img)
        if img[3, :, :].min() != 255 or img[3, :, :].max() != 255:
            raise ValueError("The alpha channel has signal?")
        img = img[:-1, :, :]  # drop alpha channel
        if self.training:
            img = self.random_hflip(img)
            img = self.random_vflip(img)
        label_id = self.metadata["label"].cat.codes.iloc[index]

        return img, label_id
