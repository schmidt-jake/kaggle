from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import RandomVerticalFlip

from mayo_clinic_strip_ai.data.metadata import Metadata


class TifDataset(Dataset):
    def __init__(self, data_dir: Path, metadata: Metadata) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.metadata = metadata

    def __len__(self) -> int:
        return len(self.metadata)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        img = self.metadata.load_img(index)
        label_id = self.metadata["label"].cat.codes.iloc[index]
        return img, label_id


class Augmenter(torch.jit.ScriptModule):
    def __init__(self):
        super().__init__()
        self.crop = RandomCrop(size=512)
        self.hflip = RandomHorizontalFlip()
        self.vflip = RandomVerticalFlip()

    @torch.jit.script_method
    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if self.training:
            img = self.crop(img)
            img = self.hflip(img)
            img = self.vflip(img)
        return img
