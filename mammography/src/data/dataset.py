import os
from logging import getLogger
from typing import Any, Dict, Set

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torchdata.datapipes.map import MapDataPipe

from mammography.src.data import preprocess_images

logger = getLogger(__name__)


class DicomDataset(IterableDataset):
    def __init__(self, meta: pd.DataFrame, image_dir: str) -> None:
        super().__init__()
        self.meta = meta
        self.image_dir = image_dir

    def __len__(self) -> int:
        return len(self.meta)

    def __getitem__(self, index: int) -> npt.NDArray[np.uint8]:
        row = self.meta.iloc[index]
        filepath = os.path.join(self.image_dir, str(row["patient_id"]), f"{row['image_id']}.dcm")
        windows = preprocess_images.process_dicom(filepath)
        thresh, mask = preprocess_images.breast_mask(windows.max(axis=0))
        if thresh > 5.0:
            logger.warning(f"Got suspiciously high threshold of {thresh} for {filepath}")
        cropped = preprocess_images.crop_and_mask(windows, mask)
        if cropped.shape[1] < 512 or cropped.shape[2] < 512:
            logger.warning(f"Crop shape {cropped.shape} too small. Image ID: {row['image_id']}")
        for i, window in enumerate(cropped):
            yield window

    def __iter__(self):
        for i in range(len(self)):
            yield from self[i]


class PNGDataset(MapDataPipe):
    def __init__(
        self, df: pd.DataFrame, augmentation: torch.nn.Sequential, keys: Set[str], filepath_format: str
    ) -> None:
        super().__init__()
        self.df = df
        self.augmentation = augmentation
        self.keys = keys
        self.filepath_format = filepath_format

    def __len__(self) -> int:
        return len(self.df)

    @staticmethod
    def _read(filepath: str) -> torch.Tensor:
        arr = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        if arr is None:
            raise RuntimeError(f"No data found at {filepath}")
        t = torch.from_numpy(arr)
        return t

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.df.iloc[index].to_dict()
        row.update(
            {
                view: self.augmentation(self._read(self.filepath_format.format(image_id=np.random.choice(row[view]))))
                for view in ["CC", "MLO"]
            }
        )
        return row
