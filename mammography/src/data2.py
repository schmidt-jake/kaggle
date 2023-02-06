from functools import partial, wraps
from typing import Any, Callable, Dict, Set

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
import torch
from torch.utils.data import default_collate
from torchdata.dataloader2 import DataLoader2, PrototypeMultiProcessingReadingService
from torchdata.datapipes.map import MapDataPipe


def read_png(filepath) -> torch.Tensor:
    arr = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise RuntimeError(f"No data found at {filepath}")
    t = torch.from_numpy(arr)
    t.unsqueeze_(dim=0)
    return t


def select_keys(d: Dict[str, Any], keys: Set[str]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if k in keys}


def apply_fn(fn: Callable, data: Dict[str, Any], input_col: str, output_col: str) -> Dict[str, Any]:
    data[output_col] = fn(data[input_col])
    return data


class DataframeDataPipe(MapDataPipe):
    def __init__(self, df: pd.DataFrame) -> None:
        super().__init__()
        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.df.iloc[index]
        return row.to_dict()


def build_dataloader(df: pd.DataFrame, augmentation: Callable) -> DataLoader2:
    pipe = DataframeDataPipe(df=df)
    fn = partial(apply_fn, fn=read_png, input_col="filepath", output_col="pixels")
    pipe = pipe.map(fn)
    # pipe = pipe.in_memory_cache()
    pipe = pipe.shuffle()
    pipe = pipe.map(partial(select_keys, keys={"pixels", "cancer"}))
    # pipe = pipe.map(augmentation, input_col="pixels", output_col="pixels")
    pipe = pipe.batch(2)
    # pipe = pipe.map(default_collate)
    dataloader = DataLoader2(datapipe=pipe, reading_service=PrototypeMultiProcessingReadingService(num_workers=0))
    return dataloader
