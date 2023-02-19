from functools import wraps
from logging import getLogger
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

logger = getLogger(__name__)


def map_fn(
    f: Callable[[Any], Any], d: Dict[str, Any], input_col: Optional[str] = None, output_col: Optional[str] = None
) -> Dict[str, Any]:
    input = d if input_col is None else d[input_col]
    if output_col is None:
        return f(input)
    else:
        d[output_col] = f(input)
        return d


class DataframeDataPipe(Dataset):
    def __init__(self, df: pd.DataFrame, fns: List[Callable[[Dict[str, Any]], Dict[str, Any]]]) -> None:
        super().__init__()
        self.df = df
        self.fns = fns

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        row = self.df.iloc[index]
        d = row.to_dict()
        for fn, input_col, output_col in self.fns:
            d = map_fn(f=fn, d=d, input_col=input_col, output_col=output_col)
        return d

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({[fn.__name__ for fn in self.fns]})"
