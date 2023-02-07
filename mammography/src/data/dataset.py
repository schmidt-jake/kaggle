from functools import wraps
from logging import getLogger
from typing import Any, Callable, Dict, List, Optional

import pandas as pd
from torch.utils.data import Dataset

logger = getLogger(__name__)


def map_fn(
    f: Callable[[Any], Any], input_key: Optional[str] = None, output_key: Optional[str] = None
) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    @wraps(f)
    def wrapper(d: Dict[str, Any]) -> Dict[str, Any]:
        input = d if input_key is None else d[input_key]
        if output_key is None:
            return f(input)
        else:
            d[output_key] = f(input)
            return d

    return wrapper


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
        for fn in self.fns:
            d = fn(d)
        return d

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({[fn.__name__ for fn in self.fns]})"
