import json
from typing import Dict, Optional, Type

import cv2
import numpy.typing as npt
import torch
from torchvision.transforms.functional_tensor import resize


class MinMaxScale(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _min = x.amin(dim=(-2, -1), keepdim=True)
        _max = x.amax(dim=(-2, -1), keepdim=True)
        x -= _min
        x /= _max - _min
        return x


class CLAHE(torch.nn.Module):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.clahe = cv2.createCLAHE(**kwargs)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.squeeze(dim=0).numpy()
        x = self.clahe.apply(x)
        return torch.from_numpy(x).unsqueeze(dim=0)

    def __repr__(self) -> str:
        return repr(self.clahe)


class PercentileScale(torch.nn.Module):
    def __init__(self, min: float, max: float) -> None:
        super().__init__()
        self.register_buffer("percentiles", torch.tensor([min, max]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (N, C, H, W) with range [0.0, 1.0]
        q = x.flatten(start_dim=2).quantile(q=self.get_buffer("percentiles"), dim=-1, keepdim=True)
        q.unsqueeze_(-1)
        _min, _max = q[0], q[1]
        x -= _min
        x /= _max - _min
        # x.clamp_(0.0, 1.0)
        return x


class ToTensor(torch.nn.Module):
    """
    Very similar to `torchvision.transforms.ToTensor`, except doesn't alter data (dtype or values).
    Converts channels-last to channels-first. Adds a leading channel dimension if necessary.
    """

    def forward(self, x: npt.NDArray) -> torch.Tensor:
        t = torch.from_numpy(x)
        if t.ndim == 2:
            t.unsqueeze_(dim=0)
        elif t.ndim == 3:
            t = t.permute(2, 0, 1)
        else:
            raise RuntimeError(f"Invalid shape: {t.shape}")
        return t


class CropCenterRight(torch.nn.Module):
    def __init__(self, height: int, width: int) -> None:
        super().__init__()
        self.height = height
        self.width = width

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        h, w = img.shape[-2:]
        if w > self.width:
            x_slice = slice(w - self.width, None)
        else:
            x_slice = slice(None)
        if h > self.height:
            dy = round((h - self.height) / 2)
            y_slice = slice(dy, dy + self.height)
        else:
            y_slice = slice(None)
        return img[..., y_slice, x_slice]


class ResizeLookup(torch.nn.Module):
    def __init__(self, lookup_table: Dict[str, float], default: Optional[float] = None) -> None:
        super().__init__()
        self.lookup_table = lookup_table
        self.default = default

    def forward(self, img: torch.Tensor, key: str, **kwargs) -> torch.Tensor:
        scale = self.lookup_table.get(key, self.default)
        if scale is None:
            raise KeyError(f"Key {key} not in lookup table and default is None.")
        size = (round(scale * img.size(-2)), round(scale * img.size(-1)))
        return resize(img=img, size=size, **kwargs)

    def resize(self, d: Dict, key: str) -> torch.Tensor:
        return self(img=d[key], key=d["machine_id"])

    def get_extra_state(self) -> Dict:
        return {"lookup_table": self.lookup_table}

    def set_extra_state(self, state) -> None:
        self.lookup_table = state["lookup_table"]

    @classmethod
    def from_json(cls: Type["ResizeLookup"], filepath: str, scale_factor: float) -> "ResizeLookup":
        with open(filepath, mode="r") as f:
            lookup_table = json.load(f)
        lookup_table = scale_factor / lookup_table
        return cls(lookup_table=lookup_table)
