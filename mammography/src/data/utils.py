from typing import Any, Dict, Set, Tuple

import cv2
import numpy as np
import numpy.typing as npt
import torch
from torchvision.transforms import functional_tensor


def get_filepath(meta: Dict[str, Any], template: str) -> str:
    return template.format(**meta)


def select_keys(d: Dict[str, Any], keys: Set[str]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if k in keys}


def read_png(filepath: str) -> npt.NDArray[np.uint]:
    arr = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if arr is None:
        raise RuntimeError(f"No data found at {filepath}")
    return arr


def crop_right_center(img: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """
    Takes a crop that is on the right side of the arr, horizontally center.
    If needed, adds padding to the left, top, and bottom.
    """
    w, h = functional_tensor.get_image_size(img)
    top = (h - height) // 2
    left = w - width
    cropped = functional_tensor.crop(img=img, top=top, left=left, height=height, width=width)
    return cropped


def breast_mask(img: npt.NDArray[np.uint16]) -> Tuple[float, npt.NDArray[np.uint16]]:
    thresh, mask = cv2.threshold(img, thresh=5, maxval=1, type=cv2.THRESH_TRIANGLE)
    if thresh > 50.0:
        _, mask = cv2.threshold(img, thresh=5, maxval=1, type=cv2.THRESH_BINARY)
    contours = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)[0]
    max_contour = max(contours, key=cv2.contourArea)
    return thresh, cv2.drawContours(
        image=np.zeros_like(img, dtype=np.uint8), contours=[max_contour], contourIdx=0, color=1, thickness=-1
    )


def crop_and_mask(img: npt.NDArray[np.uint8], mask: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    x, y, w, h = cv2.boundingRect(mask)
    cropped = img[..., y : y + h, x : x + w]
    mask = mask[..., y : y + h, x : x + w]
    return cropped * mask
