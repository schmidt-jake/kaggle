from typing import Optional, Tuple

import numpy as np
import numpy.typing as npt
from PIL import Image


def plot(
    arr: npt.NDArray[np.uint8],
    crop: Optional[int] = 512,
    thickness: int = 4,
    color: Tuple[int, int, int] = (255, 255, 255),
) -> Image:
    arr = arr.copy()
    if crop is not None:
        for i in range(thickness):
            arr[i::crop, :] = color
            arr[:, i::crop] = color
    img = Image.fromarray(arr)
    img.thumbnail((500, 500))
    return img
