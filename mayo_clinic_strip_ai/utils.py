from typing import Optional, Tuple

import cv2
from matplotlib.pyplot import imshow
import numpy as np
import numpy.typing as npt


def downsize_img(img: npt.NDArray[np.uint8], downscale_factor: float) -> npt.NDArray[np.uint8]:
    return cv2.resize(src=img, dsize=(0, 0), fx=downscale_factor, fy=downscale_factor, interpolation=cv2.INTER_AREA)


def thumbnail(img: npt.NDArray[np.uint8], max_size: int) -> npt.NDArray[np.uint8]:
    scale_factor = max_size / max(img.shape)
    return downsize_img(img=img, downscale_factor=scale_factor)


def plot_thumbnail(
    img: npt.NDArray[np.uint8],
    crop: Optional[int] = 512,
    thickness: int = 4,
    color: Tuple[int, int, int] = (255, 255, 255),
) -> npt.NDArray[np.uint8]:
    if crop is not None:
        img = img.copy()
        for i in range(thickness):
            img[i::crop, :] = color
            img[:, i::crop] = color
    return imshow(thumbnail(img, max_size=1024))


def validate_img(img: npt.NDArray[np.uint8]) -> None:
    if img.dtype != np.uint8:
        raise RuntimeError(f"Got wrong dtype: {img.dtype}")
    if img.ndim != 3:
        raise RuntimeError(f"Got wrong ndims: {img.ndim}")
    if img.shape[2] != 3:
        raise RuntimeError(f"Got wrong number of channels: {img.shape[2]}")


def normalize_background(img: npt.NDArray[np.uint8], I_0: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    if all(I_0 == 255):
        # background is pure white; normalization is a no-op
        return img
    img = np.minimum(img, I_0)  # ensures no uint8 overflow
    normed = 255 ** (np.log(img) / np.log(I_0))
    return normed.astype(img.dtype)
