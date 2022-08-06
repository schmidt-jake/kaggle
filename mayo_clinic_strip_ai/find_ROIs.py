"""
This script detects Regions of Interest (ROIs) from TIF files of whole-slide histology images.
It outputs a CSV with these columns:
- image_id: the UUID of the TIF image
- roi_num: The index of the ROI found in the image. Only unique relative to an `image_id`.
- x: The pixel x coordinate of the left border of the ROI.
- y: The pixel y coordinate of the bottom of the ROI.
- w: The pixel width of the ROI.
- h: The pixel height of the ROI.
- thresh: The foreground threshold computed from the source image.

# WARNING: For the Mayo Clinic Strip AI dataset, this script takes >6 hours to complete.
"""
from dataclasses import dataclass
import logging
import os
from typing import Tuple, Type

import cv2
import hydra
import numpy as np
import numpy.typing as npt
from omegaconf import DictConfig
import pandas as pd

logger = logging.getLogger(__name__)


class ImageError(Exception):
    pass


class ROIError(ImageError):
    pass


@dataclass
class Rect(object):
    """
    Coordinates for a rectangle.

    Parameters
    ----------
    x: int
        The left side.
    y: int
        The bottom.
    w: int
        The width.
    h: int
        The height.
    """

    x: int
    y: int
    w: int
    h: int

    @classmethod
    def from_mask(cls: Type["Rect"], mask: npt.NDArray[np.uint8]) -> "Rect":
        """
        Create a `Rect` for the minimal bounding box of the foreground of a binary mask.

        Parameters
        ----------
        mask : npt.NDArray[np.uint8]
            A binary mask array, where 0 represents background and 1 is foreground.

        Returns
        -------
        Rect
            The minimal bounding box of the foreground.
        """
        x, y, w, h = cv2.boundingRect(mask)
        return cls(x=x, y=y, w=w, h=h)

    def crop(self, img: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        """
        Crops an image using `self`'s coordinates.

        Parameters
        ----------
        img : npt.NDArray[np.uint8]
            An array of shape (HW...)

        Returns
        -------
        npt.NDArray[np.uint8]
            The cropped image
        """
        cropped = img[self.y : self.y + self.h, self.x : self.x + self.w, ...]  # noqa: E203
        return cropped


def load_tif(filepath: str) -> npt.NDArray[np.uint8]:
    """
    Loads an image from a TIF file.

    Parameters
    ----------
    filepath : str
        The path to the TIF file

    Returns
    -------
    npt.NDArray[np.uint8]
        The image pixels as an array.
    """
    img: npt.NDArray[np.uint8] = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    return img


def threshold(img: npt.NDArray[np.uint8]) -> Tuple[float, npt.NDArray[np.uint8]]:
    thresh_otsu, mask_otsu = cv2.threshold(img, thresh=0, maxval=255, type=cv2.THRESH_OTSU)
    thresh_tri, mask_tri = cv2.threshold(img, thresh=0, maxval=255, type=cv2.THRESH_TRIANGLE)
    if thresh_tri > thresh_otsu:
        logger.info("Using triangle algo")
        return thresh_tri, mask_tri
    else:
        logger.info("Using Otsu algo")
        return thresh_otsu, mask_otsu


def compute_blur(img: npt.NDArray[np.uint8]) -> float:
    blur = cv2.Laplacian(img, ddepth=cv2.CV_8U).var()
    return blur


def get_mask(img: npt.NDArray[np.uint8], ksize: int) -> npt.NDArray[np.uint8]:
    """
    Computes the foreground mask of an image.

    Parameters
    ----------
    img : npt.NDArray[np.uint8]
        An RGB, channels-last image.
    ksize : int
        The (square) kernel size to use in `cv2.blur`.

    Returns
    -------
    npt.NDArray[np.uint8]
        the binary foreground mask.

    Raises
    ------
    ImageError
        If an appropriate foreground threshold couldn't be found.
    """
    # NOTE: (empirically) order of operations is important:
    # 1. Convert to grayscale
    # 2. Blur
    # 3. Compute the foreground threshold
    x = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.blur(x, ksize=(ksize, ksize), dst=x)
    # thresh, mask = threshold(img=blur)
    thresh, _ = cv2.threshold(x, thresh=0, maxval=255, type=cv2.THRESH_OTSU, dst=x)
    if thresh == 0.0:
        raise ImageError("Both Otsu and Triangle algos failed to find foreground threshold!")
    logger.info(f"Foreground threshold: {thresh}")
    return x


def mask_low_saturation_pixels_inplace(img: npt.NDArray[np.uint8]) -> None:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    thresh, mask = cv2.threshold(hsv[:, :, 1], thresh=0, maxval=1, type=cv2.THRESH_OTSU)
    img *= np.expand_dims(mask, axis=2)


@hydra.main(config_path="config", config_name="find_ROIs", version_base=None)
def main(cfg: DictConfig):
    if "OPENCV_IO_MAX_IMAGE_PIXELS" not in os.environ.keys():
        raise ValueError("Must set OPENCV_IO_MAX_IMAGE_PIXELS env var!")

    os.makedirs(cfg.output_dir, exist_ok=True)

    meta = pd.read_csv(
        cfg.input_filepath,
        dtype={
            "image_id": "string",
        },
        usecols=["image_id"],
    )

    out_file = open(os.path.join(cfg.output_dir, "ROIs.csv"), "w", buffering=1)
    with out_file:
        out_file.write("image_id,roi_num,x,y,w,h,blur\n")
        for row_num, row in meta.iterrows():
            filepath = os.path.join(cfg.data_dir, row["image_id"] + ".tif")
            logger.info(f"Starting on {filepath}...")
            img = load_tif(filepath)
            if img.dtype != np.uint8:
                raise RuntimeError(f"Got wrong dtype: {img.dtype}")
            if img.ndim != 3:
                raise RuntimeError(f"Got wrong ndims: {img.ndim}")
            if img.shape[2] != 3:
                raise RuntimeError(f"Got wrong number of channels: {img.shape[2]}")
            cv2.bitwise_not(img, dst=img)  # opencv expects background to be black
            mask_low_saturation_pixels_inplace(img=img)
            mask = get_mask(img, ksize=cfg.blur_ksize)
            contours, _ = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
            contours = [c for c in contours if cv2.contourArea(c) > cfg.min_outline_area]
            if len(contours) > 0:
                logger.info(f"Found {len(contours)} outlines.")
                pth = os.path.join(cfg.output_dir, row["image_id"])
                os.makedirs(pth, exist_ok=True)
                for roi_num, contour in enumerate(contours):
                    np.save(os.path.join(pth, str(roi_num)), contour, allow_pickle=False)
                    roi = Rect.from_mask(contour)
                    blur = compute_blur(img=roi.crop(img=img))
                    out_file.write(f"{row['image_id']},{roi_num},{roi.x},{roi.y},{roi.w},{roi.h},{blur}\n")
            else:
                logger.warning("No outlines found.")


if __name__ == "__main__":
    main()
