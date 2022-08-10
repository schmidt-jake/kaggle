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

from csv import DictWriter
from csv import QUOTE_NONE
from dataclasses import dataclass
from functools import partial
import logging
from multiprocessing.pool import Pool
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type

import cv2
import hydra
from hydra.utils import HydraConfig
import numpy as np
import numpy.typing as npt
from omegaconf import DictConfig
import pandas as pd
from tqdm import tqdm

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


# @profile
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
    img: npt.NDArray[np.uint8] = cv2.imread(filepath, cv2.IMREAD_COLOR)
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


# @profile
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
    if ksize % 2 == 0:
        ksize += 1
    # x = cv2.GaussianBlur(src=img, ksize=[ksize] * 2, sigmaX=0)
    x = cv2.boxFilter(src=img, ksize=[ksize] * 2, ddepth=cv2.CV_8U)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY, dst=x)
    # cv2.blur(x, ksize=(ksize, ksize), dst=x)
    # thresh, mask = threshold(img=blur)
    cv2.threshold(x, thresh=0, maxval=255, type=cv2.THRESH_OTSU, dst=x)

    # x = cv2.adaptiveThreshold(
    #     src=x,
    #     maxValue=255,
    #     adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #     thresholdType=cv2.THRESH_BINARY,
    #     blockSize=ksize,
    #     C=50,
    # )
    # if thresh == 0.0:
    #     raise ImageError("Both Otsu and Triangle algos failed to find foreground threshold!")
    # logger.info(f"Foreground threshold: {thresh}")
    return x


# @profile
def saturation_mask(img: npt.NDArray[np.uint8], ksize: int, thresh: Optional[float] = None) -> npt.NDArray[np.uint8]:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s = hsv[:, :, 1].copy()
    cv2.blur(s, ksize=(ksize, ksize), dst=s)
    cv2.threshold(s, thresh=thresh, maxval=1, type=cv2.THRESH_OTSU if thresh is None else cv2.THRESH_BINARY, dst=s)
    return s


def thumbnail(img: npt.NDArray[np.uint8], max_size: int) -> npt.NDArray[np.uint8]:
    scale_factor = max_size / max(img.shape)
    return cv2.resize(src=img, dsize=(0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)


def process_file(image_id: str, cfg: DictConfig, output_dir: Path) -> List[Dict[str, Any]]:
    filepath = os.path.join(cfg.data_dir, image_id + ".tif")
    logger.info(f"Starting on {filepath}...")
    img = load_tif(filepath)
    if img.dtype != np.uint8:
        raise RuntimeError(f"Got wrong dtype: {img.dtype}")
    if img.ndim != 3:
        raise RuntimeError(f"Got wrong ndims: {img.ndim}")
    if img.shape[2] != 3:
        raise RuntimeError(f"Got wrong number of channels: {img.shape[2]}")
    img = cv2.resize(
        src=img,
        dsize=(0, 0),
        fx=1 / cfg.downscale_factor,
        fy=1 / cfg.downscale_factor,
        interpolation=cv2.INTER_CUBIC,
    )
    cv2.bitwise_not(img, dst=img)  # opencv expects background to be black
    # sat_mask = saturation_mask(img=img, thresh=cfg.saturation_thresh, ksize=cfg.blur_ksize // cfg.downscale_factor)
    # sat_mask = np.expand_dims(sat_mask, axis=2)
    mask = get_mask(
        # img=img * sat_mask,
        img=img,
        ksize=cfg.blur_ksize // cfg.downscale_factor,
    )
    contours, _ = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)
    contours = [c.squeeze(axis=1) for c in contours]
    contours = [c for c in contours if cv2.contourArea(c) > cfg.min_outline_area // cfg.downscale_factor**2]
    out_dicts: List[Dict[str, Any]] = []
    if len(contours) > 0:
        logger.info(f"Found {len(contours)} outlines.")
        pth = output_dir / "outlines" / image_id
        pth.mkdir(exist_ok=False)
        for roi_num, contour in enumerate(contours):
            np.save(pth / str(roi_num), contour * cfg.downscale_factor, allow_pickle=False)
            roi = Rect.from_mask(contour)
            out_dicts.append(
                {
                    "image_id": image_id,
                    "roi_num": roi_num,
                    "x": roi.x,
                    "y": roi.y,
                    "w": roi.w,
                    "h": roi.h,
                    "blur": compute_blur(img=roi.crop(img=img)),
                    "area": cv2.contourArea(contour),
                }
            )

        # img = np.where(np.logical_not(sat_mask), np.array((0, 0, 255), dtype=np.uint8), img)
        img = cv2.drawContours(img, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=50)
        img = thumbnail(img, max_size=1024)
        cv2.imwrite(img=img, filename=(output_dir / "thumbnails" / (image_id + ".jpeg")).as_posix())
    else:
        logger.warning("No outlines found.")
    return out_dicts


@hydra.main(config_path="config", config_name="find_ROIs", version_base=None)
def main(cfg: DictConfig):
    if "OPENCV_IO_MAX_IMAGE_PIXELS" not in os.environ.keys():
        raise ValueError("Must set OPENCV_IO_MAX_IMAGE_PIXELS env var!")

    hc = HydraConfig.get()
    run_dir = Path(hc.run.dir)

    (run_dir / "outlines").mkdir()
    (run_dir / "thumbnails").mkdir()

    meta = pd.read_csv(cfg.input_filepath, dtype={"image_id": "string"}, usecols=["image_id"])

    with (run_dir / "ROIs.csv").open(mode="w", newline="", buffering=1) as f:
        writer = DictWriter(
            f,
            fieldnames=["image_id", "roi_num", "x", "y", "w", "h", "blur"],
            dialect="unix",
            strict=True,
            quoting=QUOTE_NONE,
        )
        writer.writeheader()
        with Pool(processes=cfg.num_processes) as pool:
            for rowdicts in pool.imap_unordered(
                func=partial(process_file, cfg=cfg, output_dir=run_dir),
                iterable=tqdm(meta["image_id"].tolist()),
            ):
                if len(rowdicts) == 0:
                    logger.warning(f"No outlines found for image {rowdicts[0]['image_id']}")
                writer.writerows(rowdicts)


if __name__ == "__main__":
    main()
