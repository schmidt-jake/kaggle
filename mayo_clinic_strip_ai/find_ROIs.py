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
from typing import Any, Dict, List, Tuple, Type

import cv2
import hydra
from hydra.utils import HydraConfig
import numpy as np
import numpy.typing as npt
from omegaconf import DictConfig
import pandas as pd
from scipy.stats import mode
from tqdm import tqdm

from mayo_clinic_strip_ai.utils import thumbnail

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
    """
    # NOTE: (empirically) order of operations is important:
    # 1. Invert colors so background is black
    # 2. Convert to grayscale
    # 3. Blur
    # 4. Compute the foreground threshold
    if ksize % 2 == 0:
        # ksize must be odd
        ksize += 1
    x = cv2.bitwise_not(img)  # opencv expects background to be black
    cv2.boxFilter(src=x, ksize=[ksize] * 2, ddepth=cv2.CV_8U, dst=x)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2GRAY)
    cv2.threshold(x, thresh=0, maxval=255, type=cv2.THRESH_OTSU, dst=x)
    return x


def contour_border_intensity(
    img_rgb: npt.NDArray[np.uint8],
    contours: List[npt.NDArray[np.int32]],
    thickness: int = 20,
) -> npt.NDArray[np.uint8]:
    # create a binary mask where the contour border pixels are 1 and everything else is 0
    contour_border_mask = cv2.drawContours(
        np.zeros_like(img_rgb), contours=contours, contourIdx=-1, color=1, thickness=thickness
    )

    # convert to float dtype so we can mask with NaNs
    x = img_rgb.astype(np.float16)
    x[contour_border_mask] = np.nan

    # the background intensity is the mode of each channel, ignoring our NaNs
    I_0 = mode(x.reshape(-1, 3), axis=0, nan_policy="omit", keepdims=False).mode.astype(np.uint8)
    return I_0


def is_healthy_contour(contour: npt.NDArray[np.int32], min_area: float, max_aspect_ratio: float) -> bool:
    # filters contours that are too small (like debris)
    # filters contours that are too skinny (like slide edges)
    if max_aspect_ratio < 1.0:
        raise ValueError(f"Got max_aspect_ratio less than 1.0: {max_aspect_ratio}")
    area = cv2.contourArea(contour)
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = max(w / h, h / w)
    return area >= min_area and aspect_ratio <= max_aspect_ratio


def process_file(image_id: str, cfg: DictConfig, output_dir: Path) -> List[Dict[str, Any]]:
    filepath = os.path.join(cfg.data_dir, image_id + ".jpeg")
    logger.info(f"Starting on {filepath}...")
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    tissue_mask = get_mask(img, ksize=cfg.blur_ksize // cfg.downscale_factor)
    contours: List[npt.NDArray[np.int32]] = cv2.findContours(
        tissue_mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE
    )[0]
    contours = [c.squeeze(axis=1) for c in contours]
    contours = [
        c
        for c in contours
        if is_healthy_contour(c, min_area=cfg.min_outline_area, max_aspect_ratio=cfg.max_aspect_ratio)
    ]
    thickness = 40 // cfg.downscale_factor

    I_0 = contour_border_intensity(img_rgb=img, contours=contours, thickness=thickness)

    # draw thumbnails
    img = cv2.drawContours(img, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=thickness)
    cv2.imwrite(
        img=thumbnail(img, max_size=1024 * 2), filename=output_dir.joinpath("thumbnails", image_id + ".jpeg").as_posix()
    )
    del img

    # outputs
    out_dicts: List[Dict[str, Any]] = []
    if len(contours) > 0:
        logger.info(f"Found {len(contours)} outlines.")
        pth = output_dir / "outlines" / image_id
        pth.mkdir(exist_ok=False)
        for roi_num, contour in enumerate(contours):
            contour *= cfg.downscale_factor
            np.save(pth / str(roi_num), contour, allow_pickle=False)
            x, y, w, h = cv2.boundingRect(contour)
            out_dicts.append(
                {
                    "image_id": image_id,
                    "roi_num": roi_num,
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                    "area": cv2.contourArea(contour),
                    # opencv uses BGR order
                    "I_0_R": I_0[2],
                    "I_0_G": I_0[1],
                    "I_0_B": I_0[0],
                }
            )

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
            fieldnames=["image_id", "roi_num", "x", "y", "w", "h", "area", "I_0_R", "I_0_G", "I_0_B"],
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
