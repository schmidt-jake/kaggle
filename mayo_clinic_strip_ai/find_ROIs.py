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
from typing import List, Tuple, Type

import cv2
import numpy as np
import numpy.typing as npt
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
    img: npt.NDArray[np.uint8] = cv2.imread(filepath, cv2.IMREAD_COLOR)
    # cv2.imread loads channels in BGR order, so we flip to RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def get_mask(img: npt.NDArray[np.uint8]) -> Tuple[float, npt.NDArray[np.uint8]]:
    """
    Computes the foreground mask of an image.

    Parameters
    ----------
    img : npt.NDArray[np.uint8]
        An RGB, channels-last image.

    Returns
    -------
    Tuple[float, npt.NDArray[np.uint8]]
        (threshold, mask), where `threshold` is the optimal foreground threshold and
        `mask` is the binary foreground mask.

    Raises
    ------
    ImageError
        If an appropriate foreground threshold couldn't be found.
    """
    # NOTE: (empirically) order of operations is important:
    # 1. Invert colors (opencv expects background to be black)
    # 2. Convert to grayscale
    # 3. Blur
    # 4. Compute the foreground threshold
    gray = cv2.cvtColor(cv2.bitwise_not(img), cv2.COLOR_RGB2GRAY)
    blur = cv2.blur(gray, ksize=(501, 501))
    thresh, mask = cv2.threshold(blur, thresh=0, maxval=255, type=cv2.THRESH_OTSU)
    if thresh == 0.0:
        logger.warning("Otsu algo failed to find foreground threshold. Falling back to triangle algo...")
    thresh, mask = cv2.threshold(blur, thresh=0, maxval=255, type=cv2.THRESH_TRIANGLE)
    if thresh == 0.0:
        raise ImageError("Both Otsu and Triangle algos failed to find foreground threshold!")
    logger.debug(f"Foreground threshold: {thresh}")
    return thresh, mask


def find_ROIs(mask: npt.NDArray[np.uint8], min_gap_ratio: float = 0.1) -> List[Rect]:
    """
    Uses some assumptions/heuristics to locate Regions of Interest (ROIs) from a binary foreground mask
    of whole-slide histology images.

    Assumptions:
    - Some images seem to have multiple slices of the same tissue distributed across the image.
    - If slices are distributed horizontally, the image will be much wider than tall.
    - If slices are distributed vertically, the image will be much taller than wide.
    - Slices have substantial dead space separating them.

    Heuristics:
    - If there's a bunch of sequential rows/columns of pure background in the middle of an image, it means
    there are multiple slices.

    Parameters
    ----------
    mask : npt.NDArray[np.uint8]
        A binary foreground mask where background is 0 and foreground is 1.
    min_gap_ratio : float, optional
        Sets the minimum distance requirement between ROIs, relative to the longest image dimension, by default 0.1.
        Example: if an image has a (height, width) = (1000, 2000), then for a `min_gap_ratio` of 0.1, any ROIs detected
        in the image must be at least 2000 x 0.1 = 200 pixels apart from each other.

    Returns
    -------
    List[Rect]
        A list of coordinates for the minimal bounding boxes of each detected ROI.
    """
    # the axis along which slices are separable is typically related to aspect ratio
    axis = 0 if mask.shape[1] > mask.shape[0] else 1
    min_gap = int(mask.shape[axis] * min_gap_ratio)
    # `min_gap` must be odd
    if min_gap % 2 == 0:
        min_gap += 1
    is_foreground = (mask > 0).any(axis=axis)
    is_foreground = np.lib.stride_tricks.sliding_window_view(
        is_foreground,
        window_shape=min_gap,
        writeable=False,
    ).any(axis=1)
    is_foreground = np.pad(is_foreground, pad_width=min_gap // 2, mode="edge")
    split_offsets = np.flatnonzero(np.diff(is_foreground)) + 1
    mask_splits = np.split(mask, split_offsets, axis=1 - axis)
    ROIs = []
    for offset, mask_split in zip(np.insert(split_offsets, 0, 0), mask_splits):
        if (mask_split > 0).any():  # this split contains foreground
            rect = Rect.from_mask(mask_split)
            if rect.w < 1000 or rect.h < 1000:  # min size filter
                logger.warning(f"ROI too small: w={rect.w} h={rect.h}")
            if axis == 1:
                rect.y += offset
            elif axis == 0:
                rect.x += offset
            ROIs.append(rect)
    return ROIs


def process_img(img: npt.NDArray[np.uint8]) -> Tuple[float, List[Rect]]:
    """
    The entrypoint into the ROI-finding routine.

    Parameters
    ----------
    img : npt.NDArray[np.uint8]
        A channels-last RGB image

    Returns
    -------
    Tuple[float, List[Rect]]
        (threshold, ROIs)

    Raises
    ------
    ImageError
        If the input image is malformed
    ROIError
        If no ROIs were detected in the image
    """
    if img.ndim != 3:
        raise ImageError(f"Got wrong number of dimensions: {img.ndim}")

    if img.shape[2] != 3:
        raise ImageError("Image isn't RGB or channels last")

    thresh, mask = get_mask(img)
    ROIs = find_ROIs(mask)
    if len(ROIs) == 0:
        raise ROIError("Found no ROIs!")
    return thresh, ROIs


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("data_dir", type=str, help="The path to a directory of TIFFs")
    parser.add_argument("input_filepath", type=str, help="The path to a metadata CSV")
    parser.add_argument("output_dir", type=str, help="Where to write results")
    args = parser.parse_args()

    if "OPENCV_IO_MAX_IMAGE_PIXELS" not in os.environ.keys():
        raise ValueError("Must set OPENCV_IO_MAX_IMAGE_PIXELS env var!")

    os.makedirs(args.output_dir, exist_ok=True)

    logging.basicConfig(
        filename=os.path.join(args.output_dir, "log.txt"),
        filemode="w",
        level=logging.DEBUG,
        format="%(asctime)s:%(levelname)s:%(name)s:%(funcName)s:%(lineno)d:%(message)s",
    )

    meta = pd.read_csv(
        args.input_filepath,
        dtypes={
            "image_id": "string",
        },
        use_cols=["image_id"],
    )

    with open(os.path.join(args.output_dir, "ROIs.csv"), "w", buffering=1) as f:
        f.write("image_id,roi_num,x,y,w,h,thresh\n")
        for row_num, row in meta.iterrows():
            filepath = os.path.join(args.data_dir, row["image_id"] + ".tif")
            logger.debug(f"Starting on {filepath}...")
            img = load_tif(filepath)
            try:
                thresh, ROIs = process_img(img)
            except (ImageError, ROIError) as e:
                logger.error(f"Error with image {filepath}:\n{e}")
            else:
                for roi_num, roi in enumerate(ROIs):
                    f.write(f"{row['image_id']},{roi_num},{roi.x},{roi.y},{roi.w},{roi.h},{thresh}\n")
                    logger.debug(f"Split {roi_num} saved.")
                logger.debug("Done with image!")
