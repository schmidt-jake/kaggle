from dataclasses import dataclass
import logging
import os
from typing import List, Tuple, Type

import cv2
import numpy as np
import numpy.typing as npt

from mayo_clinic_strip_ai.metadata import load_metadata
from mayo_clinic_strip_ai.metadata import load_tif

logger = logging.getLogger(__name__)


class ImageError(Exception):
    pass


class ROIError(ImageError):
    pass


@dataclass
class Rect(object):
    x: int
    y: int
    w: int
    h: int

    @classmethod
    def from_mask(cls: Type["Rect"], mask: npt.NDArray[np.uint8]) -> "Rect":
        x, y, w, h = cv2.boundingRect(mask)
        return cls(x=x, y=y, w=w, h=h)

    def crop(self, img: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
        cropped = img[self.y : self.y + self.h, self.x : self.x + self.w, ...]  # noqa: E203
        return cropped


def get_mask(img: npt.NDArray[np.uint8]) -> Tuple[float, npt.NDArray[np.uint8]]:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.blur(gray, ksize=(501, 501))
    thresh, mask = cv2.threshold(blur, thresh=0, maxval=255, type=cv2.THRESH_OTSU)
    mask = cv2.bitwise_not(mask)
    logger.debug(f"Otsu threshold: {thresh}")
    return thresh, mask


def find_ROIs(mask: npt.NDArray[np.uint8], min_gap_ratio: float = 0.1) -> List[Rect]:
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
    dx = np.diff(is_foreground)
    split_offsets = np.flatnonzero(dx)
    mask_splits = np.split(mask, split_offsets + 1, axis=1 - axis)
    ROIs = []
    for offset, mask_split in zip(np.insert(split_offsets, 0, 0), mask_splits):
        if (mask_split > 0).any():  # this split contains foreground
            rect = Rect.from_mask(mask_split)
            if rect.w < 1000 or rect.h < 1000:  # min size filter
                logger.warning("Got too small region!")
            if axis == 1:
                rect.y += offset
            elif axis == 0:
                rect.x += offset
            ROIs.append(rect)
    return ROIs


def process_img(img: npt.NDArray[np.uint8]) -> Tuple[float, List[Rect]]:
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

    meta = load_metadata(args.input_filepath)

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
