import logging
import os
from subprocess import CalledProcessError
from typing import List, Tuple

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2**40)
import cv2
import numpy as np
import numpy.typing as npt

from mayo_clinic_strip_ai.data.metadata import Metadata

logger = logging.getLogger(__name__)


def get_contours(img: npt.NDArray[np.uint8]) -> Tuple[npt.NDArray[np.int32], ...]:
    x = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    x = cv2.GaussianBlur(x, ksize=(201, 201), sigmaX=0)
    x = cv2.threshold(x, thresh=250, maxval=255, type=cv2.THRESH_BINARY_INV)[1]
    contours = cv2.findContours(x, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)[0]
    return contours


def get_mask(img: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.blur(gray, ksize=(501, 501))
    thresh, mask = cv2.threshold(blur, thresh=0, maxval=255, type=cv2.THRESH_OTSU)
    logger.debug(f"Otsu threshold: {thresh}")
    return mask


def bounding_box_crop(img: npt.NDArray[np.uint8], mask: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    x, y, w, h = cv2.boundingRect(mask)
    cropped = img[y : y + h, x : x + w, ...]
    return cropped


def get_crops(img: npt.NDArray[np.uint8], mask: npt.NDArray[np.uint8], min_gap: int = 101):
    if min_gap % 2 == 0:
        min_gap += 1
    is_foreground_row = (mask > 0).any(axis=1)
    is_foreground_row = np.lib.stride_tricks.sliding_window_view(
        is_foreground_row,
        window_shape=min_gap,
        writeable=False,
    ).any(axis=1)
    is_foreground_row = np.pad(is_foreground_row, pad_width=min_gap // 2, mode="edge")
    dx = np.diff(is_foreground_row)
    i = np.flatnonzero(dx)
    img_splits = np.vsplit(img, i + 1)
    mask_splits = np.vsplit(mask, i + 1)
    out = []
    for img_split, mask_split in zip(img_splits, mask_splits):
        if (mask_split > 0).any():  # this split contains foreground
            x, y, w, h = cv2.boundingRect(mask_split)
            if w > 1000 and h > 1000:  # min size filter
                crop = img_split[y : y + h, x : x + w, :]
                mask_crop = mask_split[y : y + h, x : x + w, np.newaxis]

                # mask out the background
                crop = np.where(mask_crop > 0, crop, 0)
                out.append(crop)
    return out


def get_img(meta: Metadata, index: int) -> npt.NDArray[np.uint8]:
    if not os.path.exists(os.path.join(*meta.local_img_path(index))):
        meta.download_img(index)
    img = meta.load_tif(index)
    return img


def write_tif(img: npt.NDArray[np.uint8], filename: str) -> None:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    cv2.imwrite(
        filename=filename,
        img=img,
        params=[cv2.IMWRITE_TIFF_COMPRESSION, 8],  # use adobe-deflate lossless compression
    )


def process_img(img: npt.NDArray[np.uint8]) -> List[npt.NDArray[np.uint8]]:
    if img.ndim != 3:
        raise ValueError(f"Got wrong number of dimensions: {img.ndim}")

    if img.shape[2] != 3:
        raise ValueError("Image isn't RGB or channels last")

    if img.shape[1] > img.shape[0]:
        logger.debug("Transposing image")
        img = img.transpose(1, 0, 2)

    img = cv2.bitwise_not(img)  # invert colors so that background is 0 (black)
    mask = get_mask(img)
    img = bounding_box_crop(img, mask)
    mask = bounding_box_crop(mask, mask)
    crops = get_crops(img, mask, min_gap=int(0.1 * img.shape[0]))
    if len(crops) == 0:
        raise ValueError("Found no crops!")
    elif len(crops) > 3:
        raise ValueError(f"Found too many ({len(crops)}) splits!")
    return crops


if __name__ == "__main__":
    logging.basicConfig(
        filename="log.txt",
        filemode="w",
        level=logging.DEBUG,
        format="%(asctime)s:%(levelname)s:%(name)s:%(funcName)s:%(lineno)d:%(message)s",
    )

    meta = Metadata.from_csv(
        filepath="train.csv",
        data_dir="mayo_clinic_strip_ai/data/",
    )

    for row_num, row in meta.iterrows():
        logger.debug(f"Starting on image {row_num}...")
        try:
            img = get_img(meta, row_num)
            crops = process_img(img)
            for crop_num, crop in enumerate(crops):
                write_tif(
                    img=crop,
                    filename=os.path.join("mayo_clinic_strip_ai/data/out", row["image_id"], f"{crop_num}.tif"),
                )
                logger.debug(f"Split {crop_num} saved.")
            logger.debug("Done with image!")
        except (ValueError, CalledProcessError) as e:
            logger.error(f"Error with {os.path.join(*meta.local_img_path(row_num))}\n{e}")
