from glob import glob
from logging import getLogger
from multiprocessing.pool import Pool

import cv2
import numpy as np
import numpy.typing as npt
from tqdm import tqdm

from mammography.src import utils
from mammography.src.data import extract_dicom

logger = getLogger(__name__)


def breast_mask(img: npt.NDArray[np.uint16]) -> npt.NDArray[np.uint16]:
    thresh, mask = cv2.threshold(img, thresh=5, maxval=1, type=cv2.THRESH_TRIANGLE)
    if thresh > 5.0:
        logger.warning(f"Got suspiciously high threshold of {thresh}")
    # kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(15, 15))
    # mask = cv2.dilate(mask, kernel, iterations=15)
    contours = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)[0]
    max_contour = max(contours, key=cv2.contourArea)
    return cv2.drawContours(
        image=np.zeros_like(img, dtype=np.uint8), contours=[max_contour], contourIdx=0, color=1, thickness=-1
    )


def crop2mask(img: npt.NDArray[np.uint8], mask: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    x, y, w, h = cv2.boundingRect(mask)
    cropped = img[..., y : y + h, x : x + w]
    return cropped


def breast_crop(img: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    mask = breast_mask(img)
    img = img * mask
    return crop2mask(img, mask)


def process_image(filepath: str) -> None:
    image_id = utils.extract_image_id_from_filepath(filepath)
    subdir = f"mammography/data/normalized/{image_id}"
    windows, window_indexes = zip(*extract_dicom(filepath))
    windows = np.stack(windows, axis=0)
    mask = breast_mask(windows.max(axis=0))
    windows *= mask
    cropped = crop2mask(windows, mask)
    for i, window_index in enumerate(window_indexes):
        window = cropped[i, ...]
        cv2.imwrite(filename=f"{subdir}_{window_index}.png", img=window, params=[cv2.IMWRITE_PNG_COMPRESSION, 7])


def main() -> None:
    filepaths = glob("mammography/data/raw/train_images/*/*.dcm")
    with Pool() as pool:
        pool.map(process_image, tqdm(filepaths, smoothing=0), chunksize=1)


if __name__ == "__main__":
    main()
