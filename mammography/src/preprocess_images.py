import os
from logging import getLogger
from multiprocessing.pool import Pool

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm

from mammography.src import utils
from mammography.src.dicomsdl import process_dicom

logger = getLogger(__name__)


def breast_mask(img: npt.NDArray[np.uint16]) -> npt.NDArray[np.uint16]:
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


def process_image(filepath: str) -> None:
    cv2.setNumThreads(0)
    image_id = utils.extract_image_id_from_filepath(filepath)
    subdir = f"mammography/data/png/{image_id}"
    if os.path.exists(subdir + "_0.png"):
        return
    windows = process_dicom(filepath)
    thresh, mask = breast_mask(windows.max(axis=0))
    if thresh > 5.0:
        logger.warning(f"Got suspiciously high threshold of {thresh} for {filepath}")
    cropped = crop_and_mask(windows, mask)
    for i, window in enumerate(cropped):
        cv2.imwrite(filename=f"{subdir}_{i}.png", img=window, params=[cv2.IMWRITE_PNG_COMPRESSION, 0])


def main() -> None:
    meta = pd.read_csv("mammography/data/raw/train.csv")
    meta.query("patient_id != 27770", inplace=True)
    meta.query("image_id != 1942326353", inplace=True)
    meta.drop_duplicates("patient_id", inplace=True)
    # meta = meta.sample(n=5000, random_state=42)
    meta.sort_values("image_id", inplace=True)
    filepaths = (
        "mammography/data/raw/train_images/"
        + meta["patient_id"].astype(str)
        + "/"
        + meta["image_id"].astype(str)
        + ".dcm"
    )
    meta.to_csv("mammography/data/png/train.csv", index=False)

    with Pool() as pool:
        pool.map(process_image, tqdm(filepaths, smoothing=0))

    # filepaths = sorted(glob("mammography/data/raw/train_images/*/*.dcm"), key=utils.extract_image_id_from_filepath)
    # filepaths = filepaths[:100]
    # with Pool(4, context=get_context("fork")) as pool:
    #     pool.map(process_image, filepaths[:10], chunksize=None)
    #     t = time()
    #     pool.map(process_image, tqdm(filepaths, smoothing=1), chunksize=None)
    #     print(time() - t)


if __name__ == "__main__":
    main()
