import logging
import os
from argparse import ArgumentParser
from functools import partial
from multiprocessing.pool import Pool
from typing import Tuple

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from mammography.src import utils
from mammography.src.dicomsdl import process_dicom

logger = logging.getLogger(__name__)


def breast_mask(img: npt.NDArray[np.uint16]) -> Tuple[float, npt.NDArray[np.uint16]]:
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


def process_image(filepath: str, output_dir: str) -> None:
    image_id = utils.extract_image_id_from_filepath(filepath)
    windows = process_dicom(filepath)
    thresh, mask = breast_mask(windows.max(axis=0))
    if thresh > 5.0:
        logger.warning(f"Got suspiciously high threshold of {thresh} for {filepath}")
    cropped = crop_and_mask(windows, mask)
    if cropped.shape[1] < 512 or cropped.shape[2] < 512:
        logger.warning(f"Crop shape {cropped.shape} too small. Image ID: {image_id}")
    for i, window in enumerate(cropped):
        cv2.imwrite(
            filename=f"{output_dir}/{image_id}_{i}.png",
            img=window,
            params=[cv2.IMWRITE_PNG_COMPRESSION, 0],
        )


def main(metadata_path: str, input_dir: str, output_dir: str) -> None:
    logging.basicConfig(level=logging.INFO)
    logger.info("Beginning job...")
    if metadata_path.endswith(".csv"):
        meta = pd.read_csv(metadata_path)
    elif metadata_path.endswith(".pickle"):
        meta = pd.read_pickle(metadata_path)
    else:
        raise ValueError(f"Unrecognized suffix: {metadata_path}")
    filepaths = input_dir + meta["patient_id"].astype(str) + "/" + meta["image_id"].astype(str) + ".dcm"
    logger.info("Preprocessing images...")
    with Pool() as pool, logging_redirect_tqdm():
        pool.map(
            partial(process_image, output_dir=output_dir),
            tqdm(filepaths, smoothing=0, desc="preprocessing images..."),
            chunksize=10,
        )
    logger.info("Done!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input_dir", type=str)
    parser.add_argument("metadata_path", type=str)
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(**vars(args))
