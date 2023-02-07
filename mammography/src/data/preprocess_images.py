import logging
import os
from argparse import ArgumentParser
from functools import partial
from multiprocessing.pool import Pool

import cv2
import pandas as pd
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from mammography.src.data import utils
from mammography.src.data.dicom import process_dicom

logger = logging.getLogger(__name__)


def process_image(filepath: str, output_dir: str) -> None:
    image_id = utils.extract_image_id_from_filepath(filepath)
    windows = process_dicom(filepath)
    thresh, mask = utils.breast_mask(windows.max(axis=0))
    if thresh > 5.0:
        logger.warning(f"Got suspiciously high threshold of {thresh} for {filepath}")
    cropped = utils.crop_and_mask(windows, mask)
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
    filepaths = [
        os.path.join(input_dir, str(row["patient_id"]), f"{image_id}.dcm")
        for _, row in meta.iterrows()
        for image_id in row["CC"] + row["MLO"]
    ]
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
