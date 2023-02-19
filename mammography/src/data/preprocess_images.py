import logging
import os
from argparse import ArgumentParser
from functools import partial
from multiprocessing.pool import Pool

import cv2
import pandas as pd
from tqdm.auto import tqdm

from mammography.src.data import utils
from mammography.src.data.dicom import process_dicom

logger = logging.getLogger(__name__)


def process_image(filepath: str, output_dir: str) -> None:
    cv2.setNumThreads(0)
    windows = process_dicom(filepath, raw=True, output_bit_depth=16)
    windows = process_dicom(filepath, raw=False, output_bit_depth=8, scale_factor=1 / 2)
    if windows.ndim == 3:
        windows = windows.transpose(1, 2, 0)
        mask = utils.breast_mask(windows.max(axis=2))
        if windows.shape[2] not in [1, 3, 4]:
            logger.warning(f"Wrong number of windows: {windows.shape[2]}")
            windows = windows[:, :, :4]
    elif windows.ndim == 2:
        mask = utils.breast_mask(windows)
    cropped = utils.crop_and_mask(windows, mask)
    image_id = utils.extract_image_id_from_filepath(filepath)
    cv2.imwrite(
        filename=os.path.join(output_dir, f"{image_id}.png"), img=cropped, params=[cv2.IMWRITE_PNG_COMPRESSION, 0]
    )


def main(metadata_path: str, input_dir: str, output_dir: str) -> None:
    logging.basicConfig(level=logging.INFO)
    logger.info("Beginning job...")
    meta = pd.read_json(metadata_path)
    # bad = pd.read_csv("shapes.csv").query("background_pct >= 0.45 and background_pct < 0.5")["image_id"].to_list()
    filepaths = [
        os.path.join(input_dir, str(row["patient_id"]), f"{image_id}.dcm")
        for _, row in meta.iterrows()
        for image_id in row["CC"] + row["MLO"]
    ]
    # filepaths = [
    #     f for f in filepaths if not os.path.exists(f"{output_dir}/{utils.extract_image_id_from_filepath(f)}.png")
    # ]
    filepaths = sorted(filepaths)

    with Pool() as pool:
        for _ in tqdm(
            pool.imap_unordered(partial(process_image, output_dir=output_dir), filepaths),
            total=len(filepaths),
            smoothing=0.0,
        ):
            continue


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input_dir", type=str)
    parser.add_argument("metadata_path", type=str)
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(**vars(args))
