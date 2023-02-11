import logging
import os
from argparse import ArgumentParser
from operator import itemgetter

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from mammography.src.data import utils
from mammography.src.data.dataset import DataframeDataPipe
from mammography.src.data.dicom import process_dicom

logger = logging.getLogger(__name__)


def process_image(x) -> npt.NDArray[np.uint8]:
    filepath = x["filepath"]
    windows = process_dicom(filepath, raw=True, output_bit_depth=16)
    if windows.ndim == 3:
        windows = windows.transpose(1, 2, 0)
        if windows.shape[2] not in [1, 3, 4]:
            raise ValueError(f"Wrong number of windows: {windows.shape[2]}")
        thresh, mask = utils.breast_mask(windows.max(axis=2))
    elif windows.ndim == 2:
        thresh, mask = utils.breast_mask(windows)
    if thresh > 5.0:
        logger.warning(f"Got suspiciously high threshold of {thresh} for {filepath}")
    cropped = utils.crop_and_mask(windows, mask)
    if cropped.shape[0] < 512 or cropped.shape[1] < 512:
        logger.warning(f"Crop shape {cropped.shape} too small. Filepath: {filepath}")
    cropped = utils.resize(cropped, max_size=512)
    return cropped, utils.extract_image_id_from_filepath(filepath)


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
    filepaths = sorted(filepaths)
    logger.info("Preprocessing images...")
    pipe = DataframeDataPipe(
        df=pd.DataFrame({"filepath": filepaths}),
        fns=[process_image],
    )
    dataloader = DataLoader(
        pipe, batch_size=1, shuffle=False, num_workers=8, prefetch_factor=4, collate_fn=itemgetter(0)
    )
    with logging_redirect_tqdm():
        for img, image_id in tqdm(dataloader):
            cv2.imwrite(filename=f"{output_dir}/{image_id}.png", img=img, params=[cv2.IMWRITE_PNG_COMPRESSION, 3])

    logger.info("Done!")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input_dir", type=str)
    parser.add_argument("metadata_path", type=str)
    parser.add_argument("output_dir", type=str)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    main(**vars(args))
