from functools import partial
from glob import glob
import logging
from multiprocessing.pool import Pool
import os

import cv2
import pandas as pd
from tqdm import tqdm

from mayo_clinic_strip_ai import utils

logger = logging.getLogger(__name__)


def process_file(image_id: str, input_dir: str, output_dir: str, downscale_factor) -> None:
    logger.info(f"Starting on {image_id}...")
    img = cv2.imread(os.path.join(input_dir, image_id + ".tif"), cv2.IMREAD_COLOR)
    utils.validate_img(img)
    img = utils.downsize_img(img=img, downscale_factor=1 / downscale_factor)
    cv2.imwrite(filename=os.path.join(output_dir, image_id + ".jpeg"), img=img)


if __name__ == "__main__":
    from argparse import ArgumentParser

    if "OPENCV_IO_MAX_IMAGE_PIXELS" not in os.environ.keys():
        raise ValueError("Must set OPENCV_IO_MAX_IMAGE_PIXELS env var!")

    parser = ArgumentParser()
    parser.add_argument("--input-filepath", type=str)
    parser.add_argument("--input-dir", type=str)
    parser.add_argument("--downscale-factor", type=int)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--num-processes", type=int)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    meta = pd.read_csv(args.input_filepath)["image_id"]
    existing_image_ids = [f.split("/")[-1].rstrip(".jpeg") for f in glob(os.path.join(args.output_dir, "*.jpeg"))]
    meta = meta[~meta.isin(existing_image_ids)]
    with Pool(processes=args.num_processes) as pool:
        pool.map(
            partial(
                process_file,
                input_dir=args.input_dir,
                output_dir=args.output_dir,
                downscale_factor=args.downscale_factor,
            ),
            tqdm(meta.tolist()),
            chunksize=1,
        )
