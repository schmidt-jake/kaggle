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

from functools import partial
from multiprocessing.pool import Pool
import os
from pathlib import Path

import cv2
import hydra
from hydra.utils import HydraConfig
import numpy as np
import numpy.typing as npt
from omegaconf import DictConfig
import pandas as pd
from tqdm import tqdm


def thumbnail(img: npt.NDArray[np.uint8], max_size: int) -> npt.NDArray[np.uint8]:
    scale_factor = max_size / max(img.shape)
    return cv2.resize(src=img, dsize=(0, 0), fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)


def make_thumbnail(image_id: str, cfg: DictConfig, output_dir: Path) -> None:
    filepath = os.path.join(cfg.data_dir, image_id + ".tif")
    img = cv2.imread(filepath, cv2.IMREAD_COLOR)
    img = thumbnail(img, max_size=cfg.max_size)
    cv2.cvtColor(img, cv2.COLOR_BGR2RGB, dst=img)
    cv2.imwrite(img=img, filename=(output_dir / "thumbnails" / (image_id + ".jpeg")).as_posix())


@hydra.main(config_path="config", config_name="make_thumbnails", version_base=None)
def main(cfg: DictConfig):
    if "OPENCV_IO_MAX_IMAGE_PIXELS" not in os.environ.keys():
        raise ValueError("Must set OPENCV_IO_MAX_IMAGE_PIXELS env var!")

    hc = HydraConfig.get()
    run_dir = Path(hc.run.dir)

    (run_dir / "thumbnails").mkdir()

    meta = pd.read_csv(cfg.input_filepath, dtype={"image_id": "string"}, usecols=["image_id"])
    with Pool(processes=cfg.num_processes) as pool:
        pool.map_async(
            func=partial(make_thumbnail, cfg=cfg, output_dir=run_dir),
            iterable=tqdm(meta["image_id"].tolist()),
            chunksize=1,
        )


if __name__ == "__main__":
    main()
