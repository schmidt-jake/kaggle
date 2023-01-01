import logging
import os
from glob import glob
from multiprocessing.pool import Pool

import cv2
import numpy as np
from tqdm import tqdm

from mammography.src.train import crop, dicom2numpy


def process_image(filepath: str) -> None:
    cv2.setNumThreads(0)
    image_id = filepath.split("/")[-1][:-4]
    save_path = os.path.join("mammography/data/uint8_crops", f"{image_id}.npy")
    if not os.path.exists(save_path):
        arr = dicom2numpy(filepath)
        try:
            arr = crop(arr)
        except ValueError:
            logging.exception(f"{filepath=}")
        np.save(file=save_path, arr=arr, allow_pickle=False)


def main() -> None:
    filepaths = glob("mammography/data/*_images/*/*.dcm")
    with Pool(8) as pool:
        pool.map(process_image, tqdm(filepaths))


if __name__ == "__main__":
    main()
