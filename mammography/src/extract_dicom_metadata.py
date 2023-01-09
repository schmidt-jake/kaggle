from glob import glob
from multiprocessing.pool import Pool
from typing import Any, Dict

import cv2
import numpy.typing as npt
import pandas as pd
from pydicom import FileDataset, dcmread
from pydicom.multival import MultiValue
from tqdm import tqdm


def sharpness(img: npt.NDArray) -> float:
    # cv2.setNumThreads(0)
    return cv2.Laplacian(img, ddepth=cv2.CV_16UC1).var()


def extract_dicom_file_metadata(filepath: str) -> Dict[str, Any]:
    image_id = filepath.split("/")[-1][:-4]
    row: Dict[str, Any] = {"image_id": image_id, "filepath": filepath}
    dcm: FileDataset = dcmread(filepath)
    for key in dcm.dir():
        if key == "PixelData":
            continue
        val = dcm[key].value
        if isinstance(val, (str, int, float)):
            pass
        elif isinstance(val, MultiValue):
            val = list(val)
        else:
            print(f"{filepath=}, {key=}, {type(val)=}")
        row[key] = val
    # arr = dcm.pixel_array
    # row["dtype"] = arr.dtype
    # row["sharpness"] = sharpness(arr)
    # row["pixel_min"] = arr.min()
    # row["pixel_max"] = arr.max()
    # row["h"], row["w"] = arr.shape
    return row


def main() -> None:
    with Pool(8) as pool:
        meta = pool.map(
            extract_dicom_file_metadata,
            tqdm(glob("mammography/data/raw/train_images/*/*.dcm"), smoothing=0),
            chunksize=1,
        )
    meta = pd.DataFrame(meta)
    meta.to_csv("mammography/dicom_metadata.csv", index=False)


if __name__ == "__main__":
    main()
