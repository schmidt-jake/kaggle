from glob import glob
from multiprocessing.pool import Pool
from typing import Any, Dict, Union

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd
from pydicom import FileDataset, dcmread
from pydicom.multival import MultiValue
from pydicom.valuerep import DSfloat
from tqdm import tqdm

from mammography.src import utils


def sharpness(img: npt.NDArray) -> float:
    # cv2.setNumThreads(0)
    return cv2.Laplacian(img, ddepth=cv2.CV_16UC1).var()


def to_numeric(s: Any) -> Union[int, float, str]:
    try:
        s = int(s)
    except ValueError:
        pass
    try:
        s = float(s)
        if s.is_integer():
            s = int(s)
    except ValueError:
        pass
    return s


def extract_dicom_file_metadata(filepath: str) -> Dict[str, Any]:
    image_id = filepath.split("/")[-1][:-4]
    row: Dict[str, Any] = {"image_id": image_id, "filepath": filepath}
    dcm: FileDataset = dcmread(filepath)
    for key in dcm.dir():
        if key == "PixelData":
            continue
        val = dcm[key].value
        if isinstance(val, (str, DSfloat)):
            val = to_numeric(val)
        elif isinstance(val, (int, float)):
            pass
        elif isinstance(val, MultiValue):
            val = [to_numeric(v) for v in val]
        else:
            raise ValueError(f"Unknown metadata! {filepath=}, {key=}, {type(val)=}")
        row[key] = val
    arr = dcm.pixel_array
    arr = utils.maybe_invert(arr=arr, dcm=dcm)
    row["dtype"] = arr.dtype
    row["sharpness"] = sharpness(arr)
    row["pixel_min"] = arr.min()
    row["pixel_max"] = arr.max()
    row["h"], row["w"] = arr.shape
    return row


def main() -> None:
    filepaths = glob("mammography/data/raw/train_images/*/*.dcm")
    filepaths = np.random.choice(filepaths, size=1000, replace=False)
    with Pool() as pool:
        meta = pool.map(
            extract_dicom_file_metadata,
            tqdm(filepaths, smoothing=0),
            chunksize=1,
        )
    meta = pd.DataFrame(meta)
    meta["image_id"] = meta["image_id"].astype(int)
    meta = meta.merge(pd.read_csv("mammography/data/raw/train.csv"), on="image_id", validate="1:1", how="outer")
    meta["pixel_max_base2"] = np.log2(meta["pixel_max"] + 1)
    meta.eval("aspect_ratio = h / w", inplace=True)
    meta.sort_values("image_id", inplace=True, ignore_index=True)
    meta.to_pickle("mammography/dicom_metadata.pickle")


if __name__ == "__main__":
    main()
