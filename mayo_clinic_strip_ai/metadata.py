import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd


def load_metadata(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(
        filepath,
        dtype={
            "image_id": "string",
            "center_id": "category",
            "patient_id": "category",
            "image_num": "uint8",
        },
    )
    if "label" in df.columns:
        df["label"] = df["label"].astype("category")
    return df


def load_tif(filepath: str) -> npt.NDArray[np.uint8]:
    img: npt.NDArray[np.uint8] = cv2.imread(filepath, cv2.IMREAD_COLOR)
    # cv2.imread loads channels in BGR order, so we flip to RGB
    img = np.flip(img, axis=2)
    return img
