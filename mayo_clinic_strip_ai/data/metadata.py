import logging
import os
from subprocess import check_call
from tempfile import TemporaryDirectory
from typing import Tuple, Type

import cv2
import numpy as np
import numpy.typing as npt
import pandas as pd

logger = logging.getLogger(__name__)


class Metadata(pd.DataFrame):
    COMPETITION_NAME = "mayo-clinic-strip-ai"

    def __init__(self, *args, name: str, data_dir: str, **kwargs) -> None:
        """
        A specialized dataframe that can interact with the Kaggle API to download data
        and the downloaded TIFF data files. Abstracts away local data storage by providing
        an interface for downloading image data and loading it as `torch.Tensor`s.

        Parameters
        ----------
        name : str
            The name of the dataset, either "train", "test", or "other".
        data_dir : str
            The local root directory to cache downloaded data.
        """
        super().__init__(*args, **kwargs)
        self.name = name
        self.data_dir = data_dir

    @classmethod
    def from_csv(cls: Type["Metadata"], filepath: str, data_dir: str) -> "Metadata":
        """
        Wraps `pd.read_csv`, but automatically infers the `name` of the dataset
        and properly casts dataframe columns.

        Parameters
        ----------
        filepath : str
            The local filepath to a CSV containing metadata.
        data_dir : str
            The local directory to download data to.
        """
        df = pd.read_csv(
            os.path.join(data_dir, filepath),
            dtype={
                "image_id": "string",
                "center_id": "category",
                "patient_id": "string",
                "image_num": "uint8",
            },
        )
        if "label" in df.columns:
            df["label"] = df["label"].astype("category")
        return cls(
            data=df,
            name=filepath.split("/")[-1].rstrip(".csv"),
            data_dir=data_dir,
        )

    def local_img_path(self, index: int) -> Tuple[str, str]:
        """
        Returns the local path to the image for the given metadata row index.

        Parameters
        ----------
        index : int
            The index into the metadata.

        Returns
        -------
        Tuple[str, str]
            (directory, filename) — so the fully-qualified path is directory/filename
        """
        row = self.iloc[index]
        image_id = row["image_id"]
        dest_dir = os.path.join(self.data_dir, self.name)
        return dest_dir, f"{image_id}.tif"

    def download_img(self, index: int) -> str:
        """
        Download the image for a given metadata row from Kaggle as a TIFF file.

        Parameters
        ----------
        index : int
            The index into the metadata for which to download the image.

        Returns
        -------
        str
            The local filepath to which the image was downloaded.
        """
        dest_dir, filename = self.local_img_path(index)
        remote_path = os.path.join(self.name, filename)
        dest_path = os.path.join(dest_dir, filename)
        logger.debug(f"Downloading index {index} to {dest_path}...")
        with TemporaryDirectory() as tmpdir:
            check_call(
                [
                    "kaggle",
                    "competitions",
                    "download",
                    f"--file={remote_path}",
                    f"--path={tmpdir}",
                    Metadata.COMPETITION_NAME,
                ]
            )
            check_call(["unzip", os.path.join(tmpdir, f"{filename}.zip"), "-d", dest_dir])
        return dest_path

    def load_tif(self, index: int) -> npt.NDArray[np.uint8]:
        """
        Load an image as a numpy array for a given metadata row.

        Parameters
        ----------
        index : int
            The index into the metadata.

        Returns
        -------
        npt.NDArray[np.uint8]
            The loaded image.
        """
        filepath = os.path.join(*self.local_img_path(index))
        img: npt.NDArray[np.uint8] = cv2.imread(filepath, cv2.IMREAD_COLOR)
        img = np.flip(img, axis=2)
        logger.debug(f"Loaded image from {filepath}. Shape: {img.shape} Dtype: {img.dtype}")
        return img
