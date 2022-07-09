import os
from subprocess import check_call
from tempfile import TemporaryDirectory
from typing import Tuple, Type, TYPE_CHECKING

import numpy as np
import pandas as pd
from PIL import Image
import torch

if TYPE_CHECKING:
    import numpy.typing as npt

Image.MAX_IMAGE_PIXELS = None


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
            filepath,
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
        dest_dir = os.path.join(self.data_dir, self.name, row["patient_id"])
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
        dest_path = os.path.join(dest_dir, filename)
        with TemporaryDirectory() as tmpdir:
            check_call(
                [
                    "kaggle",
                    "competitions",
                    "download",
                    f"--file={dest_path}",
                    f"--path={tmpdir}",
                    Metadata.COMPETITION_NAME,
                ]
            )
            check_call(["unzip", os.path.join(tmpdir, f"{filename}.zip"), "-d", dest_dir])
        return dest_path

    def load_img(self, index: int) -> torch.Tensor:
        """
        Load an image as a `torch.Tensor` for a given metadata row.

        Parameters
        ----------
        index : int
            The index into the metadata.

        Returns
        -------
        torch.Tensor
            The loaded image.
        """
        img = Image.open(os.path.join(*self.local_img_path(index)), mode="r", formats=["TIFF"])
        arr: npt.NDArray[np.uint8] = np.array(img)
        return torch.from_numpy(arr)
