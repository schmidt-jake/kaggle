"""
Defines a dataset that implements the `torch.utils.data.Dataset` interface
that can be used with a torch.utils.data.DataLoader in a training and/or inference loop.

Docs:
https://pytorch.org/docs/stable/data.html
"""

import os
from typing import Dict, Generator, Tuple, Union

import cv2
import numpy as np
import numpy.typing as npt
from openslide import OpenSlide
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import RandomVerticalFlip

from mayo_clinic_strip_ai.find_ROIs import Rect
from mayo_clinic_strip_ai.utils import normalize_background

POS_CLS = "LAA"
NEG_CLS = "CE"
LABEL_MAP = {POS_CLS: 1, NEG_CLS: 0}


class ROIDataset(Dataset):
    def __init__(
        self,
        metadata: pd.DataFrame,
        training: bool,
        tif_dir: str,
        outline_dir: str,
        crop_size: int,
        min_intersect_pct: float,
    ) -> None:
        """
        A dataset that loads image crops efficiently using Region-of-Interest (ROI) bounding box coordinates.

        Uses OpenSlide to load only the pixels needed, which saves memory.

        Parameters
        ----------
        metadata : pd.DataFrame
            A dataframe where every row defines a unique ROI for an image. Must contain these columns:
                - image_id
                - label
                - x
                - y
                - w
                - h
        training : bool
            If True, takes a random crop from an ROI and applies random augmentations.
            If False, takes a center crop.
        tif_dir : str
            The path to a directory containing TIF images, such that data_dir/image_id.tif
            is a valid filepath.
        outline_dir : str
        crop_size : int, optional
            The side length (in pixels) of square crops to produce
        """
        # TODO: implement openslide.OpenSlideCache for better performance?
        super().__init__()
        self.metadata = metadata
        self.random_hflip = RandomHorizontalFlip()
        self.random_vflip = RandomVerticalFlip()
        self.training = training
        self.crop_size = crop_size
        self.tif_dir = tif_dir
        self.outline_dir = outline_dir
        self.min_intersect_pct = min_intersect_pct

    def __len__(self) -> int:
        return len(self.metadata)

    def read_region(self, img: OpenSlide, crop: Rect) -> npt.NDArray[np.uint8]:
        """
        Reads a region of an image into memory and returns a 3-channel RGB image,
        channels last.

        Parameters
        ----------
        img : OpenSlide
            The image object to read
        crop : Rect
            The crop coordinates to read from the image

        Returns
        -------
        npt.NDArray[np.uint8]
            An RGB channels-last array of pixels.
        """
        x = np.array(img.read_region(location=(crop.x, crop.y), level=0, size=(crop.w, crop.h)))
        x = cv2.cvtColor(x, cv2.COLOR_RGBA2RGB)
        return x

    def _random_crop(self, img: OpenSlide, roi: Rect) -> npt.NDArray[np.uint8]:
        """
        Generates a RGB, channels-last array of shape (self.crop_size, self.crop_size, 3)
        from a random area of the provided ROI coordinates.

        Parameters
        ----------
        img : OpenSlide
            The image object to read
        roi : Rect
            The ROI coordinates from which to generate a random crop

        Returns
        -------
        npt.NDArray[np.uint8]
            The RGB, channels-last (HWC) pixel array.
        """
        x_offset = np.random.randint(low=0, high=roi.w - self.crop_size)
        y_offset = np.random.randint(low=0, high=roi.h - self.crop_size)
        return self.read_region(
            img=img,
            crop=Rect(
                x=roi.x + x_offset,
                y=roi.y + y_offset,
                w=self.crop_size,
                h=self.crop_size,
            ),
        )

    def valid_random_crop(self, outline: npt.NDArray[np.int32]) -> Tuple[int, int]:
        ctr_min = outline.min(axis=0)
        w, h = outline.max(axis=0) - ctr_min
        mask = cv2.fillPoly(img=np.zeros((w, h), dtype=np.uint8), pts=[outline - outline.min(axis=0)], color=1)
        i = 0
        while i < 200:
            x_offset = np.random.randint(low=0, high=w - self.crop_size, dtype=outline.dtype)
            y_offset = np.random.randint(low=0, high=h - self.crop_size, dtype=outline.dtype)
            # fmt: off
            crop = mask[y_offset:y_offset + self.crop_size, x_offset:x_offset + self.crop_size]
            # fmt: on
            if crop.mean() >= self.min_intersect_pct:
                return x_offset + ctr_min[0], y_offset + ctr_min[1]
            i += 1
        raise RuntimeError("Couldn't find a good crop!")

    def __getitem__(self, index: int) -> Union[Tuple[torch.Tensor, int], torch.Tensor]:
        """
        Produces one complete datum to be supplied to the model.
        Each datum contains the input (and target, if training).

        Parameters
        ----------
        index : int
            An index into `self.metadata`.

        Returns
        -------
        Union[Tuple[torch.Tensor, int], torch.Tensor]
            If a label column is present in self.metadata: (image_pixels, label_id)
            If not: image_pixels

        Raises
        ------
        ValueError
            If there's an issue with the image TIF file.
        """
        row = self.metadata.iloc[index]
        if row["h"] < self.crop_size or row["w"] < self.crop_size:
            raise ValueError("ROI is smaller than crop size!")
        img = OpenSlide(os.path.join(self.tif_dir, row["image_id"] + ".tif"))
        if img.level_count > 1:
            raise ValueError(f"Got {img.level_count} levels!")
        outline: npt.NDArray[np.int32] = np.load(
            os.path.join(self.outline_dir, row["image_id"], str(row["roi_num"]) + ".npy"),
            allow_pickle=False,
        )

        try:
            x, y = self.valid_random_crop(outline)
        except RuntimeError as e:
            raise RuntimeError(f"{e}\nimage: {row.to_dict()}")
        img = self.read_region(img=img, crop=Rect(x=x, y=y, w=self.crop_size, h=self.crop_size))

        # img = self.read_region(img=img, crop=Rect.from_mask(outline))
        img = normalize_background(img, I_0=np.array([row["I_0_R"], row["I_0_G"], row["I_0_B"]], dtype=img.dtype))
        cv2.bitwise_not(src=img, dst=img)
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)
        if self.training:
            img = self.random_hflip(img)
            img = self.random_vflip(img)

        # for dim in range(1, 3):
        #     img = img.unfold(dim, self.crop_size, self.crop_size)
        # img = img.permute(0, 3, 4, 1, 2).flatten(start_dim=3)
        if "label" in row.keys():
            label_id = LABEL_MAP[row["label"]]
            return img, label_id
        else:
            return img


class StratifiedBatchSampler(object):
    @staticmethod
    def get_class_weights(y: pd.Series) -> Dict[str, float]:
        cls_weights = len(y) / (y.nunique() * y.value_counts())
        return cls_weights.to_dict()

    def __init__(self, levels: pd.DataFrame, batch_size: int, seed: int) -> None:
        self.batch_size = batch_size
        self.p = np.ones(shape=len(levels), dtype=np.float32)
        for name, col in levels.iteritems():
            _col_cls_weight = self.get_class_weights(col)
            self.p *= col.map(_col_cls_weight.get).astype(np.float32).values
        self.p /= self.p.sum()
        self.rng = np.random.default_rng(seed)
        self.indices = np.arange(start=0, stop=len(self.p), dtype=np.int32)

    def __len__(self) -> int:
        return len(self.p) // self.batch_size

    def __iter__(self) -> Generator[npt.NDArray[np.int32], None, None]:
        indices = self.rng.choice(
            a=self.indices,
            size=len(self.p),
            replace=False,
            p=self.p,
            shuffle=True,
        )
        n_batches = len(self)
        for i in range(n_batches):
            batch_indices = indices[i::n_batches]
            yield batch_indices[: self.batch_size]
