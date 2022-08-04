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
from scipy.stats import mode
import torch
from torch.utils.data import Dataset
from torchvision.transforms import RandomHorizontalFlip
from torchvision.transforms import RandomVerticalFlip
from torchvision.transforms import Resize

from mayo_clinic_strip_ai.find_ROIs import Rect

# from mayo_clinic_strip_ai.stain import normalize_staining

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
        final_size: int,
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
        self.resize = Resize(size=final_size)
        self.training = training
        self.crop_size = crop_size
        self.tif_dir = tif_dir
        self.outline_dir = outline_dir

    def __len__(self) -> int:
        return len(self.metadata)

    @staticmethod
    def _read_region(img: OpenSlide, crop: Rect) -> npt.NDArray[np.uint8]:
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
        x = img.read_region(
            location=(crop.x, crop.y),
            size=(crop.w, crop.h),
            level=0,
        )
        x = np.array(x)
        x = cv2.cvtColor(x, cv2.COLOR_RGBA2RGB)
        # x, H, E = normalize_staining(x)
        x = cv2.bitwise_not(x)
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
        return self._read_region(
            img=img,
            crop=Rect(
                x=roi.x + x_offset,
                y=roi.y + y_offset,
                w=self.crop_size,
                h=self.crop_size,
            ),
        )

    def valid_random_crop(self, img: OpenSlide, roi: Rect) -> npt.NDArray[np.uint8]:
        """
        Generates a RGB, channels-last array of shape (self.crop_size, self.crop_size, 3)
        from a random area of the provided ROI coordinates, trying to ensure that the crop doesn't
        have too much dead space / background.

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
        x = self._random_crop(img=img, roi=roi)
        i = 0
        while mode(x, axis=None, keepdims=False).mode < 20.0:
            if i > 10:
                print("Couldn't find a good crop!")
                break
            x = self._random_crop(img=img, roi=roi)
            i += 1
        return x

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
            mmap_mode="r",
            allow_pickle=False,
        )

        M = cv2.moments(outline)
        cx = int(M["m10"] / M["m00"])
        cy = int(M["m01"] / M["m00"])
        img = self._read_region(
            img=img,
            crop=Rect(
                x=cx - (self.crop_size // 2),
                y=cy - (self.crop_size // 2),
                w=self.crop_size,
                h=self.crop_size,
            ),
        )

        # roi = Rect.from_mask(outline)  # type: ignore[arg-type]
        # if self.training:
        #     img = self.valid_random_crop(img=img, roi=roi)
        # else:
        #     x_offset = (roi.w - self.crop_size) // 2
        #     y_offset = (roi.h - self.crop_size) // 2
        #     img = self._read_region(
        #         img=img,
        #         crop=Rect(
        #             x=roi.x + x_offset,
        #             y=roi.y + y_offset,
        #             w=self.crop_size,
        #             h=self.crop_size,
        #         ),
        #     )
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1)
        img = self.resize(img)
        if self.training:
            img = self.random_hflip(img)
            img = self.random_vflip(img)
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

    def __init__(self, levels: pd.DataFrame, batch_size: int) -> None:
        self.batch_size = batch_size
        self.p = np.ones(shape=len(levels), dtype=np.float32)
        for name, col in levels.iteritems():
            _col_cls_weight = self.get_class_weights(col)
            self.p *= col.map(_col_cls_weight.get).astype(np.float32).values
        self.p /= self.p.sum()
        self.rng = np.random.default_rng()
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
