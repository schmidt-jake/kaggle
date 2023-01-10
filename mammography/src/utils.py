from glob import glob
from logging import getLogger
from math import ceil, log2
from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import pandas as pd
from mpl_toolkits.axes_grid1 import ImageGrid
from pydicom import FileDataset, dcmread
from pydicom.multival import MultiValue
from pydicom.pixel_data_handlers.util import apply_windowing

logger = getLogger(__name__)


def find_filepath(image_id: int, glob_pattern: str = "mammography/data/raw/train_images/*/*.dcm") -> str:
    image_id = str(image_id)
    for filepath in glob(glob_pattern):
        if image_id in filepath:
            logger.info(f"Found {filepath=}")
            return filepath
    raise ValueError(f"Found no {filepath=}")


def maybe_flip_left(arr: npt.NDArray) -> npt.NDArray:
    """
    Flips `arr` horizontally if the sum of pixels on its left half is greater than its right.
    """
    # Standardize image laterality using pixel values b/c ImageLaterality meta is inaccurate
    w = arr.shape[1]
    if arr[:, : w // 2].sum() > arr[:, w // 2 :].sum():
        # if dcm.ImageLaterality == "R":
        arr = np.flip(arr, axis=1)
    return arr


def maybe_invert(arr: npt.NDArray, dcm: FileDataset) -> npt.NDArray:
    if dcm.PhotometricInterpretation == "MONOCHROME1":
        logger.info("Applying MONOCHROME1 correction")
        arr = 2**dcm.BitsStored - 1 - arr
    return arr


def convert_to_uint8(arr: npt.NDArray, dcm: FileDataset) -> npt.NDArray:
    arr = arr / (2**dcm.BitsStored - 1)
    arr = arr * (2**8 - 1)
    return arr.astype(np.uint8)


def dicom2numpy(
    filepath: str, should_apply_window_fn: bool = True, window_index: int = 0
) -> Tuple[npt.NDArray, FileDataset]:
    # adapted from https://www.kaggle.com/code/raddar/convert-dicom-to-np-array-the-correct-way/notebook
    dcm: FileDataset = dcmread(filepath)
    arr = dcm.pixel_array
    # if should_apply_window_fn:
    #     arr = apply_windowing(arr=arr, ds=dcm, index=window_index)
    # logger.debug(f"{arr.dtype=}\n{arr.min()=}\n{arr.max()=}\n{dcm.BitsStored=}")
    return arr, dcm


def get_suspected_bit_depth(arr: npt.NDArray) -> int:
    """
    Returns the smallest even base-2 exponent that is larger than `max(arr) + 1`.
    """
    suspected_bit_depth = ceil(log2(arr.max() + 1))
    suspected_bit_depth += suspected_bit_depth % 2
    return suspected_bit_depth


def plot_all_windows(arr: npt.NDArray, dcm: FileDataset, vmax_base2: Optional[int] = None) -> None:
    fig = plt.figure(figsize=(20, 20))
    center, width = dcm.get("WindowCenter", None), dcm.get("WindowWidth", None)
    if not (isinstance(center, MultiValue) and isinstance(width, MultiValue)):
        center, width = [center], [width]
    windows = pd.DataFrame({"center": center, "width": width})
    windows.dropna(inplace=True)
    windows.drop_duplicates(inplace=True)
    grid = ImageGrid(
        fig=fig,
        rect=111,  # similar to subplot(111)
        nrows_ncols=(1, 1 + len(windows)),
        axes_pad=(0.0, 0.0),  # pad between axes in inches
    )
    grid[0].imshow(arr, vmin=0, vmax=2**vmax_base2 - 1 if vmax_base2 is not None else None)
    grid[0].set_title(f"Raw\n{arr.min()=}\n{arr.max()=}\n{arr.dtype=}\n{dcm.BitsStored=}", fontsize="small")
    for ax, (index, window) in zip(grid[1:], windows.iterrows()):
        windowed = apply_windowing(arr=arr.copy(), ds=dcm, index=index)
        ax.imshow(windowed, vmin=0, vmax=2**vmax_base2 - 1 if vmax_base2 is not None else None)
        title = "\n".join(
            [
                f"{window['center']=}",
                f"{window['width']=}",
                f"{windowed.min()=:.2f}",
                f"{windowed.max()=:.2f}",
                f"{windowed.dtype}",
            ]
        )
        if np.allclose(arr, windowed):
            title += "\nIDENTICAL"
        ax.set_title(title, fontsize="small")


def plot_samples(
    *args: Tuple[pd.DataFrame, Optional[int], bool],
    n: int = 10,
    unique_patients: bool = True,
    random_seed: Optional[int] = None,
) -> None:
    fig = plt.figure(figsize=(20, 20))
    grid = ImageGrid(
        fig=fig,
        rect=111,  # similar to subplot(111)
        nrows_ncols=(len(args), n),
        axes_pad=(0.0, 1.0),  # pad between axes in inches
    )
    for (df, vmax_base2, should_apply_voi_lut), ax in zip(args, grid.axes_row):
        if unique_patients:
            df = df.groupby("patient_id").sample(n=1, random_seed=random_seed)
        sample = df.sample(n=n, random_seed=random_seed)
        for a, row in zip(ax, sample.to_dict("records")):
            filepath = find_filepath(row["image_id"])
            arr, dcm = dicom2numpy(filepath, should_apply_voi_lut=should_apply_voi_lut)
            # arr = cv2.resize(arr, (256, 256))
            a.imshow(arr, vmin=0, vmax=2**vmax_base2 - 1 if vmax_base2 is not None else None)
            title = "\n".join(
                [
                    f"image_id={row['image_id']}",
                    # f"VOILUT_fn={getattr(dcm, 'VOILUTFunction', None)}",
                    f"site_id={row['site_id']}",
                    f"machine_id={row['machine_id']}",
                    # f"view={row['view']}",
                    # f"laterality={row['laterality']}",
                    f"pixel_range={arr.min():.2f},{arr.max():.2f}",
                ]
            )
            if dcm.PhotometricInterpretation == "MONOCHROME1":
                title += "\nMONOCHROME CORRECTED"
            if hasattr(dcm, "WindowCenter"):
                title += f"\n{dcm.WindowCenter=}"
            if hasattr(dcm, "WindowWidth"):
                title += f"\n{dcm.WindowWidth=}"
            a.set_title(title, fontsize="small")