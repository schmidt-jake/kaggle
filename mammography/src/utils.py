import re
from glob import glob
from logging import getLogger

# import dicomsdl
# import matplotlib.pyplot as plt
import numexpr as ne
import numpy as np
import numpy.typing as npt

# import pandas as pd
import torch

# from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.transforms import functional_tensor

# from typing import Optional, Tuple


logger = getLogger(__name__)

DICOM_FILEPATH = re.compile(r"^[\w\-/]+/(?P<patient_id>\d+)/(?P<image_id>\d+)\.dcm$")


def inspect_module(module: torch.nn.Module) -> None:
    train_nodes, eval_nodes = get_graph_node_names(module)
    for node in train_nodes:
        try:
            m = module.get_submodule(node)
            print(node, m)
        except AttributeError:
            print(node)


def extract_image_id_from_filepath(filepath: str) -> int:
    match = re.match(DICOM_FILEPATH, filepath)
    if match is not None:
        return int(match.group("image_id"))
    raise ValueError(f"Not a valid dicom path: {filepath}")


def find_filepath(image_id: int, glob_pattern: str = "mammography/data/raw/train_images/*/{image_id}.dcm") -> str:
    for filepath in glob(glob_pattern.format(image_id=image_id)):
        return filepath
    raise ValueError(f"Found no filepath={filepath}")


def maybe_flip_left(arr: npt.NDArray) -> npt.NDArray:
    """
    Flips `arr` horizontally if the sum of pixels on its left half is greater than its right.
    """
    # Standardize image laterality using pixel values b/c ImageLaterality meta is inaccurate
    w = arr.shape[-1]
    left, right = arr[..., : w // 2], arr[..., w // 2 :]
    if ne.evaluate("sum(left)") > ne.evaluate("sum(right)"):
        arr = arr[..., ::-1]
    return arr


def get_suspected_bit_depth(pixel_value: int) -> int:
    """
    Returns the smallest even base-2 exponent that is larger than `pixel_value + 1`.
    """
    suspected_bit_depth = np.ceil(np.log2(pixel_value + 1))
    suspected_bit_depth += suspected_bit_depth % 2
    return suspected_bit_depth.astype(np.float32)


# def maybe_invert(arr: npt.NDArray, dcm: FileDataset) -> npt.NDArray:
#     if dcm.PhotometricInterpretation == "MONOCHROME1":
#         # logger.info("Applying MONOCHROME1 correction")
#         arr = 2**dcm.BitsStored - 1 - arr
#     return arr


# def scale_to_01(arr: npt.NDArray[np.uint16], dcm: FileDataset) -> npt.NDArray[np.float32]:
#     arr = arr.astype(np.float32) / (2**dcm.BitsStored - 1)
#     return arr


def crop_right_center(img: torch.Tensor, size: int) -> torch.Tensor:
    """
    Takes a crop that is on the right side of the arr, horizontally center.
    If needed, adds padding to the left, top, and bottom.
    """
    w, h = functional_tensor.get_image_size(img)
    top = (h - size) // 2
    left = w - size
    cropped = functional_tensor.crop(img=img, top=top, left=left, height=size, width=size)
    return cropped


# @jit(nopython=True)
def to_bit_depth(arr: npt.NDArray[np.uint16], src_depth: int, dest_depth: int) -> npt.NDArray:
    scale_factor = (2**dest_depth - 1) / (2**src_depth - 1)
    arr = arr.astype(np.float32) * scale_factor
    return np.rint(arr).astype(np.uint8)


# def dicom2numpy(filepath: str) -> Tuple[npt.NDArray, FileDataset]:
#     dcm: FileDataset = dcmread(filepath)
#     arr = dcm.pixel_array
#     return arr, dcm


# def get_unique_windows(dcm: dicomsdl.DataSet) -> pd.DataFrame:
#     center, width = dcm.get("WindowCenter", None), dcm.get("WindowWidth", None)
#     if not (isinstance(center, MultiValue) and isinstance(width, MultiValue)):
#         center, width = [center], [width]
#     windows = pd.DataFrame({"center": center, "width": width})
#     windows.dropna(inplace=True)
#     windows.drop_duplicates(inplace=True)
#     return windows


# def plot_all_windows(arr: npt.NDArray, dcm: FileDataset, vmax_base2: Optional[int] = None) -> None:
#     fig = plt.figure(figsize=(20, 20))
#     windows = get_unique_windows(dcm)
#     grid = ImageGrid(
#         fig=fig,
#         rect=111,  # similar to subplot(111)
#         nrows_ncols=(1, 1 + len(windows)),
#         axes_pad=(0.0, 0.0),  # pad between axes in inches
#     )
#     grid[0].imshow(arr, vmin=0, vmax=2**vmax_base2 - 1 if vmax_base2 is not None else None)
#     grid[0].set_title(
#         f"Raw\nmin={arr.min()}\nmax={arr.max()}\ndtype={arr.dtype}\nBitsStored={dcm.BitsStored}", fontsize="small"
#     )
#     for ax, (index, window) in zip(grid[1:], windows.iterrows()):
#         windowed = apply_windowing(arr=arr.copy(), ds=dcm, index=index)
#         ax.imshow(windowed, vmin=0, vmax=2**vmax_base2 - 1 if vmax_base2 is not None else None)
#         title = "\n".join(
#             [
#                 f"window center={window['center']}",
#                 f"window width{window['width']}",
#                 f"min={windowed.min():.2f}",
#                 f"max={windowed.max():.2f}",
#                 f"{windowed.dtype}",
#             ]
#         )
#         if np.allclose(arr, windowed):
#             title += "\nIDENTICAL"
#         ax.set_title(title, fontsize="small")


# def plot_arr(arr: npt.NDArray, dcm: FileDataset, ax: plt.Axes, vmax_base2: Optional[int] = None, **meta) -> None:
#     ax.imshow(arr, vmin=0, vmax=2**vmax_base2 - 1 if vmax_base2 is not None else None, cmap="gray")
#     title = [f"{k}={v}" for k, v in meta.items()]
#     title.append(f"pixel_range={arr.min():.2f},{arr.max():.2f}")
#     if dcm.PhotometricInterpretation == "MONOCHROME1":
#         title.append("\nMONOCHROME CORRECTED")
#     if hasattr(dcm, "WindowCenter"):
#         title.append(f"\nwindow center={dcm.WindowCenter}")
#     if hasattr(dcm, "WindowWidth"):
#         title.append(f"window width=\n{dcm.WindowWidth}")
#     title = "\n".join(title)
#     ax.set_title(title, fontsize="small")


# def plot_samples(
#     *args: Tuple[pd.DataFrame, Optional[int], bool],
#     n: int = 10,
#     unique_patients: bool = True,
#     random_seed: Optional[int] = None,
# ) -> None:
#     fig = plt.figure(figsize=(20, 20))
#     grid = ImageGrid(
#         fig=fig,
#         rect=111,  # similar to subplot(111)
#         nrows_ncols=(len(args), n),
#         axes_pad=(0.0, 1.0),  # pad between axes in inches
#     )
#     for (df, vmax_base2, should_apply_window_fn), ax in zip(args, grid.axes_row):
#         if unique_patients:
#             df = df.groupby("patient_id").sample(n=1, random_state=random_seed)
#         sample = df.sample(n=n, random_state=random_seed)
#         for a, row in zip(ax, sample.to_dict("records")):
#             filepath = find_filepath(row["image_id"])
#             arr, dcm = dicom2numpy(filepath, should_apply_window_fn=should_apply_window_fn)
#             # arr = cv2.resize(arr, (256, 256))
#             plot_arr(arr=arr, dcm=dcm, ax=a, vmax_base2=vmax_base2, **row)
