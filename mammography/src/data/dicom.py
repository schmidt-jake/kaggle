from enum import Enum
from logging import getLogger

import cv2
import dicomsdl
import numexpr as ne
import numpy as np
import numpy.typing as npt
import pandas as pd

from mammography.src.data.utils import convert_bit_depth, maybe_flip_left
from mammography.src.utils import get_suspected_bit_depth

logger = getLogger(__name__)


class VOILUTFunction(int, Enum):
    LINEAR = 0
    SIGMOID = 1


def get_windows(dcm: dicomsdl.DataSet) -> npt.NDArray[np.uint16]:
    centers, widths = dcm.WindowCenter, dcm.WindowWidth
    if not (isinstance(centers, list) and isinstance(widths, list)):
        centers, widths = [centers], [widths]
    windows = pd.DataFrame({"center": centers, "width": widths})
    windows.drop_duplicates(inplace=True)
    windows.sort_values("width", inplace=True, ascending=False)
    return windows.values


def linear(
    arr: npt.NDArray[np.float32], center: npt.NDArray[np.float32], width: npt.NDArray[np.float32], bits_stored: int
) -> npt.NDArray[np.float32]:
    y_max = 2**bits_stored - 1
    below = ne.evaluate("arr <= center - 0.5 - (width - 1) / 2")
    above = ne.evaluate("arr > center - 0.5 + (width - 1) / 2")
    arr = ne.evaluate("where(below, 0, arr)")
    arr = ne.evaluate("where(above, y_max, arr)")
    arr = ne.evaluate("where(~(below|above), ((arr - (center - 0.5)) / (width - 1) + 0.5) * y_max, arr)")
    return arr.astype(np.float32)


def sigmoid(
    arr: npt.NDArray[np.float32], center: npt.NDArray[np.float32], width: npt.NDArray[np.float32], bits_stored: int
) -> npt.NDArray[np.float32]:
    y_max = 2**bits_stored - 1
    arr = ne.evaluate("y_max / (1 + exp(-4 * (arr - center) / width))")
    return arr


def normalize(
    arr: npt.NDArray, invert: bool, output_bit_depth: int, input_bit_depth: int = None
) -> npt.NDArray[np.uint]:
    if input_bit_depth is None:
        input_bit_depth = get_suspected_bit_depth(arr.max())
    # input_bit_depth = 16
    if input_bit_depth != output_bit_depth:
        arr = convert_bit_depth(arr=arr, input_bit_depth=input_bit_depth, output_bit_depth=output_bit_depth)
    if invert:
        arr = cv2.bitwise_not(arr)
    arr = maybe_flip_left(arr=arr)
    return arr


def equalize_hist_mask(img: npt.NDArray, mask: npt.NDArray):
    coord = np.where(mask == 1)
    pixels = img[coord]
    equalized_pixels = cv2.equalizeHist(pixels)
    for i, C in enumerate(zip(coord[0], coord[1])):
        img[C[0], C[1]] = equalized_pixels[i][0]


def invert_mask(img: npt.NDArray, mask: npt.NDArray):
    return np.where(mask, 255 - img, img)


def process_dicom(
    filepath: str, raw: bool = False, output_bit_depth: int = 8, scale_factor: float = 1 / 4
) -> npt.NDArray[np.uint]:
    dcm = dicomsdl.open(filepath)
    arr: npt.NDArray = dcm.pixelData(storedvalue=True)
    arr = cv2.resize(arr, dsize=None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
    if dcm.PixelRepresentation != 0:
        raise RuntimeError()
    if raw:
        return normalize(arr, invert=dcm.PhotometricInterpretation == "MONOCHROME1", output_bit_depth=output_bit_depth)
    # ne.set_num_threads(1)
    windows = get_windows(dcm)
    # windows = windows[[0], :]
    # windows = windows.reshape(-1, 2, 1, 1)
    # arr = np.broadcast_to(arr, shape=(windows.shape[0], *arr.shape))
    fn = getattr(dcm, "VOILUTFunction", "LINEAR")
    fn = (fn or "LINEAR").upper()
    fn = getattr(VOILUTFunction, fn)
    bits_stored = dcm.BitsStored
    if fn == VOILUTFunction.LINEAR:
        arr = linear(arr=arr, center=windows[0, 0], width=windows[0, 1], bits_stored=bits_stored)
    elif fn == VOILUTFunction.SIGMOID:
        arr = sigmoid(arr=arr, center=windows[0, 0], width=windows[0, 1], bits_stored=bits_stored)
    return normalize(arr, invert=dcm.PhotometricInterpretation == "MONOCHROME1", output_bit_depth=output_bit_depth)
