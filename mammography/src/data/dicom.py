from enum import Enum

import dicomsdl
import numexpr as ne
import numpy as np
import numpy.typing as npt

from mammography.src import utils


class VOILUTFunction(int, Enum):
    LINEAR = 0
    SIGMOID = 1


def get_windows(dcm: dicomsdl.DataSet) -> npt.NDArray[np.uint16]:
    centers, widths = dcm.WindowCenter, dcm.WindowWidth
    if not (isinstance(centers, list) and isinstance(widths, list)):
        centers, widths = [centers], [widths]
    windows = np.stack([centers, widths], axis=1)
    _, ix = np.unique(windows, axis=0, return_index=True)
    ix.sort()
    windows = windows[ix, :].astype(np.float32)
    return windows


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


def normalize(arr: npt.NDArray[np.float32], invert: bool) -> npt.NDArray[np.uint8]:
    pixel_bit_depth = utils.get_suspected_bit_depth(arr.max())
    ne.evaluate("arr * 255 / (2**pixel_bit_depth - 1)", out=arr)
    arr = np.rint(arr)
    arr = arr.astype(np.uint8)
    if invert:
        ne.evaluate("255 - arr", out=arr, casting="unsafe")
    arr = utils.maybe_flip_left(arr=arr)
    return arr


def process_dicom(filepath: str, raw: bool = False) -> npt.NDArray[np.uint]:
    dcm = dicomsdl.open(filepath)
    arr: npt.NDArray = dcm.pixelData(storedvalue=True)
    if raw:
        return arr
    ne.set_num_threads(1)
    if dcm.PixelRepresentation != 0:
        raise RuntimeError()
    windows = get_windows(dcm)
    windows = windows[[0], :]
    windows = windows.reshape(-1, 2, 1, 1)
    arr = np.broadcast_to(arr, shape=(windows.shape[0], *arr.shape))
    fn = getattr(dcm, "VOILUTFunction", "LINEAR")
    fn = (fn or "LINEAR").upper()
    fn = getattr(VOILUTFunction, fn)
    bits_stored = dcm.BitsStored
    if fn == VOILUTFunction.LINEAR:
        arr = linear(arr=arr, center=windows[:, 0], width=windows[:, 1], bits_stored=bits_stored)
    elif fn == VOILUTFunction.SIGMOID:
        arr = sigmoid(arr=arr, center=windows[:, 0], width=windows[:, 1], bits_stored=bits_stored)
    return normalize(arr, invert=dcm.PhotometricInterpretation == "MONOCHROME1")
