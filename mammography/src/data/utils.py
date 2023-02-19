import re
from logging import getLogger
from typing import Any, Dict, List, Set

import cv2
import numpy as np
import numpy.typing as npt
import torch
from torch.utils.data import default_collate
from torchvision.transforms import functional_tensor

logger = getLogger(__name__)

DICOM_FILEPATH = re.compile(r"^[\w\-/]+/(?P<patient_id>\d+)/(?P<image_id>\d+)\.dcm$")


def resize(img: npt.NDArray[np.uint], max_size: int) -> npt.NDArray[np.uint]:
    height, width = img.shape[0], img.shape[1]
    f = max_size / min(height, width)  # resizing factor
    dim = (round(width * f), round(height * f))
    resized = cv2.resize(src=img, dsize=dim)
    return resized


def convert_bit_depth(arr: npt.NDArray[np.uint], input_bit_depth: int, output_bit_depth: int) -> npt.NDArray[np.uint]:
    if input_bit_depth == output_bit_depth:
        return arr
    if np.issubdtype(arr.dtype, np.integer):
        arr = arr.astype(np.float32)
    # ne.evaluate("arr * (2 ** output_bit_depth - 1) / (2**input_bit_depth - 1)", out=arr)
    arr *= (2**output_bit_depth - 1) / (2**input_bit_depth - 1)
    arr = np.rint(arr)
    arr = arr.astype(getattr(np, f"uint{output_bit_depth}"))
    return arr


def extract_image_id_from_filepath(filepath: str) -> int:
    match = re.match(DICOM_FILEPATH, filepath)
    if match is not None:
        return int(match.group("image_id"))
    raise ValueError(f"Not a valid dicom path: {filepath}")


def get_filepath(meta: Dict[str, Any], template: str) -> str:
    return template.format(**meta)


def select_keys(d: Dict[str, Any], keys: Set[str]) -> Dict[str, Any]:
    return {k: v for k, v in d.items() if k in keys}


def read_png(filepath: str) -> npt.NDArray[np.uint]:
    arr = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    if arr is None:
        raise RuntimeError(f"No data found at {filepath}")
    return arr


def maybe_flip_left(arr: npt.NDArray) -> npt.NDArray:
    """
    Flips `arr` horizontally if the sum of pixels on its left half is greater than its right.
    """
    # Standardize image laterality using pixel values b/c ImageLaterality meta is inaccurate
    split = arr.shape[-1] // 2
    left, right = arr[..., :split], arr[..., split:]
    if left.sum() > right.sum():
        arr = arr[..., ::-1].copy()
    return arr


def extract_largest_object_mask(mask: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """
    Extracts the largest object from `mask` and closes any holes in it.
    """
    contours = cv2.findContours(mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)[0]
    max_contour = max(contours, key=cv2.contourArea)
    # perimeter = cv2.arcLength(max_contour, closed=True)
    # max_contour = cv2.approxPolyDP(max_contour, epsilon=0.01 * perimeter, closed=True)
    return cv2.drawContours(image=np.zeros_like(mask), contours=[max_contour], contourIdx=0, color=1, thickness=-1)


def mask_pct_foreground(mask: npt.NDArray) -> float:
    x, y, w, h = cv2.boundingRect(mask)
    mask = mask[y : y + h, x : x + w]
    return mask.mean()


def get_length_scale(mask):
    x, y, w, h = cv2.boundingRect(mask)
    area = w * h
    length_scale = np.sqrt(area)
    return length_scale


def breast_mask(img: npt.NDArray[np.uint]) -> npt.NDArray[np.uint8]:
    max_thresh = 50.0
    # if img.dtype is np.uint16:
    #     max_thresh *= (2**16 - 1) / (2**8 - 1)
    thresh, mask = cv2.threshold(img, thresh=5, maxval=1, type=cv2.THRESH_TRIANGLE)
    if thresh > max_thresh:
        logger.warning(f"Got suspiciously high threshold of {thresh}, using 5.0 instead...")
        thresh, mask = cv2.threshold(img, thresh=5, maxval=1, type=cv2.THRESH_BINARY)
    mask = extract_largest_object_mask(mask)
    mask_pct = mask_pct_foreground(mask)
    if mask_pct < 0.8:
        ksize = round(get_length_scale(mask) / 10.0)
        kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(ksize, ksize))
        mask_candidate = cv2.morphologyEx(mask, op=cv2.MORPH_OPEN, kernel=kernel, iterations=2)
        delta = mask_pct_foreground(mask_candidate) - mask_pct
        if delta > 0.0:
            logger.info(f"Improved mask foreground by {delta:.1%} using ksize {ksize}")
            mask = extract_largest_object_mask(mask_candidate)
    return mask


def crop_and_mask(img: npt.NDArray[np.uint8], mask: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    x, y, w, h = cv2.boundingRect(mask)
    cropped = img[y : y + h, x : x + w]
    mask = mask[y : y + h, x : x + w]
    return cropped * mask


def pad_ragged_image_batch(images: List[torch.Tensor]) -> torch.Tensor:
    # https://discuss.pytorch.org/t/whats-the-fastest-way-to-prepare-a-batch-from-different-sized-images-by-padding/119568
    max_h = max([img.shape[-2] for img in images])
    max_w = max([img.shape[-1] for img in images])
    for i, img in enumerate(images):
        pad_left, pad_right, pad_top, pad_bottom = 0, 0, 0, 0
        diff_w = max_w - img.shape[-1]
        diff_h = max_h - img.shape[-2]
        if diff_w > 0:
            pad_left = diff_w
        if diff_h > 0:
            pad_top = diff_h // 2
            pad_bottom = diff_h - pad_top
        if any([pad_left, pad_right, pad_top, pad_bottom]):
            images[i] = torch.nn.functional.pad(input=img, pad=(pad_left, pad_right, pad_top, pad_bottom))
    return torch.stack(images, dim=0)


def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    # convert list of dicts to dict of lists
    batch: Dict[str, List[Any]] = {k: [elem[k] for elem in batch] for k in batch[0].keys()}
    # collate each dict item
    return {k: pad_ragged_image_batch(v) if k in ["CC", "MLO"] else default_collate(v) for k, v in batch.items()}
