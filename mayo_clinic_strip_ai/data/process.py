import logging
import os
from subprocess import CalledProcessError
from typing import Any, List, Tuple

os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = str(2**40)
import cv2
from numcodecs import Blosc
import numpy as np
import numpy.typing as npt
from pandas import Series
import zarr

from mayo_clinic_strip_ai.data.metadata import Metadata

logger = logging.getLogger(__name__)

compressor = Blosc(
    cname="zstd",
    clevel=9,
    shuffle=Blosc.AUTOSHUFFLE,
)


def get_contours(img: npt.NDArray[np.uint8]) -> Tuple[npt.NDArray[np.int32], ...]:
    x = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    x = cv2.GaussianBlur(x, ksize=(201, 201), sigmaX=0)
    x = cv2.threshold(x, thresh=250, maxval=255, type=cv2.THRESH_BINARY_INV)[1]
    contours = cv2.findContours(x, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_SIMPLE)[0]
    return contours


def get_background_rows(img: npt.NDArray[np.uint8]) -> Tuple[npt.NDArray, float]:
    # img is HWC

    gray: npt.NDArray[np.uint8] = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    thresh: float = cv2.threshold(gray, thresh=0, maxval=255, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)[0]
    is_background = (gray >= thresh).all(axis=1)
    return is_background, thresh


def get_splits(
    img: npt.NDArray[np.uint8],
    background_rows: npt.NDArray,
    min_gap: int = 101,
) -> List[npt.NDArray[np.uint8]]:
    # use a sliding window to smooth results
    background_rows = np.lib.stride_tricks.sliding_window_view(
        background_rows,
        window_shape=min_gap,
        writeable=False,
    ).all(axis=1)
    background_rows = np.pad(background_rows, pad_width=min_gap // 2, mode="edge")
    dx = np.diff(background_rows, n=1)
    i = np.flatnonzero(dx)
    splits = np.split(img, i + 1, axis=0)
    return splits


def remove_background(img: npt.NDArray[np.uint8], min_gap: int = 101) -> List[npt.NDArray[np.uint8]]:
    # img is HW
    background_rows, thresh = get_background_rows(img=img)
    logger.debug(f"Initial threshold: {thresh}")
    splits = get_splits(img=img, background_rows=background_rows, min_gap=min_gap)
    logger.debug(f"Got {len(splits)} raw splits")
    splits = [split for split in splits if (split <= thresh).any()]
    logger.debug(f"Got {len(splits)} final splits")
    return splits


def get_img(meta: Metadata, index: int) -> npt.NDArray[np.uint8]:
    if not os.path.exists(os.path.join(*meta.local_img_path(index))):
        meta.download_img(index)
    img = meta.load_img(index)
    return img


def finalize_img(img: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    gray: npt.NDArray[np.uint8] = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    thresh, binary = cv2.threshold(gray, thresh=0, maxval=255, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    logger.debug(f"Got final threshold {thresh}")
    binary = cv2.bitwise_not(binary)
    binary = cv2.morphologyEx(
        binary,
        op=cv2.MORPH_OPEN,
        kernel=cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(13, 13)),
        iterations=2,
    )
    x, y, w, h = cv2.boundingRect(binary)
    crop = img[y : y + h, x : x + w, :]  # noqa: E203
    return crop


def write_zarr(img: npt.NDArray, path: str, grp: zarr.Group, **attrs: Any) -> None:
    z: zarr.Array = grp.array(
        name=path,
        data=img,
        shape=img.shape,
        dtype=img.dtype,
        chunks=(20_000, 20_000, 3),
        read_only=True,
        compressor=compressor,
        overwrite=True,
    )
    z.attrs.update(**attrs)


# def write_tif():
#     out_dir = os.path.join(
#         "mayo_clinic_strip_ai/data/out",
#         meta.name,
#         row["image_id"],
#     )
#     os.makedirs(out_dir, exist_ok=True)
#     cv2.imwrite(
#         os.path.join(out_dir, f"{j}.tif"),
#         crop,
#     )


def process_img(img: npt.NDArray[np.uint8], row: Series, grp: zarr.Group) -> None:
    if img.ndim != 3:
        raise ValueError(f"Got wrong number of dimensions: {img.ndim}")

    if img.shape[2] != 3:
        raise ValueError("Image isn't RGB or channels last")

    if img.shape[1] > img.shape[0]:
        logger.debug("Transposing image")
        img = img.transpose(1, 0, 2)

    splits = remove_background(img=img, min_gap=1001)
    if len(splits) == 0:
        raise ValueError("Found no splits!")
    elif len(splits) > 3:
        raise ValueError(f"Found too many ({len(splits)}) splits!")
    for j, split in enumerate(splits):
        crop = finalize_img(split)
        write_zarr(
            img=crop,
            path=os.path.join(row["image_id"], f"{j}.zarr"),
            grp=grp,
            split=j,
            **row.to_dict(),
        )
        logger.debug("split saved.")
    logger.debug("Done with image!")


if __name__ == "__main__":
    logging.basicConfig(
        filename="log.txt",
        filemode="w",
        level=logging.DEBUG,
        format="%(asctime)s:%(levelname)s:%(name)s:%(funcName)s:%(lineno)d:%(message)s",
    )

    meta = Metadata.from_csv(
        filepath="train.csv",
        data_dir="mayo_clinic_strip_ai/data/",
    )
    root: zarr.Group = zarr.group("mayo_clinic_strip_ai/data/zarrs/")
    with root:
        with root.create_group(meta.name) as grp:
            for i, row in meta.iterrows():
                logger.debug(f"Starting on image {i}...")
                try:
                    img = get_img(meta, i)
                    process_img(img, row=meta.iloc[i], grp=grp)
                except (ValueError, CalledProcessError) as e:
                    logger.error(f"Error with {os.path.join(*meta.local_img_path(i))}\n{e}")
