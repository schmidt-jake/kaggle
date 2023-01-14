import os
from glob import glob
from multiprocessing.pool import Pool

import cv2
import numpy as np
import numpy.typing as npt
from pydicom import dcmread
from pydicom.pixel_data_handlers.util import apply_voi_lut
from tqdm import tqdm


def breast_mask(img: npt.NDArray[np.uint16]) -> npt.NDArray[np.uint16]:
    assert img.dtype == np.uint16
    thresh, mask = cv2.threshold(img, thresh=5, maxval=1, type=cv2.THRESH_BINARY)
    # kernel = cv2.getStructuringElement(shape=cv2.MORPH_ELLIPSE, ksize=(15, 15))
    # mask = cv2.dilate(mask, kernel, iterations=15)
    contours = cv2.findContours(mask.astype(np.uint8), mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE)[0]
    max_contour = max(contours, key=cv2.contourArea)
    return cv2.drawContours(
        image=np.zeros_like(img, dtype=np.uint8), contours=[max_contour], contourIdx=0, color=1, thickness=-1
    )


def crop2mask(img: npt.NDArray[np.uint8], mask: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    x, y, w, h = cv2.boundingRect(mask)
    cropped = img[y : y + h, x : x + w]
    return cropped


def breast_crop(img: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    mask = breast_mask(img)
    img = img * mask
    return crop2mask(img, mask)


def standardize(arr, dcm):
    arr = to_uint8(arr, dcm)
    # https://escapetech.eu/manuals/qmedical/commands/index_Values_of_Interest__.html
    if dcm["PhotometricInterpretation"].value == "MONOCHROME1":
        # https://dicom.nema.org/medical/Dicom/2017c/output/chtml/part03/sect_C.7.6.3.html#sect_C.7.6.3.1.2
        cv2.bitwise_not(arr, dst=arr)
    if dcm["ImageLaterality"].value == "L":
        arr = np.flip(arr, axis=1)
    return arr


def dicom2numpy(filepath: str) -> npt.NDArray[np.uint16]:
    dcm = dcmread(filepath)
    arr = dcm.pixel_array
    arr = breast_crop(arr)
    if "VOILUTFunction" in dcm:
        arr = apply_voi_lut(arr, dcm)
    return standardize(arr, dcm)


def to_uint8(arr, dcm):
    arr = arr.astype(np.float32)
    arr /= arr.max() / 255.0
    # arr /= (2 ** dcm["BitsStored"].value - 1) / (2**8 - 1)
    arr = arr.astype(np.uint8)
    return arr


def process_image(filepath: str) -> None:
    cv2.setNumThreads(0)
    image_id = filepath.split("/")[-1][:-4]
    save_path = os.path.join("mammography/data/uint8_crops_v2/png", f"{image_id}.png")
    if os.path.exists(save_path):
        return
    arr = dicom2numpy(filepath)
    cv2.imwrite(save_path, arr)


def main() -> None:
    filepaths = glob("mammography/data/raw/train_images/*/*.dcm")
    with Pool(8) as pool:
        pool.map(process_image, tqdm(filepaths))


if __name__ == "__main__":
    main()
