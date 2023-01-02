from glob import glob
from multiprocessing.pool import Pool

import cv2
import pandas as pd
from tqdm import tqdm


def inspect_png(filepath: str):
    cv2.setNumThreads(0)
    image_id = filepath.split("/")[-1][:-4]
    arr = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    h, w = arr.shape
    ptp = arr.ptp()
    return {"image_id": image_id, "h": h, "w": w, "ptp": ptp}


def main() -> None:
    with Pool(8) as pool:
        x = pd.DataFrame(pool.map(inspect_png, tqdm(glob("mammography/data/uint8_crops/png/*.png"))))
    x.to_csv("inspect.csv")


if __name__ == "__main__":
    main()
