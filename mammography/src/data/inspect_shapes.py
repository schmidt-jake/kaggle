from argparse import ArgumentParser
from glob import glob
from multiprocessing.pool import Pool

import cv2
import pandas as pd
from tqdm import tqdm


def inspect(filepath):
    arr = cv2.imread(filepath, cv2.IMREAD_UNCHANGED)
    return {
        "image_id": int(filepath.split("/")[-1][:-4]),
        "h": arr.shape[0],
        "w": arr.shape[1],
        "background_pct": (arr <= 5.0).mean(),
    }


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("glob_pattern", type=str)
    parser.add_argument("output_path", type=str)
    args = parser.parse_args()
    filepaths = glob(args.glob_pattern)
    with Pool() as pool:
        meta = pd.DataFrame(tqdm(pool.imap_unordered(inspect, filepaths), total=len(filepaths)))
    meta.to_csv(args.output_path, index=False)
    # image_ids = set(
    #     image_id
    #     for _, row in pd.concat(
    #         [pd.read_json("mammography/data/resized2/train.json"), pd.read_json("mammography/data/resized2/val.json")]
    #     ).iterrows()
    #     for image_id in row["CC"] + row["MLO"]
    # )
    # for f in tqdm(glob("mammography/data/resized2/*.png")):
    #     image_id = int(f.split("/")[-1][:-4])
    #     if image_id not in image_ids:
    #         print(f"Removing {f}...")
    #         os.remove(f)
