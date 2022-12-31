from glob import glob
from typing import Any, Dict

import pandas as pd
from pydicom import dcmread
from pydicom.multival import MultiValue
from tqdm import tqdm


def read_dicom_file(filepath: str) -> Dict[str, Any]:
    image_id = filepath.split("/")[-1][:-4]
    row: Dict[str, Any] = {"image_id": image_id}
    with dcmread(filepath) as dicom:
        for key in dicom.dir():
            if "pixeldata" in key.lower():
                continue
            val = dicom[key].value
            if isinstance(val, (str, int, float)):
                row[key] = val
            elif isinstance(val, MultiValue):
                row[key] = list(val)
            else:
                print(key, type(val))
    return row


def main() -> None:
    meta = pd.DataFrame([read_dicom_file(filepath) for filepath in tqdm(glob("mammography/data/*_images/*/*.dcm"))])
    meta.to_csv("mammography/dicom_metadata.csv", index=False)


if __name__ == "__main__":
    main()
