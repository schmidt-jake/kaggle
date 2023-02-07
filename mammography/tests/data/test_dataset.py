from functools import partial
from operator import itemgetter

import numpy as np
import pandas as pd

from mammography.src.data import utils
from mammography.src.data.dataset import DataframeDataPipe, map_fn
from mammography.src.data.dicom import process_dicom
from mammography.src.data.metadata import get_breast_metadata


def test_dataset_dicom() -> None:
    meta = pd.read_csv("mammography/data/raw/test.csv")
    meta = get_breast_metadata(meta)
    print(meta.info())

    fns = []
    for view in ["CC", "MLO"]:
        fns.extend(
            [
                map_fn(np.random.choice, input_key=view, output_key=view),
                map_fn(
                    partial(
                        utils.get_filepath, template=f"mammography/data/raw/test_images/{{patient_id}}/{{{view}}}.dcm"
                    ),
                    output_key=view,
                ),
                map_fn(process_dicom, input_key=view, output_key=view),
                map_fn(itemgetter(0), input_key=view, output_key=view),
            ]
        )

    pipe = DataframeDataPipe(df=meta, fns=fns)
    print(pipe)
    for i in range(len(pipe)):
        print(pipe[i])
