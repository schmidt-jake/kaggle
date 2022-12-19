#!/bin/bash

set -euo pipefail

export OPENCV_IO_MAX_IMAGE_PIXELS=999999999999999

ROOT_DATA_DIR=mayo_clinic_strip_ai/data

for dataset in train test
do
    echo "Starting on $dataset..."
    # mprof run --interval=5 --output=$dataset.dat --multiprocess \
    python mayo_clinic_strip_ai/find_ROIs.py \
        input_filepath=$ROOT_DATA_DIR/$dataset.csv \
        data_dir=$ROOT_DATA_DIR/$dataset/
done
