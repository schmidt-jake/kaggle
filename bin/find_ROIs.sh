#!/bin/bash

set -euo pipefail

export OPENCV_IO_MAX_IMAGE_PIXELS=999999999999999

ROOT_DATA_DIR=mayo_clinic_strip_ai/data

for dataset in train test
do
    echo "Starting on $dataset..."
    python -m mayo_clinic_strip_ai.find_ROIs $ROOT_DATA_DIR/$dataset/ $ROOT_DATA_DIR/$dataset.csv $ROOT_DATA_DIR/ROIs/$dataset
done
