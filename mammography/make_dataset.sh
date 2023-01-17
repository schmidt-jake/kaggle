#!/bin/bash

set -euo pipefail

# Create PNGs
python -m mammography.src.preprocess_images

tar \
    -C mammography/data/png \
    --use-compress-program="pigz --processes 8 --best --verbose" \
    -cvf mammography/data/uint8_crops/png.tar.gz .

kaggle datasets version --path mammography/data/uint8_crops/ --message "all-patients" -r=skip
