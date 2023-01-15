#!/bin/bash

set -euo pipefail

# --use-compress-program="pigz --processes 8 --best --verbose" \
# | pigz -v -p 8 --best > mammography/data/uint8_crops/png.tar.gz

tar \
    -C mammography/data/uint8_crops/png/ \
    --use-compress-program="pigz --processes 8 --best --verbose" \
    -cvf - . \
