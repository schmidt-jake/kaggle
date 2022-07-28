#!/bin/bash

# Downloads the competition dataset and stores it at ./mayo_clinic_strip_ai/data
# WARNING: this script requires >350GB of free disk space

set -euo pipefail

LOCAL_DIR=mayo_clinic_strip_ai/data
COMPETITON_NAME=mayo-clinic-strip-ai
ZIP_NAME=$LOCAL_DIR/$COMPETITON_NAME.zip

# download the dataset as a zip file
kaggle competitions download --path=$LOCAL_DIR $COMPETITON_NAME

# unzip the dataset
unzip $ZIP_NAME -d $LOCAL_DIR

# delete the zip file
rm -r $ZIP_NAME
