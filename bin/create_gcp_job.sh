#!/bin/bash

set -euo pipefail

# https://cloud.google.com/vertex-ai/docs/training/create-custom-job#create_custom_job-gcloud
gcloud ai custom-jobs create \
  --region=us-central1 \
  --display-name=training \
  --config=bin/job-config.yaml
