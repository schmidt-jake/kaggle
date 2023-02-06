#!/bin/bash

set -euo pipefail

IMAGE=us-central1-docker.pkg.dev/rnsa-breast-cancer-detection/containers/mammography:0.1

docker buildx build -t $IMAGE -f Dockerfile .
docker push $IMAGE
