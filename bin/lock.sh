#!/bin/bash

set -euo pipefail

UPGRADE_FLAG=${1:-"--no-upgrade"}

pip-compile-multi \
    --autoresolve \
    --live \
    $UPGRADE_FLAG \
    --use-cache \
    --no-backtracking \
    --directory=$2
