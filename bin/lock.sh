#!/bin/bash

set -euo pipefail

pip install -U pip setuptools wheel

pip install pip-compile-multi

UPGRADE_FLAG=${1:-"--no-upgrade"}

pip-compile-multi \
    --autoresolve \
    --live \
    $UPGRADE_FLAG \
    --use-cache \
    --directory=requirements
