#!/bin/bash

set -euo pipefail

pip-compile-multi \
    --autoresolve \
    --live \
    --use-cache \
    --no-backtracking \
    $@
