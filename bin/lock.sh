#!/bin/bash

set -euo pipefail

pip install -U pip setuptools wheel

pip install pip-compile-multi

pip-compile-multi --autoresolve --live --upgrade --directory requirements
