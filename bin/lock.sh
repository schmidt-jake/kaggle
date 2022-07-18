#!/bin/bash

set -euo pipefail

pip-compile-multi --autoresolve --live --upgrade --directory requirements
