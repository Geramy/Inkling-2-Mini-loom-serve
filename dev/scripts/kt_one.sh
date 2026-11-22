#!/usr/bin/env bash
# usage: kt_one.sh <gfx1201|gfx1151> <test file> [args]   (box; runs one dense test from the inkS kt area under flock -s)
set -euo pipefail
A=$1; F=$2; shift 2
cd ~/inkling/inkS/kt/test/dense
export PATH=$HOME/loom-serve/.venv-inkD/bin:$PATH VIRTUAL_ENV=$HOME/loom-serve/.venv-inkD PYTHONDONTWRITEBYTECODE=1
flock -s ~/hrx-pr1/gpu.lock python "$F" "$A" "$@"
