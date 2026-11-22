#!/usr/bin/env bash
# usage: jtimed.sh <arch> <port> <stop> [prompt B]   (local) restarts the inkK server with INKLING_STOP=<stop> (prefill 512, no capture) and times the prompt under the exclusive GPU lock
cd "$(dirname "$0")/.."
INKLING_STOP=$3 INKLING_PREFILL=512 bash scripts/jrun.sh "$1" "$2" 0,0,0,64 Q8 600 2>&1 | head -1 | cut -c1-100
INKLING_BOX_DIR=inkling/inkK bash scripts/sync.sh 2>&1 | grep -v future
ssh user@gpu-box ". ~/loom-serve/.venv-inkK/bin/activate; cd ~/inkling/inkK; timeout 300 flock -x ~/hrx-pr1/gpu.lock python scripts/jtime.py $2 ~/inkling/inkB/oracle/dumps/Q8_K_XL_${4:-B} ${RUNS:-10}"
