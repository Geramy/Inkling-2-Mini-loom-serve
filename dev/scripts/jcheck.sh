#!/usr/bin/env bash
# usage: jcheck.sh <port> <prompt A|B|C> <visit> [capture_all.py flags]   (local) syncs scripts and runs capture_all.py on the box under a shared GPU lock
cd "$(dirname "$0")/.."
INKLING_BOX_DIR=inkling/inkK bash scripts/sync.sh 2>&1 | grep -v future
PORT=$1; P=$2; V=$3; shift 3
ssh user@gpu-box ". ~/loom-serve/.venv-inkK/bin/activate; cd ~/inkling/inkK; timeout ${TMO:-100} flock -s ~/hrx-pr1/gpu.lock python scripts/capture_all.py $PORT ~/inkling/inkB/oracle/dumps/Q8_K_XL_$P $V --gguf ~/models/inkling2-mini/Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf $*" 2>&1 | tail -${TAIL:-30}
