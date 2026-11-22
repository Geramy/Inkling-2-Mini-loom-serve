#!/usr/bin/env bash
# usage: jchecklong.sh <port> <visit> [capture_all.py flags]   (local) like jcheck.sh for the 1332 token prompt (full dump on the box)
cd "$(dirname "$0")/.."
INKLING_BOX_DIR=inkling/inkK bash scripts/sync.sh 2>&1 | grep -v future
PORT=$1; V=$2; shift 2
ssh user@gpu-box ". ~/loom-serve/.venv-inkK/bin/activate; cd ~/inkling/inkK; timeout ${TMO:-110} flock -s ~/hrx-pr1/gpu.lock python scripts/capture_all.py $PORT ~/inkling/inkE/oracle_long/Q8_K_XL_L1 $V --gguf ~/models/inkling2-mini/Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf $*" 2>&1 | tail -${TAIL:-30}
