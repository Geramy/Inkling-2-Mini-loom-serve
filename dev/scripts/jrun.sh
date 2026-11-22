#!/usr/bin/env bash
# usage: jrun.sh <arch> <port> <capture: id,visit,occ,width | all,<visit>> [Q8|Q4] [max wait s]   (local) syncs, restarts the inkK server, waits until it is ready or failed, prints diagnostics. env: INKLING_STOP, INKLING_PREFILL, INKLING_ROWS, INKLING_CONTEXT
cd "$(dirname "$0")/.."
export INKLING_BOX_DIR=inkling/inkK
M=/home/user/models/inkling2-mini/Mainline_Support
case "${4:-Q8}" in Q4) G=$M/Inkling-2-Mini-UD-Q4_K_XL.gguf ;; *) G=$M/Inkling-2-Mini-UD-Q8_K_XL.gguf ;; esac
CAP=${3:-0,0,0,2304}
case "$CAP" in all,*) TOTAL=$(grep -o '"total": [0-9]*' package/dispatch/capture_layout.json | grep -o '[0-9]*$'); CAP=99,${CAP#all,},0,$TOTAL ;; esac
INKLING_CAPTURE=$CAP INKLING_GGUF=$G bash scripts/restart.sh "$1" "$2" 3 > /dev/null 2>&1
ssh user@gpu-box "for i in \$(seq 1 \$(( ${5:-600} / 5 ))); do s=\$(curl -s -m 5 localhost:$2/v1/status | grep -o '\"registration\":\"[a-z_]*\"'); case \"\$s\" in *ready*|*failed*) break;; esac; sleep 5; done"
bash scripts/diag.sh "$2"
