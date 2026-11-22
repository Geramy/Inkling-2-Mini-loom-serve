#!/usr/bin/env bash
# usage: jsplit.sh <arch> <port> [Q4|Q8] "<stop counts>"   (local) decode step time (prompt A in a session, single token appends) for INKLING_STOP = each count; differences give the time of each hook over the 48 visits
cd "$(dirname "$0")/.."
ARCH=$1; PORT=$2; Q=${3:-Q4}; STOPS=${4:-"0 1 3 5 7 8 9 10 11 12 13 14 99"}
for n in $STOPS; do
  INKLING_STOP=$n INKLING_CONTEXT=${CTX:-4096} INKLING_PREFILL=16 bash scripts/jrun.sh "$ARCH" "$PORT" 0,0,0,64 $Q 600 2>&1 | head -1 | cut -c1-40 | tr '\n' ' '
  echo -n "stop $n: "
  ssh user@gpu-box ". ~/loom-serve/.venv-inkK/bin/activate; cd ~/inkling/inkK; timeout 300 flock -x ~/hrx-pr1/gpu.lock python scripts/jdecode.py $PORT ~/inkling/inkB/oracle/dumps/Q8_K_XL_A ${STEPS:-30} ${CTXARG:-}"
done
