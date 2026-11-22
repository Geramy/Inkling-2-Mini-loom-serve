#!/usr/bin/env bash
# usage: jvisits.sh <arch> <port> <prompt A|B|C> <prefill> "<visits>" [capture_all.py flags]   (local) restarts the server per visit and prints the NMSE table of each visit
cd "$(dirname "$0")/.."
ARCH=$1; PORT=$2; P=$3; PRE=$4; VISITS=$5; shift 5
for v in $VISITS; do
  INKLING_PREFILL=$PRE bash scripts/jrun.sh "$ARCH" "$PORT" all,$v Q8 ${WAIT:-600} 2>&1 | cut -c1-160 | head -1
  TMO=${TMO:-300} bash scripts/jcheck.sh "$PORT" "$P" "$v" "$@" | cut -c1-110
done
