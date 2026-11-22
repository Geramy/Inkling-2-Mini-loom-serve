#!/usr/bin/env bash
cd ~/inkling/inkU
run() { flock -x ~/hrx-pr1/gpu.lock bash scripts/perf_session.sh "$@" > s-$1-$3-$2.out 2>&1 || { echo "FAILED $*"; exit 1; }; echo "done $*"; }
run r9700 2 spin full; run r9700 1 bare quick; run r9700 3 spin full; run r9700 2 bare quick; run r9700 1 blocking quick; run r9700 3 bare quick
run strix 1 spin full; run strix 2 spin full; run strix 3 spin full; run strix 1 bare quick
echo ALLDONE
