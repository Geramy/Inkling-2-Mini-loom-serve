#!/usr/bin/env bash
# usage: bench_card.sh <arch> <q4|q8> <port> [bench_chat.py flags]   (on the box) cold start time (process start to ready, first request), then the prefill and decode benchmark under the exclusive GPU lock, with the memory use of both cards
set -uo pipefail
ARCH=$1; Q=$2; PORT=$3; shift 3
cd ~/inkling/inkK
R=run-card-$ARCH-$Q
if [ -f $R/pid ]; then P=$(cat $R/pid); kill $P 2>/dev/null; for i in 1 2 3 4 5 6 7 8; do kill -0 $P 2>/dev/null || break; sleep 1; done; kill -9 $P 2>/dev/null; fi
sleep 3
while curl -s -m 1 localhost:$PORT/v1/status > /dev/null; do sleep 1; done
mem() { for c in /sys/class/drm/card[0-9]/device; do echo "  $(basename $(dirname $c)) vram $(( $(cat $c/mem_info_vram_used) / 1048576 )) of $(( $(cat $c/mem_info_vram_total) / 1048576 )) MiB, gtt $(( $(cat $c/mem_info_gtt_used) / 1048576 )) MiB"; done; }
echo "memory before:"; mem
T0=$(date +%s.%N)
nohup ./scripts/serve_card.sh $ARCH $Q $PORT > serve-card-$ARCH-$Q.log 2>&1 &
for i in $(seq 1 600); do s=$(curl -s -m 5 localhost:$PORT/v1/status | grep -o '"registration":"[a-z_]*"'); case "$s" in *ready*|*failed*) break;; esac; sleep 0.5; done
T1=$(date +%s.%N)
echo "cold start to ready ($s): $(echo "$T1 - $T0" | bc) s"
. ~/loom-serve/.venv-inkK/bin/activate
T2=$(date +%s.%N)
flock -x ~/hrx-pr1/gpu.lock curl -s -m 600 localhost:$PORT/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"inkling-2-mini","max_tokens":1,"temperature":0,"messages":[{"role":"user","content":"Hello"}]}' > /dev/null
T3=$(date +%s.%N)
echo "first request (weights activation, first kernel launches): $(echo "$T3 - $T2" | bc) s"
echo "memory after the first request:"; mem
flock -x ~/hrx-pr1/gpu.lock timeout 3000 python scripts/bench_chat.py $PORT "$@"
echo "memory after the benchmark:"; mem
