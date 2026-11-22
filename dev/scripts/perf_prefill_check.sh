#!/usr/bin/env bash
# usage: perf_prefill_check.sh <r9700|strix>   (GPU box; under flock -x ~/hrx-pr1/gpu.lock) Prefill rate at about 130, 520, 1258 and 2454 tokens (spin URI), one serve session; rows to bench-<gpu>/extra_log.jsonl.
set -u
G=${1:?}; U=${INKU:-$HOME/inkling/inkU}; B=$U/bench-$G; M=$U/mat-$G
case $G in
  r9700) PRESET=chat-r9700; PORT=18190; DEV=amdgpu://GPU-e0880c05003c324d ;;
  strix) PRESET=chat; PORT=18191; DEV=amdgpu://id/000000e2000000808e270f2b0bbb4e00 ;;
esac
export LOOM_ROOT=$HOME/loom PYTHONDONTWRITEBYTECODE=1
. $HOME/loom/.venv/bin/activate; mkdir -p $B/raw/prefill-check; cd $B
pgrep -af 'loom-serve|llama-server' | grep -v pgrep | grep -q . && { echo "another server runs"; exit 3; }
( cd raw/prefill-check; exec nohup loom-serve --device="$DEV?completion_spin_ns=4294967295" --model=$M --preset=$PRESET ${POOLARGS:---pool_backing=fixed} --listen=127.0.0.1:$PORT --workers=8 > serve.log 2>&1 < /dev/null ) &
PID=$!; trap 'kill $PID 2>/dev/null; wait $PID 2>/dev/null' EXIT
for _ in $(seq 1200); do curl -s -m 2 http://127.0.0.1:$PORT/v1/status 2>/dev/null | grep -q '"registration": *"ready"' && break; sleep 0.5; done
$HOME/loom/.tools/bin/python $U/scripts/perf_prefill_n.py http://127.0.0.1:$PORT thinkingmachines/Inkling-2-Mini 480 1940 4688 9100 | tee raw/prefill-check/out-$G${POOLTAG:-}.log
