#!/usr/bin/env bash
# usage: jit_time.sh <arch> <port> [capture]   (local) starts a sequence server on the box with the real capacities and reports the time to ready and the request floor
set -euo pipefail
cd "$(dirname "$0")/.."
ARCH=${1:?arch}; PORT=${2:?port}; CAP=${3:-2,0,0,2304}
BOX=${BOX:-user@gpu-box}; DEST=${INKLING_BOX_DIR:-inkling/inkH}
bash scripts/sync.sh 2>&1 | grep -v "in the future" || true
ssh "$BOX" "cd ~/$DEST; [ -f run-sequence-$ARCH/pid ] && kill \$(cat run-sequence-$ARCH/pid) 2>/dev/null; sleep 1
. ~/loom-serve/.venv-inkC/bin/activate
export INKLING_CAPTURE=$CAP INKLING_CONTEXT=${INKLING_CONTEXT:-32768} INKLING_PREFILL=${INKLING_PREFILL:-512} INKLING_ROWS=${INKLING_ROWS:-4} INKLING_CHECKPOINTS=${INKLING_CHECKPOINTS:-4} INKLING_GGUF=${INKLING_GGUF:-}
start=\$(date +%s.%N)
nohup ./scripts/serve.sh $ARCH $PORT > serve-sequence-$ARCH.log 2>&1 &
until curl -s localhost:$PORT/v1/status 2>/dev/null | grep -q '\"registration\":\"\(ready\|failed\)\"'; do sleep 0.2; done
ready=\$(date +%s.%N)
curl -s localhost:$PORT/v1/status | python3 -c 'import sys,json; l=json.load(sys.stdin)[\"data\"][0][\"loom\"]; c=l.get(\"compilation\",{}); print(l[\"registration\"], l.get(\"error\",{}).get(\"message\",\"\")[:300], \"prepare_ms\", l[\"preparation_duration_ns\"]/1e6, \"stage_sum_ms\", c.get(\"stage_duration_sum_ns\",0)/1e6, \"commands\", c.get(\"commands\"), \"kernels\", c.get(\"kernels\"))'
echo \"start to registration ready: \$(echo \"\$ready - \$start\" | bc) s\"
flock -s ~/hrx-pr1/gpu.lock python scripts/floor_bench.py $PORT 1 40
flock -s ~/hrx-pr1/gpu.lock python scripts/floor_bench.py $PORT 16 40
curl -s localhost:$PORT/v1/status | python3 -c 'import sys,json; l=json.load(sys.stdin)[\"data\"][0][\"loom\"]; print(json.dumps(l[\"parameters\"][\"primary\"]))'"
