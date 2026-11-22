#!/usr/bin/env bash
# usage: restart.sh <arch> [port] [wait_seconds]   (local; syncs the package, restarts the server on the box, prints the registration state)
# env passes through to serve.sh: INKLING_MODE, INKLING_CAPTURE, INKLING_GGUF, INKLING_CONTEXT, INKLING_PREFILL, INKLING_ROWS
set -euo pipefail
cd "$(dirname "$0")/.."
ARCH=${1:?arch}; PORT=${2:-18090}; WAIT=${3:-40}
BOX=${BOX:-user@gpu-box}
DEST=${INKLING_BOX_DIR:-inkling/inkH}
MODE=${INKLING_MODE:-sequence}
bash scripts/sync.sh 2>&1 | grep -v "in the future" || true
ENVS="INKLING_ROOT=\$HOME/$DEST LOOM_VENV=\$HOME/loom-serve/.venv-inkK INKLING_MODE=$MODE INKLING_CAPTURE=${INKLING_CAPTURE:-0,0,0,2304} INKLING_GGUF=${INKLING_GGUF:-} INKLING_CONTEXT=${INKLING_CONTEXT:-4096} INKLING_PREFILL=${INKLING_PREFILL:-512} INKLING_ROWS=${INKLING_ROWS:-2} INKLING_CHECKPOINTS=${INKLING_CHECKPOINTS:-0} INKLING_PROBES=${INKLING_PROBES:-0} INKLING_TUNE_PROBE=${INKLING_TUNE_PROBE:-0} INKLING_STOP=${INKLING_STOP:-99} INKLING_POOL_TOKENS=${INKLING_POOL_TOKENS:-}"
ssh "$BOX" "cd ~/$DEST; if [ -f run-$MODE-$ARCH/pid ]; then P=\$(cat run-$MODE-$ARCH/pid); kill \$P 2>/dev/null; for i in 1 2 3 4 5 6 7 8; do kill -0 \$P 2>/dev/null || break; sleep 1; done; kill -9 \$P 2>/dev/null; fi; sleep 1; env $ENVS nohup ./scripts/serve.sh $ARCH $PORT > serve-$MODE-$ARCH.log 2>&1 & sleep $WAIT; tail -c 1500 ~/$DEST/serve-$MODE-$ARCH.log; echo; curl -s localhost:$PORT/v1/status | head -c 1800"
