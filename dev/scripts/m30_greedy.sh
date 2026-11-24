#!/usr/bin/env bash
# usage: m30_greedy.sh <r9700|strix> <model dir> <out.json> [venv dir]   (box; run under flock -s) One serve session, oracle prompts A, B, C (and long documents when GREEDY_LONG=1) greedy 64 tokens, stop by PID.
set -u
G=${1:?gpu}; M=${2:?dir}; OUT=${3:?out}; VD=${4:-$HOME/inkling/tools/M30-venv}
V=${GREEDY_LOGS:-$HOME/inkling/integ4}; HERE=$(cd "$(dirname "$0")" && pwd)
case $G in
  r9700) PRESET=chat-r9700; PORT=18190; DEV='amdgpu://GPU-e0880c05003c324d?completion_spin_ns=4294967295' ;;
  strix) PRESET=chat; PORT=18191; DEV='amdgpu://id/000000e2000000808e270f2b0bbb4e00?completion_spin_ns=4294967295' ;;
esac
URL=http://127.0.0.1:$PORT; T=$HOME/loom/.tools/bin/python
export LOOM_ROOT=${ABROOT:-$HOME/loom030} PYTHONDONTWRITEBYTECODE=1 GREEDY_LONG=${GREEDY_LONG:-0}
if pgrep -x 'loom-serve|llama-server|llama-bench' | grep -q .; then echo "another server runs"; exit 3; fi
. "$VD/bin/activate"; mkdir -p "$V/ablogs"
TAG=$(basename "$OUT" .json)
( exec nohup loom-serve --device="$DEV" --model="$M" --preset="$PRESET" --pool_backing=fixed --listen=127.0.0.1:$PORT --workers=8 > "$V/ablogs/$TAG.serve.log" 2>&1 < /dev/null ) &
PID=$!
stopit() { if tr '\0' ' ' < /proc/$PID/cmdline 2>/dev/null | grep -q loom-serve; then kill -TERM $PID; for _ in $(seq 120); do kill -0 $PID 2>/dev/null || break; sleep 1; done; fi; }
trap stopit EXIT
for _ in $(seq 1800); do
  kill -0 $PID 2>/dev/null || { echo "loom-serve died"; tail -8 "$V/ablogs/$TAG.serve.log"; exit 1; }
  S=$(curl -s -m 2 $URL/v1/status 2>/dev/null); echo "$S" | grep -q '"registration": *"ready"' && break; echo "$S" | grep -q '"registration": *"failed"' && break; sleep 0.5
done
if curl -s -m 2 $URL/v1/status | grep -q '"registration": *"failed"'; then echo "registration failed"; curl -s $URL/v1/status | head -c 1500; exit 4; fi
$T -I "$HERE/greedy_check.py" $URL thinkingmachines/Inkling-2-Mini "$OUT"
stopit; trap - EXIT
