#!/usr/bin/env bash
# usage: perf_once.sh <r9700|strix> <model dir> <tag>   (GPU box; under flock -x) One serve session, skills perf.py (short decode, prefill 1k, 2k, decode 2k), no campaign view.
# One unprofiled serve session of a package variant (spin wait, as the baseline), ab_session.py metrics, stop by PID. Appends to $INKV/ab_log.jsonl.
set -u
G=${1:?gpu}; M=${2:?dir}; TAG=${3:?tag}; MET=${4:-x}; REPS=${5:-3}
V=${INKV:-$HOME/inkling/inkV}; HERE=$(cd "$(dirname "$0")" && pwd)
case $G in
  r9700) PRESET=chat-r9700; PORT=18190; DEV='amdgpu://GPU-e0880c05003c324d?completion_spin_ns=4294967295' ;;
  strix) PRESET=chat; PORT=18191; DEV='amdgpu://id/000000e2000000808e270f2b0bbb4e00?completion_spin_ns=4294967295' ;;
esac
URL=http://127.0.0.1:$PORT; T=$HOME/loom/.tools/bin/python
export LOOM_ROOT=$HOME/loom PYTHONDONTWRITEBYTECODE=1
if pgrep -af 'loom-serve|llama-server|llama-bench' | grep -v pgrep | grep -q .; then echo "another server runs"; exit 3; fi
. "$HOME/loom/.venv/bin/activate"; mkdir -p "$V/ablogs"
( exec nohup loom-serve --device="$DEV" --model="$M" --preset="$PRESET" --pool_backing=fixed --listen=127.0.0.1:$PORT --workers=8 > "$V/ablogs/$TAG.serve.log" 2>&1 < /dev/null ) &
PID=$!
stopit() { if tr '\0' ' ' < /proc/$PID/cmdline 2>/dev/null | grep -q loom-serve; then kill -TERM $PID; for _ in $(seq 120); do kill -0 $PID 2>/dev/null || break; sleep 1; done; fi; }
trap stopit EXIT
for _ in $(seq 1800); do
  kill -0 $PID 2>/dev/null || { echo "loom-serve died"; tail -5 "$V/ablogs/$TAG.serve.log"; exit 1; }
  curl -s -m 2 $URL/v1/status 2>/dev/null | grep -q '"registration": *"ready"' && break; sleep 0.5
done
(cd "$V" && $T $HOME/inkling/loom-skills/skills/loom-benchmark/scripts/perf.py $URL "$TAG" --model thinkingmachines/Inkling-2-Mini --log perf_check.jsonl --no-campaign > "$V/ablogs/$TAG.perf.out" 2>&1); tail -n1 "$V/ablogs/$TAG.perf.out" | cut -c1-400
stopit; trap - EXIT
