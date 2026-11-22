#!/usr/bin/env bash
# usage: profile_run.sh <r9700|strix> <session> [out dir]   (GPU box; run under: flock -x ~/hrx-pr1/gpu.lock)
# One profiled serve session: loom-serve with HAL dispatch statistics, the requests of profile_driver.py for <session>, a graceful stop by PID.
# The kernel table is printed at shutdown into serve.log. Weights come from the hard-linked fetch dir MAT (default inkV/mat-<gpu>).
set -u
G=${1:?r9700|strix}; S=${2:?session}; V=${INKV:-$HOME/inkling/inkV}; O=${3:-$V/prof/$G}; M=${MAT:-$V/mat-$G}
HERE=$(cd "$(dirname "$0")" && pwd)
case $G in
  r9700) PRESET=chat-r9700; PORT=18190; DEV=amdgpu://GPU-e0880c05003c324d ;;
  strix) PRESET=chat; PORT=18191; DEV=amdgpu://id/000000e2000000808e270f2b0bbb4e00 ;;
  *) echo gpu; exit 2 ;;
esac
DEV="$DEV?completion_spin_ns=4294967295"
URL=http://127.0.0.1:$PORT; T=$HOME/loom/.tools/bin/python
export LOOM_ROOT=$HOME/loom PYTHONDONTWRITEBYTECODE=1
mkdir -p "$O"
if pgrep -af 'loom-serve|llama-server|llama-bench' | grep -v pgrep | grep -q .; then echo "another server runs"; exit 3; fi
{ date -u +%FT%TZ; uptime; for c in /sys/class/drm/card[0-9]/device; do echo "$c busy $(cat $c/gpu_busy_percent 2>/dev/null)"; done; } > "$O/$S.pre.txt"
. "$HOME/loom/.venv/bin/activate"
( cd "$O"; exec nohup loom-serve --device="$DEV" --model="$M" --preset="$PRESET" --pool_backing=fixed --listen=127.0.0.1:$PORT --workers=8 \
    --device_profiling_mode=dispatch-events,executable-metadata --device_profiling_flush_interval_ms=5 --print_device_statistics \
    > "$O/$S.serve.log" 2>&1 < /dev/null ) &
PID=$!
stopit() { if tr '\0' ' ' < /proc/$PID/cmdline 2>/dev/null | grep -q loom-serve; then kill -TERM $PID; for _ in $(seq 120); do kill -0 $PID 2>/dev/null || break; sleep 1; done; fi; }
trap stopit EXIT
for _ in $(seq 1200); do
  kill -0 $PID 2>/dev/null || { echo "loom-serve died"; tail -5 "$O/$S.serve.log"; exit 1; }
  curl -s -m 2 $URL/v1/status 2>/dev/null | grep -q '"registration": *"ready"' && break; sleep 0.5
done
$T -I "$HERE/profile_driver.py" $URL thinkingmachines/Inkling-2-Mini "$S" > "$O/$S.req.json" 2> "$O/$S.driver.err"; rc=$?
stopit; trap - EXIT
echo "$G $S driver rc=$rc stopped alive=$(kill -0 $PID 2>/dev/null && echo yes || echo no) dropped=$(grep -c 'dropped_records=[1-9]' "$O/$S.serve.log")"
