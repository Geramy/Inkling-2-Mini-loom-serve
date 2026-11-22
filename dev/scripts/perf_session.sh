#!/usr/bin/env bash
# usage: perf_session.sh <r9700|strix> <n> <spin|bare|blocking> <full|quick>   (GPU box; run under: flock -x ~/hrx-pr1/gpu.lock)
# One serve session of the published preset: start, first request (cold start), skills perf.py (short, --long), c4_steady 4, decode at 8k, stop by PID.
# spin = skills' active-wait URI (the baseline); bare = URI without a spin value (SDK default); blocking = --wait_mode=blocking (A/B arm). Only spin goes to perf_log.jsonl and the campaign view.
set -u
G=${1:?r9700|strix}; N=${2:?n}; MODE=${3:?spin|bare|blocking}; SCOPE=${4:-full}
U=${INKU:-$HOME/inkling/inkU}; B=$U/bench-$G; M=${MAT:-$U/mat-$G}
K=$HOME/inkling/loom-skills/skills/loom-benchmark/scripts; T=$HOME/loom/.tools/bin/python; VENV=$HOME/loom/.venv
export LOOM_ROOT=$HOME/loom PYTHONDONTWRITEBYTECODE=1 LOOM_KIT=${LOOM_KIT:-$HOME/inkling/inkR/kit}
case $G in
  r9700) PRESET=chat-r9700; PORT=18190; DEV=amdgpu://GPU-e0880c05003c324d ;;
  strix) PRESET=chat; PORT=18191; DEV=amdgpu://id/000000e2000000808e270f2b0bbb4e00 ;;
  *) echo "gpu"; exit 2 ;;
esac
MODEL=thinkingmachines/Inkling-2-Mini; URL=http://127.0.0.1:$PORT
EXTRA=(); LOG=perf_log.jsonl; C4LOG=c4_log.jsonl; LABEL=loom-baseline; PC=(); C4L=(--label loom-baseline)
case $MODE in
  spin) DEV="$DEV?completion_spin_ns=4294967295" ;;
  bare) ;;
  blocking) EXTRA=(--wait_mode=blocking) ;;
  *) echo "mode"; exit 2 ;;
esac
if [ "$MODE" != spin ] || [ -n "${TAG:-}" ]; then TAG=${TAG:-$MODE}; LOG=perf_log_$TAG.jsonl; C4LOG=c4_log_$TAG.jsonl; LABEL=loom-baseline-$TAG; PC=(--no-campaign); C4L=(); fi
R=$B/raw/${TAG:-$MODE}-$N; mkdir -p "$R"; cd "$B" || exit 1
if pgrep -af 'loom-serve|llama-server|llama-bench' | grep -v pgrep | grep -q .; then echo "another server runs"; pgrep -af 'loom-serve|llama-server' | cut -c1-150; exit 3; fi
{ date -u +%FT%TZ; uptime; for c in /sys/class/drm/card[0-9]/device; do echo "$c busy $(cat $c/gpu_busy_percent 2>/dev/null)"; done; } > "$R/preflight.txt"
. "$VENV/bin/activate"
t0=$(date +%s.%N)
( cd "$R"; exec nohup loom-serve --device="$DEV" --model="$M" --preset="$PRESET" --pool_backing=fixed --listen=127.0.0.1:$PORT --workers=8 "${EXTRA[@]}" > serve.log 2>&1 < /dev/null ) &
PID=$!
stopit() { if tr '\0' ' ' < /proc/$PID/cmdline 2>/dev/null | grep -q loom-serve; then kill $PID; for _ in $(seq 60); do kill -0 $PID 2>/dev/null || break; sleep 1; done; fi; }
trap stopit EXIT
for _ in $(seq 1200); do
  kill -0 $PID 2>/dev/null || { echo "loom-serve died"; tail -5 "$R/serve.log"; exit 1; }
  curl -s -m 2 $URL/v1/status 2>/dev/null | grep -q '"registration": *"ready"' && break; sleep 0.5
done
t1=$(date +%s.%N)
req() { curl -s -m 900 -o /dev/null -w '%{time_total}' $URL/v1/chat/completions -H 'Content-Type: application/json' -d "{\"model\":\"$MODEL\",\"max_tokens\":8,\"temperature\":0,\"chat_template_kwargs\":{\"enable_thinking\":false},\"messages\":[{\"role\":\"user\",\"content\":\"[$1] Say hi.\"}]}"; }
f1=$(req first$N$RANDOM); f2=$(req second$N$RANDOM)
python3 - "$B/coldstart.jsonl" "$MODE" "$N" "$(echo "$t1 - $t0" | bc)" "$f1" "$f2" "$(tr '\n' ' ' < $R/preflight.txt)" <<'PY'
import json, sys, time
p, mode, n, ready, f1, f2, pre = sys.argv[1:]
open(p, "a").write(json.dumps({"time": time.strftime("%F %T"), "mode": mode, "session": int(n), "start_to_ready_s": round(float(ready), 2), "first_request_s": float(f1), "second_request_s": float(f2), "preflight": pre.strip()}) + "\n")
PY
echo "ready $(echo "$t1 - $t0" | bc) s; first $f1 s second $f2 s"
$T $K/perf.py $URL $LABEL --model $MODEL --log $LOG "${PC[@]}" > "$R/short.log" 2>&1; tail -n1 "$R/short.log" | cut -c1-300
if [ "$SCOPE" = full ]; then
  $T $K/perf.py $URL $LABEL --model $MODEL --long --log $LOG "${PC[@]}" > "$R/long.log" 2>&1; tail -n1 "$R/long.log" | cut -c1-300
fi
$T $K/c4_steady.py $URL 4 --model $MODEL "${C4L[@]}" > "$R/c4.log" 2>&1; tail -n1 "$R/c4.log" | cut -c1-300
grep '^{' "$R/c4.log" | tail -1 >> $C4LOG
if [ "$SCOPE" = full ]; then
  $T $U/scripts/perf_decode_ctx.py $URL $MODEL 8000 --label $LABEL --log extra_log.jsonl > "$R/d8k.log" 2>&1; tail -n1 "$R/d8k.log" | cut -c1-300
fi
echo "{\"session_end\": \"$MODE-$N\"}" | tee -a $LOG >> $C4LOG
curl -s $URL/v1/status | python3 -c 'import json,sys; d=json.load(sys.stdin); print(json.dumps(d["server"]["resources"]["devices"][0])[:300])' > "$R/status_end.txt" 2>&1
stopit; trap - EXIT
echo "stopped $PID alive=$(kill -0 $PID 2>/dev/null && echo yes || echo no)"
