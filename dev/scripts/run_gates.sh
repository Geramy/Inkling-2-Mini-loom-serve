#!/usr/bin/env bash
# usage: run_gates.sh <r9700|strix> <stage...>   (runs on the GPU box; stages: prep serve gates extra stop all)
# Skills correctness gates (LEVEL=dev) against the published package route: loom-fetch materialized copy, loom-serve --model=DIR --preset=P.
# The skills tree is read-only: a mirror W/skills holds a copy of loom-benchmark/scripts whose agent_golden.py sends "name" in tool messages.
# Wrap GPU stages in: flock -s ~/hrx-pr1/gpu.lock
set -euo pipefail
G=${1:?r9700|strix}; shift
R=${INKR:-$HOME/inkling/inkR}; W=$R/W
SK=${SKILLS:-$HOME/inkling/loom-skills/skills}
VENV=${LOOM_VENV:-$HOME/inkling/tools/M30-venv}
case $G in
  r9700) ARCH=gfx1201; PRESET=chat-r9700; PORT=18190; URI=amdgpu://GPU-e0880c05003c324d ;;
  strix) ARCH=gfx1151; PRESET=chat;       PORT=18191; URI=amdgpu://id/000000e2000000808e270f2b0bbb4e00 ;;
  *) echo "gpu must be r9700 or strix"; exit 2 ;;
esac
M=$W/mat-$G; PKG=$W/pkg-$G; B=$W/bench-$G; URL=http://127.0.0.1:$PORT
GGUF=$W/gguf/Inkling-2-Mini-UD-Q4_K_XL.gguf
export LOOM_ROOT=${LOOM_ROOT:-$HOME/loom030} K=$W/skills/loom-benchmark/scripts T=$HOME/loom/.tools/bin/python LOOM_KIT=$R/kit
export LLAMACPP_DIR=$HOME/inkling/inkB/build/bin LLAMA_URL=${LLAMA_URL:-http://127.0.0.1:18400} LEVEL=dev LOOM_PRESET=$PRESET PYTHONDONTWRITEBYTECODE=1

prep() {
  mkdir -p "$W/snap/Mainline_Support" "$W/gguf" "$B"
  for q in Q4 Q8; do
    n=Inkling-2-Mini-UD-${q}_K_XL.gguf
    ln -sfn "$HOME/models/inkling2-mini/Mainline_Support/$n" "$W/gguf/$n"
    ln -sfn "$HOME/models/inkling2-mini/Mainline_Support/$n" "$W/snap/Mainline_Support/${n}_file"
  done
  if [ ! -f "$M/model.json" ]; then
    "$VENV/bin/loom-fetch" "$R/publish/unsloth-Inkling-2-Mini-loom/model.json" --preset "$PRESET" --materialize=share --output "$M" --origin checkpoint="$W/snap" --no-input
  fi
  mkdir -p "$PKG"; ln -sfn "$M/model.json" "$PKG/model.json"; ln -sfn "$M/program" "$PKG/program"; ln -sfn "$M/tokenizer.json" "$PKG/tokenizer.json"
  python3 - "$HOME/inkling/inkO/W/pkg_q4/llm_config.json" "$PKG/llm_config.json" "$GGUF" <<'PY'
import json, sys
c = json.load(open(sys.argv[1])); c["gguf"] = sys.argv[3]; c["model_id"] = "thinkingmachines/Inkling-2-Mini"
json.dump(c, open(sys.argv[2], "w"), indent=1)
PY
  for f in llama_cache golden.json golden2.json golden_dev.json llamacpp_args.txt llamacpp_pick.json; do
    [ -e "$B/$f" ] || cp -r "${SEED:-$HOME/inkling/inkO/W/bench}/$f" "$B/$f"
  done
  rm -rf "$W/skills"; mkdir -p "$W/skills/loom-benchmark"
  for d in loom-kb loom-port-llm loom-roofline loom-setup; do ln -sfn "$SK/$d" "$W/skills/$d"; done
  for d in data reference; do ln -sfn "$SK/loom-benchmark/$d" "$W/skills/loom-benchmark/$d"; done
  cp -r "$SK/loom-benchmark/scripts" "$W/skills/loom-benchmark/scripts"
  sed -i 's/msgs.append({"role": "tool", "tool_call_id"/msgs.append({"role": "tool", "name": c["function"]["name"], "tool_call_id"/' "$W/skills/loom-benchmark/scripts/agent_golden.py"
  grep -q '"name": c\["function"\]\["name"\], "tool_call_id"' "$W/skills/loom-benchmark/scripts/agent_golden.py"
  cp "$SK/loom-benchmark/scripts/agent_golden.py" "$K/agent_golden_unmodified.py"
}

serve() {
  mkdir -p "$W/run-$G"; . "$VENV/bin/activate"
  (cd "$W/run-$G"; exec nohup loom-serve --device="$URI" --model="$M" --preset="$PRESET" --pool_backing=fixed --listen=127.0.0.1:$PORT --workers=4 ${SERVE_EXTRA:-} > serve.log 2>&1 < /dev/null) &
  echo $! > "$W/run-$G/pid"
  for _ in $(seq 600); do
    kill -0 "$(cat "$W/run-$G/pid")" 2>/dev/null || { echo "loom-serve died"; tail -5 "$W/run-$G/serve.log"; return 1; }
    curl -s -m 2 "$URL/v1/status" 2>/dev/null | grep -q '"registration": *"ready"' && break; sleep 1
  done
  mkdir -p "$HOME/.cache/loom-serve"; cat "$W/run-$G/pid" > "$HOME/.cache/loom-serve/gates-$PORT.pid"; echo $PORT > "$HOME/.cache/loom-serve/gates-$PORT.port"
  "$T" "$K/wait_model.py" "$URL" --wait 600
}
stop() {
  local p; p=$(cat "$W/run-$G/pid" 2>/dev/null || true)
  if [ -n "$p" ] && tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | grep -q loom-serve; then
    kill "$p"; for _ in $(seq 60); do kill -0 "$p" 2>/dev/null || break; sleep 1; done
  fi
  rm -f "$HOME/.cache/loom-serve/gates-$PORT".{pid,port} "$W/run-$G/pid"
}

gates() { cd "$B"; bash "$K/gates.sh" "$PKG" "$URL" "$GGUF" "$B" 2>&1 | tail -25 | tee "$B/gates_summary.log" || true; }

extra() {
  cd "$B"
  "$T" "$K/compare.py" "$PKG" golden.json --alt golden2.json --llama "$LLAMA_URL" --url "$URL" --out compare.json > compare.log 2>&1 || true
  tail -1 compare.log
  "$T" "$K/agent_golden_unmodified.py" "$PKG" "$URL" --out agent_unmodified.json > agent_unmodified.log 2>&1 || true
  tail -2 agent_unmodified.log
  "$T" "$K/agent_golden.py" "$PKG" "$URL" --llama "$LLAMA_URL" --thinking off --out agent_effort_none.json > agent_effort_none.log 2>&1 || true
  tail -2 agent_effort_none.log
  "$T" "$K/sampling_check.py" "$PKG" "$URL" --llama "$LLAMA_URL" --draws 200 > sampling_draws200.log 2>&1 || true
  tail -3 sampling_draws200.log
  "$T" "$R/scripts/sampling_supplement.py" "$PKG" "$URL" > sampling_supplement.log 2>&1 || true
  tail -5 sampling_supplement.log
  "$T" "$K/ctx_admit.py" "$PKG" "$URL" --out context_admit.json > context_admit.log 2>&1 || true
  tail -2 context_admit.log
  "$T" "$K/ctx_admit.py" "$PKG" "$URL" --complete 1800 --out context_admit_complete.json > context_admit_complete.log 2>&1 || true
  tail -2 context_admit_complete.log
}

for s in "$@"; do
  case $s in
    prep) prep ;; serve) serve ;; gates) gates ;; extra) extra ;; stop) stop ;;
    all) prep; serve; trap stop EXIT; gates || true; extra ;;
    *) echo "unknown stage $s"; exit 2 ;;
  esac
done
