#!/usr/bin/env bash
# usage: make_card.sh TREE   Writes info.json, README.md and REPORT.md into TREE with the loom-skills card tools, then validates them.
# Steps: (1) llm_config.json with the chat block (only when CFG is set), (2) card tools on a copy of TREE (every .loom file of the tree is named by model.json, so the programs hash of the tree equals the hash of the fetched files), (3) schema, render check, scrub check, loom-fetch dry run, quick start check.
# Env: SKILLS (loom-skills checkout), T (python of the skills tools), BENCH (dir with <gpu>/ev_*.json), SRC (this repo), CFG (inspect_gguf.py cfg.json),
#      GATES_TIME (ISO time of the gate run, default now), WORK (scratch dir on the same disk as CKPT: the fetch uses hard links, never copies the weights), MEMORY_BYTES, CKPT (dir with Mainline_Support/*.gguf_file links, for the loom-fetch check).
set -euo pipefail
TREE=$(cd "${1:?tree}" && pwd)
SKILLS=${SKILLS:-$HOME/inkling/loom-skills}
T=${T:-$HOME/loom/.tools/bin/python}
SRC=${SRC:-$(cd "$(dirname "$0")/.." && pwd)}
BENCH=${BENCH:-$SRC/bench}
export LOOM_ROOT=${LOOM_ROOT:-$HOME/loom}
# Largest memory in use after the first request at the chat preset (131072 context, 4 rows, prefill 2048): 8060S gtt in docs/CONTEXT_TABLE.md (R9700: 24862375936).
MEMORY_BYTES=${MEMORY_BYTES:-24925405184}
C=$SKILLS/skills/loom-benchmark/scripts/card

[ -z "${CFG:-}" ] || $T -I $SRC/scripts/make_llm_config.py --publish "$TREE" --cfg "$CFG" --skills "$SKILLS" --chat "$SRC/kit/chat_block.json"

WORK=$(cd "${WORK:-$PWD}" && pwd)
STAGE=$(mktemp -d -p "$WORK")
trap 'rm -rf "$STAGE"' EXIT
cp -r "$TREE"/. "$STAGE"/
EV=()
EVD="$WORK/ev_$$"; mkdir "$EVD"; trap 'rm -rf "$STAGE" "$EVD"' EXIT
for g in strix-halo radeon-r9700; do
  $T -I - "$BENCH/$g/gates.json" "$EVD/gates_$g.json" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
notes = {"sampling": "default temperature 1.0, check expects greedy", "agent": "tool messages sent with name"}
for n, t in notes.items():
    d["gates"][n]["detail"] = t
json.dump(d, open(sys.argv[2], "w"))
PY
  $T -I $C/make_info.py gates --source-kind machine --gpu $g --package "$STAGE" --time "${GATES_TIME:-$(date -u +%Y-%m-%dT%H:%M:%S)}" "$EVD/gates_$g.json" "$BENCH/$g/kernel_tests/kernel_tests.log" -o "$EVD/ev_gates_$g.json"
  EV+=("$EVD/ev_gates_$g.json" "$BENCH/$g/ev_perf.json" "$BENCH/$g/ev_roofline_ud-q4-k-xl.json")
done
rm -f "$STAGE/info.json" "$STAGE/README.md" "$STAGE/REPORT.md"
$T -I $C/make_info.py build "$STAGE" --repo local/unsloth-Inkling-2-Mini-loom --memory-bytes "$MEMORY_BYTES" --output unsloth-Inkling-2-Mini-loom \
  --report generated --gpus strix-halo,radeon-r9700 --device-results none --evidence "${EV[@]}" -o "$STAGE/info.json"
$T -I $C/render_card.py "$STAGE"
$T -I $C/render_card.py --check "$STAGE" "$STAGE/README.md"
cp "$STAGE/info.json" "$STAGE/README.md" "$STAGE/REPORT.md" "$TREE"/

# validation of the files now in TREE
$T -I - "$TREE" "$C" <<'PY'
import json, os, sys
d, c = sys.argv[1:3]
sys.path.insert(0, c)
import card_lib
info = json.load(open(os.path.join(d, "info.json")))
errs = card_lib.validate(info, card_lib.programs_sha256(d))
print("schema:", info["schema"], "errors:", len(errs))
for e in errs: print(" ", e)
sys.exit(1 if errs else 0)
PY
bash "$SRC/scripts/scrub_check.sh" "$TREE"
if [ -n "${CKPT:-}" ]; then
  O=$(mktemp -d -p "$WORK"); trap 'rm -rf "$STAGE" "$EVD" "$O"' EXIT
  for p in chat chat-r9700 chat-q8 chat-q8-r9700; do
    $LOOM_ROOT/.venv/bin/loom-fetch "$TREE/model.json" --preset $p --materialize=share --output "$O/$p" --origin checkpoint="$CKPT" --no-input >/dev/null && echo "loom-fetch $p: ok"
  done

fi
