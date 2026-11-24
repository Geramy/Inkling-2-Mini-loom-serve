#!/usr/bin/env bash
# usage: mkgates.sh   (box) builds ~/inkling/optCHAT/gr: a gates root for run_gates.sh whose materialized models carry the CHATFIX package files and presets.
# The weights stay hard links of the shared GGUF. Run: INKR=~/inkling/optCHAT/gr LOOM_VENV=~/inkling/tools/M30-venv SKILLS=... bash gr/scripts/run_gates.sh r9700 prep serve gates extra stop
set -euo pipefail
O=$HOME/inkling/optCHAT; R=$O/gr
rm -rf "$R"; mkdir -p "$R/W" "$R/scripts" "$R/kit"
cp "$O"/kit_CHAT/chat_probe.py "$O"/kit_CHAT/chat_block.json "$O"/kit_CHAT/chat_golden.json "$R/kit/"
cp "$O"/scripts/run_gates.sh "$O"/scripts/sampling_supplement.py "$R/scripts/"
for pair in "r9700 mat-r9700 chat-r9700" "strix mat-strix chat"; do
  set -- $pair
  M=$R/W/mat-$1
  cp -rl "$O/$2" "$M"
  for f in control/chat.loom:chat.loom control/chat_sampler.loom:chat_sampler.loom dispatch/spans.loom:spans.loom dispatch/model.loom:model.loom; do
    rm -f "$M/program/${f%%:*}"; cp "$O/${f##*:}" "$M/program/${f%%:*}"
  done
  rm -f "$M/model.json"
  python3 - "$O/$2/model.json" "$M/model.json" "$3" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
p = d["presets"][sys.argv[3]]
p["execution"]["prefix_cache_capacity"] = 16
p["memory"]["transient_bytes"] += 2800000000
d["presets"]["chat"] = p
json.dump(d, open(sys.argv[2], "w"), indent=1)
PY
done
echo built "$R"
