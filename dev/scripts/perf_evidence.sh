#!/usr/bin/env bash
# usage: perf_evidence.sh <r9700|strix> BENCH_DIR MAT_DIR OUT   (GPU box)
# Writes the card evidence ev_perf.json from the perf logs of scripts/perf_session.sh (Loom rows) and the CPU llama.cpp rows (label llama-cpu-baseline).
# BENCH_DIR holds perf_log.jsonl and c4_log.jsonl; MAT_DIR is the fetched preset directory that was measured.
set -euo pipefail
G=${1:?r9700|strix}; B=$(cd "${2:?bench dir}" && pwd); MAT=$(cd "${3:?fetched preset dir}" && pwd); OUT=${4:?out file}
T=${T:-$HOME/loom/.tools/bin/python}; export LOOM_ROOT=${LOOM_ROOT:-$HOME/loom}
C=${SKILLS:-$HOME/inkling/loom-skills}/skills/loom-benchmark/scripts/card
case $G in r9700) GPU=radeon-r9700; PRESET=chat-r9700 ;; strix) GPU=strix-halo; PRESET=chat ;; *) exit 2 ;; esac
cat > "$B/commands.txt" <<EOF
loom-serve --device='amdgpu://<URI>?completion_spin_ns=4294967295' --model=<preset dir> --preset=$PRESET --pool_backing=fixed --workers=8
perf.py URL loom-baseline --model thinkingmachines/Inkling-2-Mini
perf.py URL loom-baseline --model thinkingmachines/Inkling-2-Mini --long
c4_steady.py URL 4 --model thinkingmachines/Inkling-2-Mini --label loom-baseline
perf.py URL llama-cpu-baseline --kind llama --repeats 3 [--long]  (llama-server -ngl 0 -c 33792 -np 1 -t 12 -ctk f32 -ctv f32 --jinja)
EOF
cd "$B"
$T -I $C/make_info.py perf --perf-log perf_log.jsonl --loom-label loom-baseline --llama-label llama-cpu-baseline --c4-loom c4_log.jsonl \
  --source-kind machine --gpu "$GPU" --package "$MAT" --baseline-version "llama.cpp-cpu/418fe97+PR32704" \
  --baseline-args "-ngl 0 -c 33792 -np 1 -t 12 -ctk f32 -ctv f32 --jinja" \
  --differs "not supported: ROCm and Vulkan llama.cpp builds (owner rule); this baseline is the CPU build, 12 threads, f32 KV cache" \
  --differs "not measured: 32k rows and 4 streams (baseline server has context 33792 and one slot)" \
  --commands-file commands.txt -o "$OUT"
