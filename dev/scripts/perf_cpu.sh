#!/usr/bin/env bash
# usage: perf_cpu.sh   (GPU box; run under: flock -x ~/hrx-pr1/gpu.lock, strictly after all GPU timing)
# CPU llama.cpp reference row (oracle build 418fe97 + PR 32704, 12 threads, -ngl 0, f32 KV): skills perf.py --kind llama against the oracle server, 3 short runs (--repeats 3) and 1 --long run.
# Rows go to bench-r9700/perf_log.jsonl (label llama-cpu-baseline); the server is stopped at the end.
set -u
U=${INKU:-$HOME/inkling/inkU}; B=$U/bench-r9700
K=$HOME/inkling/loom-skills/skills/loom-benchmark/scripts; T=$HOME/loom/.tools/bin/python
export LOOM_ROOT=$HOME/loom PYTHONDONTWRITEBYTECODE=1 LOOM_KIT=$HOME/inkling/inkR/kit
URL=http://127.0.0.1:18400; L=llama-cpu-baseline; R=$B/raw/cpu; mkdir -p "$R"; cd "$B" || exit 1
if pgrep -af 'loom-serve|llama-server|llama-bench' | grep -v pgrep | grep -q .; then echo "another server runs"; exit 3; fi
{ date -u +%FT%TZ; uptime; } > "$R/preflight.txt"
bash $U/scripts/oracle_server.sh start q4 || exit 1
trap 'bash $U/scripts/oracle_server.sh stop' EXIT
curl -s -m 600 $URL/v1/chat/completions -H 'Content-Type: application/json' -d '{"max_tokens":4,"temperature":0,"messages":[{"role":"user","content":"Say hi."}]}' > /dev/null
for i in 1 2 3; do
  $T $K/perf.py $URL $L --kind llama --repeats 3 --log perf_log.jsonl > "$R/short$i.log" 2>&1; tail -n1 "$R/short$i.log" | cut -c1-400
  echo "{\"session_end\": \"cpu-$i\"}" >> perf_log.jsonl
done
$T $K/perf.py $URL $L --kind llama --long --log perf_log.jsonl > "$R/long.log" 2>&1; tail -n1 "$R/long.log" | cut -c1-500
echo "{\"session_end\": \"cpu-long\"}" >> perf_log.jsonl
