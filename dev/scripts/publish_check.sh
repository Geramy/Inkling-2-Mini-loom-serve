#!/usr/bin/env bash
# usage: publish_check.sh TREE SNAP WORK REF [r9700|strix ...]   (GPU box; wrap in: flock -s ~/hrx-pr1/gpu.lock)
# Clean-room check of the publish tree. No upload.
#  a) scrub (host, user, paths, agent names), required files, tree size
#  b) loom-fetch of the preset into an EMPTY directory (--materialize=share, origin = local snapshot SNAP)
#  c) loom-serve of that directory, wait_model.py, oracle prompt C greedy 64 tokens equal to the reference REF/<gpu>.json (route_check.py output, key C)
#  d) the quick start commands of README.md as printed (only the origin and the fetch target are local)
# WORK is scratch on the disk of the GGUF files (share = hard links, no copy); it is emptied at the end. Servers are stopped by PID.
set -uo pipefail
TREE=$(cd "${1:?tree}" && pwd); SNAP=$(cd "${2:?snap}" && pwd); mkdir -p "${3:?work}"; WORK=$(cd "$3" && pwd); REF=$(cd "${4:?ref dir}" && pwd); shift 4
GPUS=${*:-r9700 strix}
VENV=${LOOM_VENV:-$HOME/loom/.venv}; K=${SKILLS:-$HOME/inkling/loom-skills/skills}/loom-benchmark/scripts
T=${T:-$HOME/loom/.tools/bin/python}; export LOOM_ROOT=${LOOM_ROOT:-$HOME/loom}
HERE=$(cd "$(dirname "$0")" && pwd); PROMPT=${PROMPT:-$HOME/inkling/inkB/oracle/prompts/C_user.txt}
FAIL=0; PID=
ok() { echo "PASS  $*"; }
bad() { echo "FAIL  $*"; FAIL=1; }
stop_server() { [ -n "$PID" ] && tr '\0' ' ' < /proc/$PID/cmdline 2>/dev/null | grep -q loom-serve && { kill "$PID"; for _ in $(seq 60); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done; }; PID=; }
trap 'stop_server; rm -rf "$WORK"/clean-* "$WORK"/qs-*' EXIT

echo "== a) scrub and files"
bash "$HERE/scrub_check.sh" "$TREE" && ok "scrub_check" || bad "scrub_check"
for f in .gitattributes LICENSE NOTICE README.md REPORT.md info.json PORTING.md model.json model.json.lock llm_config.json weights.loom tokenizer.json; do [ -f "$TREE/$f" ] && ok "present $f" || bad "missing $f"; done
unnamed=$($T -I - "$TREE" <<'PY'
import json, os, sys
d = sys.argv[1]; m = json.load(open(os.path.join(d, "model.json"))); named = set()
for s in m["programs"].values():
    for e in s["inputs"] + s.get("libraries", []): named.add(e if isinstance(e, str) else e["path"])
have = {os.path.relpath(os.path.join(r, f), d).replace(os.sep, "/") for r, _, fs in os.walk(os.path.join(d, "program")) for f in fs if f.endswith(".loom")}
print(sorted(have ^ named))
PY
)
[ "$unnamed" = "[]" ] && ok "every .loom file is named by model.json" || bad "loom files and model.json differ: $unnamed"
echo "tree: $(find "$TREE" -type f | wc -l) files, $(du -sb "$TREE" | cut -f1) bytes (no weights)"; find "$TREE" -type f -size +50M | grep . && bad "file over 50 MB in the tree"

wait_ready() { # url
  for _ in $(seq 600); do
    kill -0 "$PID" 2>/dev/null || { echo "loom-serve died"; return 1; }
    curl -s -m 2 "$1/v1/status" 2>/dev/null | grep -q '"registration": *"ready"' && return 0; sleep 1
  done; return 1
}

for G in $GPUS; do
  case $G in
    r9700) PRESET=chat-r9700; PORT=18290; URI=amdgpu://GPU-e0880c05003c324d; NAME=1201 ;;
    strix) PRESET=chat;       PORT=18291; URI=amdgpu://id/000000e2000000808e270f2b0bbb4e00; NAME=1151 ;;
    *) echo "unknown gpu $G"; exit 2 ;;
  esac
  URL=http://127.0.0.1:$PORT; CLEAN=$WORK/clean-$G
  echo "== b) $G: fetch preset $PRESET into an empty directory"
  rm -rf "$CLEAN"; mkdir -p "$CLEAN"; [ -z "$(ls -A "$CLEAN")" ] || bad "$CLEAN not empty"
  "$VENV/bin/loom-fetch" "$TREE/model.json" --preset "$PRESET" --output "$CLEAN" --materialize=share --origin checkpoint="$SNAP" --no-input >/dev/null 2>&1 && ok "loom-fetch $PRESET" || { bad "loom-fetch $PRESET"; continue; }
  echo "fetched: $(find "$CLEAN" -type f | wc -l) files; loom files: $(find "$CLEAN" -name '*.loom' | wc -l); has lib/pr1: $([ -e "$CLEAN/program/lib/pr1" ] && echo yes || echo no)"
  echo "== c) $G: serve and ask"
  (cd "$WORK"; exec nohup "$VENV/bin/loom-serve" --device="$URI" --model="$CLEAN" --preset="$PRESET" --pool_backing=fixed --listen=127.0.0.1:$PORT --workers=4 > "$WORK/serve-$G.log" 2>&1 < /dev/null) &
  PID=$!
  wait_ready "$URL" && ok "registered" || { bad "not ready"; tail -5 "$WORK/serve-$G.log"; stop_server; continue; }
  "$T" -I "$K/wait_model.py" "$URL" --wait 600 >/dev/null 2>&1 && ok "wait_model.py" || bad "wait_model.py"
  curl -s "$URL/v1/models" | $T -I -c 'import json,sys; d=json.load(sys.stdin); print("models", [(m["id"], m.get("context_length", m.get("context_capacity"))) for m in d["data"]])'
  $T -I - "$PORT" "$PROMPT" "$REF/$G.json" <<'PY' && ok "prompt C greedy 64 tokens equals the reference" || bad "prompt C differs from the reference"
import http.client, json, sys
port, pf, ref = int(sys.argv[1]), sys.argv[2], sys.argv[3]
c = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
body = {"model": "thinkingmachines/Inkling-2-Mini", "max_tokens": 64, "temperature": 0, "messages": [{"role": "user", "content": open(pf, encoding="utf-8").read().strip()}]}
c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
d = json.loads(c.getresponse().read()); m = d["choices"][0]["message"]
got = {"usage": d["usage"], "finish": d["choices"][0]["finish_reason"], "reasoning": m.get("reasoning_content") or "", "content": m.get("content") or ""}
want = json.load(open(ref))["C"]
print("usage", got["usage"]["prompt_tokens"], got["usage"]["completion_tokens"], "text", repr((got["reasoning"] + got["content"])[:80]))
sys.exit(0 if got == want else 1)
PY
  stop_server

  echo "== d) $G: quick start of README.md as printed"
  QS=$WORK/qs-$G; rm -rf "$QS"; mkdir -p "$QS"
  $T -I - "$TREE/README.md" "$G" "$QS/cmds.sh" "$TREE" "$SNAP" <<'PY'
import re, sys
text = open(sys.argv[1]).read(); g, out, tree, snap = sys.argv[2:6]
want = {"r9700": "gfx1201", "strix": "gfx1151"}[g]
blocks = re.findall(r"<details>\s*<summary>(.*?)</summary>\s*```bash\n(.*?)```", text, re.S)
body = [b for s, b in blocks if want in s][0]
lines = []
for l in body.splitlines():
    if l.startswith("pipx "): continue
    m = re.match(r"loom-fetch huggingface:\S+?:(\S+) (.*)", l)
    if m: l = f"loom-fetch {tree}/model.json --preset {m.group(1)} {m.group(2)} --origin checkpoint={snap} --no-input"
    lines.append(l)
open(out, "w").write("\n".join(lines) + "\n")
PY
  sed -i "s#--output=unsloth-Inkling-2-Mini-loom#--output=$QS/unsloth-Inkling-2-Mini-loom#; s#--materialize=copy#--materialize=share#; s#--model=unsloth-Inkling-2-Mini-loom#--model=$QS/unsloth-Inkling-2-Mini-loom#" "$QS/cmds.sh"
  sed 's/^/  qs: /' "$QS/cmds.sh" | cut -c1-230
  FETCH=$(grep '^loom-fetch' "$QS/cmds.sh"); SERVE=$(grep '^loom-serve' "$QS/cmds.sh")
  if ss -ltn | grep -q ':8080 '; then bad "port 8080 busy, quick start serve skipped"; continue; fi
  (. "$VENV/bin/activate"; eval "$FETCH") >/dev/null 2>&1 && ok "quick start fetch" || { bad "quick start fetch"; continue; }
  (cd "$QS"; . "$VENV/bin/activate"; eval "exec nohup $SERVE" > "$QS/serve.log" 2>&1 < /dev/null) &
  PID=$!
  wait_ready http://127.0.0.1:8080 && ok "quick start serve registered" || { bad "quick start serve not ready"; tail -5 "$QS/serve.log"; stop_server; continue; }
  curl -s http://127.0.0.1:8080/v1/status | $T -I -c 'import json,sys; d=json.load(sys.stdin); print("device", [x.get("name") for x in d["server"]["resources"]["devices"]])'
  R=$(curl -s -m 600 http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{"model": "thinkingmachines/Inkling-2-Mini", "messages": [{"role": "user", "content": "Hello"}], "reasoning_effort": "none"}')
  echo "$R" | $T -I -c 'import json,sys; d=json.load(sys.stdin); m=d["choices"][0]["message"]; print("reply", d["usage"]["completion_tokens"], repr((m.get("content") or "")[:100]))' && ok "quick start request" || bad "quick start request: ${R:0:200}"
  stop_server
done
echo "publish_check: $([ $FAIL = 0 ] && echo PASS || echo FAIL)"
exit $FAIL
