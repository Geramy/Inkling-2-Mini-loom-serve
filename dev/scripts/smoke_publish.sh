#!/usr/bin/env bash
# usage: smoke_publish.sh <r9700|strix> [inkling dir of the work dir, default inkS]   (GPU box; wrap in flock -s ~/hrx-pr1/gpu.lock)
# Serves the materialized preset, sends oracle prompt C greedily (64 tokens), prints the registered context, memory in use and a hash of the answer, stops the server by PID.
set -euo pipefail
G=${1:?r9700|strix}; export INKR=$HOME/inkling/${2:-inkS}
S=$(dirname "$0")/run_gates.sh
case $G in r9700) PORT=18190 ;; strix) PORT=18191 ;; esac
trap 'bash "$S" "$G" stop' EXIT
bash "$S" "$G" serve | tail -2
URL=http://127.0.0.1:$PORT
curl -s $URL/v1/models | python3 -c 'import json,sys; d=json.load(sys.stdin); print("models", [(m["id"], m.get("context_length", m.get("context_capacity", m))) for m in d["data"]])'
python3 - "$URL" "$HOME/inkling/inkB/oracle/prompts/C_user.txt" "$INKR/smoke_$G.json" <<'PY'
import json, sys, urllib.request, glob, hashlib
url, pf, out = sys.argv[1:]
body = {"model": "thinkingmachines/Inkling-2-Mini", "max_tokens": 64, "temperature": 0, "messages": [{"role": "user", "content": open(pf).read()}]}
r = json.load(urllib.request.urlopen(urllib.request.Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=600))
m = r["choices"][0]["message"]
res = {"reasoning": m.get("reasoning_content"), "content": m.get("content"), "usage": r["usage"]}
json.dump(res, open(out, "w"), indent=1)
print("usage", r["usage"], "sha", hashlib.sha256(json.dumps(res, sort_keys=True).encode()).hexdigest()[:16])
for c in sorted(glob.glob("/sys/class/drm/card[0-9]/device")):
    try: print(c.split("/")[4], {k: round(int(open(f"{c}/mem_info_{k}_used").read()) / 1e9, 2) for k in ("vram", "gtt")})
    except OSError: pass
PY
