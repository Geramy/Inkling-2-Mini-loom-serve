"""Requests of one profiled serve session (profile_run.sh). usage: profile_driver.py URL MODEL SESSION
Sessions: base (warm-up only); dshort_p/_d, d8k_p/_d, d32k_p/_d (decode after a short, 8k, 32k prompt; _p = prefill plus 1 decode step, _d = N tokens);
pf2k, pf8k, pf32k (prefill only, max_tokens 1); c4_p/_d (4 concurrent streams). Every session starts with the same warm-up request, so
the difference of two sessions removes weight load, first-use work and the prompt prefill. Prints one JSON line."""
import json
import os
import sys
import threading
import time

sys.path.insert(0, os.environ.get("SKILLS_BENCH", os.path.expanduser("~/inkling/loom-skills/skills/loom-benchmark/scripts")))
from bench_common import doc, stream, thinking_extra

url, model, session = sys.argv[1:4]
TH = thinking_extra("off")
nonce = str(time.time())
ESSAY = "Write a long, detailed essay discussing this documentation.\n\n"
PF2K_CHARS = int(os.environ.get("PF2K_CHARS", "7940"))
TOPICS = ["the history of the printing press", "how glaciers shape valleys", "the economics of container shipping", "the rules and strategy of chess"]


def req(content, n, tag):
    s = stream(url, model, [{"role": "user", "content": content}], n, extra=TH, timeout=900)
    if s["status"] != 200:
        raise SystemExit(f"HTTP {s['status']}: {s.get('error')}")
    ttft = s["total"] if s["ttft"] is None else s["ttft"]
    return {"tag": tag, "prompt_tokens": s["prompt_tokens"], "completion_tokens": s["tokens"], "ttft_s": round(ttft, 4),
            "total_s": round(s["total"], 4), "decode_wall_s": round(s["total"] - ttft, 4)}


out = []
out.append(req(f"[{nonce} warm] Say hi.", 8, "warm"))
kind, _, part = session.partition("_")
if session == "base":
    pass
elif kind in ("dshort", "d8k", "d32k"):
    n = 2 if part == "p" else int(os.environ.get("DEC_TOKENS", "400" if kind == "dshort" else "256" if kind == "d8k" else "128"))
    text = f"[{nonce}] Write a long essay about the history of the printing press." if kind == "dshort" else \
        f"[{nonce}] {ESSAY}" + doc(8000 if kind == "d8k" else 32000)
    out.append(req(text, n, session))
elif session in ("pf2k", "pf8k", "pf32k"):
    reps = {"pf2k": 3, "pf8k": 1, "pf32k": 1}[session]
    for i in range(reps):
        if session == "pf2k":
            d = doc(2000)
            d = d[:PF2K_CHARS] if PF2K_CHARS else d
        else:
            d = doc(8000 if session == "pf8k" else 32000)
        out.append(req(f"[{nonce} {i}] Summarize this documentation in one sentence.\n\n" + d, 1, session))
elif kind == "c4":
    n = 2 if part == "p" else int(os.environ.get("DEC_TOKENS", "300"))
    res = [None] * 4
    arr = [[] for _ in range(4)]

    def run(i):
        t0 = time.perf_counter()
        s = stream(url, model, [{"role": "user", "content": f"[{nonce} {i}] Write a long essay about {TOPICS[i]}."}], n, extra=TH, timeout=900,
                   on_token=arr[i].append)
        res[i] = {"tag": session, "prompt_tokens": s["prompt_tokens"], "completion_tokens": s["tokens"], "ttft_s": round(s["ttft"] or s["total"], 4),
                  "total_s": round(s["total"], 4), "decode_wall_s": round(s["total"] - (s["ttft"] or s["total"]), 4), "t0": t0}
    th = [threading.Thread(target=run, args=(i,)) for i in range(4)]
    [t.start() for t in th]
    [t.join() for t in th]
    lo, hi = (max(a[0] for a in arr), min(a[-1] for a in arr)) if all(arr) else (0.0, 0.0)
    inwin = [sum(1 for t in a if lo <= t <= hi) for a in arr]
    for r in res:
        r.pop("t0")
    out += res
    out.append({"tag": "c4_window", "window_s": round(hi - lo, 4), "tokens_in_window": inwin})
else:
    raise SystemExit("unknown session " + session)
print(json.dumps({"session": session, "requests": out}))
