"""usage: chat_fill_ctx.py [--convs 4] [--tokens 8000] [--big 120000] URL
Fills the prefix cache with several conversations, then sends one request of about --big tokens (the pool must reclaim cache entries), then repeats a first conversation."""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chatlib import cached, chat, get
from chat_cache_probe_text import system_text

ap = argparse.ArgumentParser()
ap.add_argument("url")
ap.add_argument("--convs", type=int, default=4)
ap.add_argument("--tokens", type=int, default=8000)
ap.add_argument("--big", type=int, default=120000)
a = ap.parse_args()


def obs():
    out = {}
    for l in str(get(a.url, "/metrics")).splitlines():
        for k in ("pool_resident", "pool_capacity", "prefix_cache_endpoints"):
            if f'field="{k}"' in l and not l.startswith("#"):
                out[k] = l.rsplit(" ", 1)[1]
    return out


convs = [[{"role": "system", "content": system_text(f"fill{i}-{int(time.time())}", a.tokens)}, {"role": "user", "content": "Say hi."}] for i in range(a.convs)]
for i, m in enumerate(convs):
    r = chat(a.url, m, temperature=0, max_tokens=4, reasoning_effort="none")
    print(json.dumps({"fill": i, "prompt": cached(r)[0], "cached": cached(r)[1], "ttft": round(r["ttft"], 2), **obs()}), flush=True)
big = [{"role": "system", "content": system_text(f"big-{int(time.time())}", a.big)}, {"role": "user", "content": "Say hi."}]
r = chat(a.url, big, temperature=0, max_tokens=4, reasoning_effort="none")
print(json.dumps({"big": True, "status": r["status"], "prompt": cached(r)[0], "cached": cached(r)[1], "total": round(r["total"], 1), "error": r["error"] and str(r["error"])[:200], **obs()}), flush=True)
r = chat(a.url, convs[-1], temperature=0, max_tokens=4, reasoning_effort="none")
print(json.dumps({"repeat_last_fill": True, "status": r["status"], "prompt": cached(r)[0], "cached": cached(r)[1], "ttft": round(r["ttft"], 2), **obs()}), flush=True)
