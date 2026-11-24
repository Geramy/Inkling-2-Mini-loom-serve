"""usage: chat_prefix_probe.py [--sizes 1500,3000,4500,8000] [--turns 3] [--stream] URL
Controlled prefix reuse: a system prompt of about N tokens and fixed assistant replies (no generated text), one token per request.
Turn k+1 extends turn k by one assistant and one user message. Prints prompt, cached and TTFT per turn."""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chatlib import cached, chat
from chat_cache_probe_text import system_text

ap = argparse.ArgumentParser()
ap.add_argument("url")
ap.add_argument("--sizes", default="1500,3000,4500,8000")
ap.add_argument("--turns", type=int, default=3)
ap.add_argument("--nostream", action="store_true")
ap.add_argument("--effort", default="none")
ap.add_argument("--shared", action="store_true", help="every size shares one system prefix (a different conversation each size)")
a = ap.parse_args()
base_nonce = f"{int(time.time() * 1000) % 10**9}"
for size in [int(s) for s in a.sizes.split(",")]:
    nonce = base_nonce if a.shared else f"{int(time.time() * 1000) % 10**9}"
    msgs = [{"role": "system", "content": system_text(nonce, size)}, {"role": "user", "content": f"Question about the notes, variant {size}."}]
    for turn in range(a.turns):
        r = chat(a.url, msgs, stream=not a.nostream, temperature=0, max_tokens=1, reasoning_effort=a.effort)
        pt, ct, _ = cached(r)
        print(json.dumps({"size": size, "turn": turn + 1, "prompt": pt, "cached": ct, "new": (pt - ct) if pt is not None and ct is not None else None, "ttft": round(r["ttft"], 3), "status": r["status"]}), flush=True)
        msgs = msgs + [{"role": "assistant", "content": f"Noted, turn {turn + 1}. The relevant entries are listed above."}, {"role": "user", "content": f"Next question number {turn + 2}, please keep it brief."}]
