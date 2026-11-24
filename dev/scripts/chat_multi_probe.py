"""usage: chat_multi_probe.py [--convs 6] [--turns 2] [--tokens 3000] [--echo] [--shared] URL
Round-robin conversations (turn k of every conversation before turn k+1). Generated replies are fed back (with reasoning_content when --echo).
--shared: all conversations share the same system prompt and differ only in the user message (a common-prefix case)."""
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
ap.add_argument("--convs", type=int, default=6)
ap.add_argument("--turns", type=int, default=2)
ap.add_argument("--tokens", type=int, default=3000)
ap.add_argument("--echo", action="store_true")
ap.add_argument("--shared", action="store_true")
ap.add_argument("--max-tokens", type=int, default=1500)
ap.add_argument("--out", default="")
a = ap.parse_args()
nonce = f"{int(time.time() * 1000) % 10**9}"
convs = []
for i in range(a.convs):
    sysm = system_text(nonce if a.shared else f"{nonce}{i}", a.tokens)
    convs.append([{"role": "system", "content": sysm}, {"role": "user", "content": f"Question {i}: in one sentence, what is on note {i} of the list?"}])
rows = []
for turn in range(a.turns):
    for i, msgs in enumerate(convs):
        r = chat(a.url, msgs, temperature=0, max_tokens=a.max_tokens, reasoning_effort="none")
        pt, ct, comp = cached(r)
        row = {"conv": i, "turn": turn + 1, "prompt": pt, "cached": ct, "completion": comp, "ttft": round(r["ttft"], 2), "finish": r["finish"], "status": r["status"], "text": r["content"][:60], "reasoning": r["reasoning"][:60]}
        rows.append(row)
        print(json.dumps(row), flush=True)
        m = {"role": "assistant", "content": r["content"]}
        if a.echo and r["reasoning"]:
            m["reasoning_content"] = r["reasoning"]
        convs[i] = msgs + [m, {"role": "user", "content": f"Follow-up {turn}: say it again with fewer words."}]
if a.out:
    open(a.out, "a").write("\n".join(json.dumps(r) for r in rows) + "\n")
