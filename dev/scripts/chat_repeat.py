"""usage: chat_repeat.py [--n 5] [--tokens 5000] [--max-tokens 96] [--stream] [--flush] URL
Sends the same greedy request n times; prints cached tokens and where each output first differs from the first. --flush sends an unrelated long request between the repeats."""
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
ap.add_argument("--n", type=int, default=5)
ap.add_argument("--tokens", type=int, default=5000)
ap.add_argument("--max-tokens", type=int, default=96)
ap.add_argument("--stream", action="store_true")
ap.add_argument("--flush", action="store_true")
ap.add_argument("--seedtag", default="alpha-fixed")
a = ap.parse_args()
msgs = [{"role": "system", "content": system_text(a.seedtag, a.tokens)}, {"role": "user", "content": "What is the survey count of the first glaciers note?"}]
outs = []
for i in range(a.n):
    r = chat(a.url, msgs, stream=a.stream, temperature=0, max_tokens=a.max_tokens, reasoning_effort="none")
    text = r["reasoning"] + "|" + r["content"]
    outs.append(text)
    k = next((j for j, (x, y) in enumerate(zip(outs[0], text)) if x != y), -1 if len(text) == len(outs[0]) else min(len(text), len(outs[0])))
    print(json.dumps({"i": i, "cached": cached(r)[1], "prompt": cached(r)[0], "len": len(text), "first_diff_vs_first": k, "ttft": round(r["ttft"], 3)}), flush=True)
    if a.flush:
        chat(a.url, [{"role": "system", "content": system_text(f"flush{i}", a.tokens)}, {"role": "user", "content": "Say hi."}], temperature=0, max_tokens=4, reasoning_effort="none")
