"""usage: chat_cache_correct.py [--tokens N] [--out F] URL      then: chat_cache_correct.py --compare A.json B.json
A fixed sequence of greedy requests (fixed assistant history, so every request is a function of the script only): a conversation and its follow-ups,
a fork of the same history, a second conversation that shares only the system prompt, an unrelated conversation, eviction traffic, and repeats.
The output (reasoning, content, completion tokens) of every request is written; --compare checks that two runs (cache on and off) are identical."""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chatlib import cached, chat
from chat_cache_probe_text import system_text

ap = argparse.ArgumentParser()
ap.add_argument("url", nargs="?")
ap.add_argument("--tokens", type=int, default=5000)
ap.add_argument("--out", default="")
ap.add_argument("--max-tokens", type=int, default=96)
ap.add_argument("--compare", nargs=2)
a = ap.parse_args()

if a.compare:
    A, B = (json.load(open(p)) for p in a.compare)
    bad = 0
    for x, y in zip(A, B):
        same = (x["reasoning"], x["content"], x["completion"], x["prompt"]) == (y["reasoning"], y["content"], y["completion"], y["prompt"])
        bad += not same
        print("SAME" if same else "DIFF", x["label"], f"prompt {x['prompt']} cached {x['cached']} vs {y['cached']}")
    print(f"{'FAIL' if bad else 'PASS'}: {bad} different of {len(A)}")
    sys.exit(1 if bad or len(A) != len(B) else 0)

SYS_X = system_text("alpha-fixed", a.tokens)
SYS_Z = system_text("zulu-fixed", a.tokens)
ASSIST = lambda i: {"role": "assistant", "content": f"Noted ({i}). The listed entries are consistent."}
USER = lambda t: {"role": "user", "content": t}
X1 = [{"role": "system", "content": SYS_X}, USER("What is the survey count of the first glaciers note?")]
X2 = X1 + [ASSIST(1), USER("And the margin of that same note?")]
X2b = X1 + [ASSIST(1), USER("Name the topic of note 3 instead.")]
X3 = X2 + [ASSIST(2), USER("List the next review cycle of note 2.")]
Y1 = [{"role": "system", "content": SYS_X}, USER("Which note mentions railway signalling first?")]
Y2 = Y1 + [ASSIST(1), USER("What is its block number?")]
Z1 = [{"role": "system", "content": SYS_Z}, USER("What is the survey count of the first glaciers note?")]
Z2 = Z1 + [ASSIST(1), USER("And the margin of that same note?")]
EVICT = [[{"role": "system", "content": system_text(f"filler{i}", a.tokens)}, USER("Summarize note 1 briefly.")] for i in range(4)]
SEQ = [("X1 cold", X1), ("X2 follow-up", X2), ("X2b fork of the same history", X2b), ("Y1 shares only the system prompt", Y1), ("Y2 follow-up", Y2), ("Z1 other conversation", Z1), ("Z2 follow-up", Z2),
       ("X3 second follow-up", X3), ("X2 repeat", X2)] + [(f"EVICT{i}", m) for i, m in enumerate(EVICT)] + [("X2 after eviction traffic", X2), ("X1 after eviction traffic", X1), ("Z2 repeat", Z2)]
rows = []
for label, msgs in SEQ:
    r = chat(a.url, msgs, stream=len(rows) % 2 == 1, temperature=0, max_tokens=a.max_tokens, reasoning_effort="none")
    pt, ct, comp = cached(r)
    row = {"label": label, "prompt": pt, "cached": ct, "completion": comp, "reasoning": r["reasoning"], "content": r["content"], "ttft": round(r["ttft"], 3), "status": r["status"]}
    rows.append(row)
    print(json.dumps({k: row[k] for k in ("label", "prompt", "cached", "completion", "ttft", "status")}), flush=True)
if a.out:
    json.dump(rows, open(a.out, "w"))
