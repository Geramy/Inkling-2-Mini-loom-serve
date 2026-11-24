"""usage: chat_numerics.py [--n 24] [--tokens 5000] [--max-tokens 40] [--reps 3] --out F URL      then: chat_numerics.py --compare A.json B.json [C.json ...]
For n fixed prompts, sends each one reps times in a row (cold, hit, hit) with greedy decoding and stores the generated text.
--compare: for every pair of files, the fraction of prompts whose first-run text is identical, and the median first-difference position (characters)."""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chatlib import cached, chat
from chat_cache_probe_text import system_text

ap = argparse.ArgumentParser()
ap.add_argument("url", nargs="?")
ap.add_argument("--n", type=int, default=24)
ap.add_argument("--tokens", type=int, default=5000)
ap.add_argument("--max-tokens", type=int, default=40)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--out", default="")
ap.add_argument("--compare", nargs="+")
a = ap.parse_args()


def first_diff(x, y):
    return next((i for i, (p, q) in enumerate(zip(x, y)) if p != q), -1 if len(x) == len(y) else min(len(x), len(y)))


if a.compare:
    files = [json.load(open(p)) for p in a.compare]
    for i, fa in enumerate(files):
        for j in range(i + 1, len(files)):
            fb = files[j]
            same = [x["texts"][0] == y["texts"][0] for x, y in zip(fa, fb)]
            diffs = [first_diff(x["texts"][0], y["texts"][0]) for x, y in zip(fa, fb) if x["texts"][0] != y["texts"][0]]
            print(f"{a.compare[i]} vs {a.compare[j]} (first run): identical {sum(same)}/{len(same)}; first difference at chars {sorted(diffs)}")
    for name, f in zip(a.compare, files):
        reps = [[x["texts"][0] == t for t in x["texts"][1:]] for x in f]
        flat = [v for r in reps for v in r]
        if flat:
            print(f"{name}: repeat of the same prompt equals its first run in {sum(flat)}/{len(flat)} requests (cached tokens of repeats: {sorted({c for x in f for c in x['cached'][1:]})})")
    sys.exit(0)

rows = []
for k in range(a.n):
    msgs = [{"role": "system", "content": system_text(f"num{k}", a.tokens)}, {"role": "user", "content": f"Question {k}: what is the survey count of note {k} in the list?"}]
    texts, cs = [], []
    for _ in range(a.reps):
        r = chat(a.url, msgs, temperature=0, max_tokens=a.max_tokens, reasoning_effort="none")
        texts.append(r["reasoning"] + "|" + r["content"])
        cs.append(cached(r)[1])
    rows.append({"k": k, "texts": texts, "cached": cs})
    print(json.dumps({"k": k, "cached": cs, "hit_equals_cold": [t == texts[0] for t in texts[1:]]}), flush=True)
if a.out:
    json.dump(rows, open(a.out, "w"))
