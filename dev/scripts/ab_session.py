"""One A/B arm session: usage: ab_session.py URL MODEL TAG METRICS [--reps N] [--out FILE]
METRICS comma list of: dshort, d8k, d32k, pf2k, pf8k, c4. Warm-up request first; every timed request has its own nonce.
Prints and appends one JSON line {tag, metric: median tok/s, runs}. Decode: (tokens - 1) / (total - first text token)."""
import argparse
import json
import os
import statistics
import sys
import threading
import time

sys.path.insert(0, os.environ.get("SKILLS_BENCH", os.path.expanduser("~/inkling/loom-skills/skills/loom-benchmark/scripts")))
from bench_common import doc, stream, thinking_extra

ap = argparse.ArgumentParser()
ap.add_argument("url"); ap.add_argument("model"); ap.add_argument("tag"); ap.add_argument("metrics")
ap.add_argument("--reps", type=int, default=3); ap.add_argument("--out", default="ab_log.jsonl")
a = ap.parse_args()
TH = thinking_extra("off")
ESSAY = "Write a long, detailed essay discussing this documentation.\n\n"
TOPICS = ["the history of the printing press", "how glaciers shape valleys", "the economics of container shipping", "the rules and strategy of chess"]


def go(text, n):
    return stream(a.url, a.model, [{"role": "user", "content": text}], n, extra=TH, timeout=900)


def decode(prefix, n):
    s = go(f"[{time.time()}] {prefix}", n)
    return s["decode_tps"] if s["tokens"] >= n // 2 else None


def pf(chars):
    chars = chars or 10**6
    s = go(f"[{time.time()}] Summarize this documentation in one sentence.\n\n" + doc(2000)[:chars], 1)
    return s["prompt_tokens"] / s["total"]


def c4(n=300):
    arr = [[] for _ in range(4)]
    th = [threading.Thread(target=lambda i=i: stream(a.url, a.model, [{"role": "user", "content": f"[{time.time()} {i}] Write a long essay about {TOPICS[i]}."}], n, extra=TH, timeout=900, on_token=arr[i].append)) for i in range(4)]
    [t.start() for t in th]; [t.join() for t in th]
    lo, hi = max(x[0] for x in arr), min(x[-1] for x in arr)
    return sum(sum(1 for t in x if lo <= t <= hi) for x in arr) / (hi - lo)


def pf4(chars=0):
    s = go(f"[{time.time()}] Summarize this documentation in one sentence.\n\n" + (doc(2000)[:chars] if chars else doc(2000)), 4)
    return s["prompt_tokens"] / (s["ttft"] or s["total"])


M = {"pf2k4": lambda: pf4(), "pf2kfull": lambda: pf(0), "dshort": lambda: decode("Write a long essay about the history of the printing press.", 300),
     "d8k": lambda: decode(ESSAY + doc(8000), 192), "d32k": lambda: decode(ESSAY + doc(32000), 96),
     "pf2k": lambda: pf(7940), "pf8k": lambda: go(f"[{time.time()}] Summarize.\n\n" + doc(8000), 1)["prompt_tokens"] / 1.0, "c4": c4}
go(f"[{time.time()}] Say hi.", 8)
decode("Write a long essay about the history of the printing press.", 120)
res = {"tag": a.tag}
for m in a.metrics.split(","):
    if m == "pf8k":
        vals = []
        for _ in range(a.reps):
            s = go(f"[{time.time()}] Summarize this documentation in one sentence.\n\n" + doc(8000), 1)
            vals.append(s["prompt_tokens"] / s["total"])
    else:
        M[m]()
        vals = [M[m]() for _ in range(a.reps)]
    vals = [v for v in vals if v]
    res[m] = round(statistics.median(vals), 2) if vals else None
    res[m + "_runs"] = [round(v, 2) for v in vals]
print(json.dumps(res))
open(a.out, "a").write(json.dumps(res) + "\n")
