"""Decode rate after a document of about N tokens (the skills' perf.py has 2k and 32k only). usage: perf_decode_ctx.py URL MODEL N [--label L] [--log F]
Same rules as perf.py: greedy, essay request, enable_thinking false, warm-up with another nonce, valid only when the tokens are at least half of max_tokens."""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.environ.get("SKILLS_BENCH", os.path.expanduser("~/inkling/loom-skills/skills/loom-benchmark/scripts")))
from bench_common import doc, stream, thinking_extra

ap = argparse.ArgumentParser()
ap.add_argument("url")
ap.add_argument("model")
ap.add_argument("n", type=int)
ap.add_argument("--label", default="loom-baseline")
ap.add_argument("--log", default="extra_log.jsonl")
ap.add_argument("--max-tokens", type=int, default=256)
a = ap.parse_args()
nonce = str(time.time())


def run(tag):
    msg = f"[{nonce} {tag}] Write a long, detailed essay discussing this documentation.\n\n" + doc(a.n)
    return stream(a.url, a.model, [{"role": "user", "content": msg}], a.max_tokens, extra=thinking_extra("off"))


w = run("warm")
r = run("timed")
ok = r["tokens"] >= a.max_tokens // 2
row = {"kind": "loom-extra", "label": a.label, "time": time.strftime("%F %T"), "metric": f"decode_tps_{a.n // 1000}k", "prompt_tokens": r["prompt_tokens"],
       "value": round(r["decode_tps"], 2) if ok else None, "completion_tokens": r["tokens"], "max_tokens": a.max_tokens, "ttft_s": round(r["ttft"], 3),
       "warmup_tps": round(w["decode_tps"], 2)}
print(json.dumps(row))
open(a.log, "a").write(json.dumps(row) + "\n")
