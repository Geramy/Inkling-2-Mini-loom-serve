"""Prefill rate for prompts of a given size in characters of the skills' 1k document (consistency check against INTEGRATION_LOG's 520-token row, which is a chat request with max_tokens 1).
usage: perf_prefill_n.py URL MODEL CHARS [CHARS ...] [--repeats 5] [--log F]  -> prompt_tokens, median / min / max of prompt_tokens / TTFT and of the request wall time"""
import json
import os
import statistics
import sys
import time

sys.path.insert(0, os.environ.get("SKILLS_BENCH", os.path.expanduser("~/inkling/loom-skills/skills/loom-benchmark/scripts")))
from bench_common import doc, stream, thinking_extra

args = sys.argv[1:]
url, model = args[0], args[1]
log = "extra_log.jsonl"
reps = 5
rest = args[2:]
if "--repeats" in rest:
    i = rest.index("--repeats")
    reps = int(rest[i + 1])
    del rest[i:i + 2]
if "--log" in rest:
    i = rest.index("--log")
    log = rest[i + 1]
    del rest[i:i + 2]
text = doc(1000)
nonce = time.time()
stream(url, model, [{"role": "user", "content": f"[{nonce} warm] Say hi."}], 4, extra=thinking_extra("off"))
for c in map(int, rest):
    rs = []
    for i in range(reps):
        t0 = time.perf_counter()
        s = stream(url, model, [{"role": "user", "content": f"[{nonce} {c} {i}] Summarize this documentation in one sentence.\n\n" + text[:c]}], 4, extra=thinking_extra("off"))
        rs.append((s["prompt_tokens"], s["prompt_tokens"] / s["ttft"], time.perf_counter() - t0, s["ttft"]))
    row = {"kind": "loom-extra", "metric": "prefill_by_size", "chars": c, "prompt_tokens": rs[0][0], "prefill_tps_median": round(statistics.median(x[1] for x in rs), 1),
           "min": round(min(x[1] for x in rs), 1), "max": round(max(x[1] for x in rs), 1), "ttft_s_median": round(statistics.median(x[3] for x in rs), 3),
           "wall_s_median": round(statistics.median(x[2] for x in rs), 3), "time": time.strftime("%F %T")}
    print(json.dumps(row))
    open(log, "a").write(json.dumps(row) + "\n")
