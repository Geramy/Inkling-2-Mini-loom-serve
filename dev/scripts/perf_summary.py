#!/usr/bin/env python3
"""Summary of a bench dir written by perf_session.sh: per metric the raw per-session values, the median, the spread, and the efficiency against bench/<gpu>/roofline.json.
usage: perf_summary.py BENCH_DIR [log suffix: '' (spin, default) or _bare, _blocking]   (standard library only)"""
import json
import statistics
import sys

d = sys.argv[1]
suf = sys.argv[2] if len(sys.argv) > 2 else ""
roof = json.load(open(f"{d}/roofline_ours.json"))["checkpoints"]["ud-q4-k-xl"]["metrics"]
DOC = {1258: "prefill_1k", 2454: "prefill_2k", 9083: "prefill_8k", 18436: "prefill_16k", 37007: "prefill_32k"}


def rows(path):
    try:
        return [json.loads(x) for x in open(path) if x.strip().startswith("{")]
    except OSError:
        return []


def sessions(rs):
    out, cur = [], []
    for r in rs:
        if "session_end" in r:
            out.append(cur)
            cur = []
        else:
            cur.append(r)
    return out + ([cur] if cur else [])


vals = {}
for i, s in enumerate(sessions(rows(f"{d}/perf_log{suf}.jsonl"))):
    for r in (x for x in s if x.get("kind", "loom") == "loom"):
        for k, m in (("decode_tps", "decode"), ("decode_tps_2k", "decode_2k"), ("decode_tps_32k", "decode_32k")):
            if r.get(k) is not None:
                vals.setdefault(m, []).append(r[k])
        for p in r.get("prefill", []) + r.get("prefill_long", []):
            n = min(DOC, key=lambda x: abs(x - p["prompt_tokens"]))
            vals.setdefault(DOC[n], []).append(p["prefill_tps"])
for s in sessions(rows(f"{d}/c4_log{suf}.jsonl")):
    for r in s:
        if r.get("agg_tps"):
            vals.setdefault("streams4", []).append(r["agg_tps"])
for r in rows(f"{d}/extra_log.jsonl"):
    if r.get("metric", "").startswith("decode") and r["label"] == "loom-baseline" + (suf.replace("_", "-") if suf else "") and r["value"] is not None:
        vals.setdefault("decode_8k", []).append(r["value"])
order = ["decode", "decode_2k", "decode_8k", "decode_32k", "prefill_1k", "prefill_2k", "prefill_8k", "prefill_16k", "prefill_32k", "streams4"]
print(f"{'metric':11} {'n':>2} {'median':>8} {'min':>8} {'max':>8} {'spread%':>7} {'ceiling':>8} {'target':>8} {'eff':>5} {'eff/target':>10}  raw")
for m in order:
    v = vals.get(m)
    if not v:
        continue
    med = statistics.median(v)
    r = roof.get(m, {})
    c, t = r.get("ceiling"), r.get("target")
    print(f"{m:11} {len(v):2d} {med:8.1f} {min(v):8.1f} {max(v):8.1f} {100 * (max(v) - min(v)) / med:7.1f} {c or 0:8.1f} {t or 0:8.1f} {med / c if c else 0:5.2f} {med / t if t else 0:10.2f}  {v}")
