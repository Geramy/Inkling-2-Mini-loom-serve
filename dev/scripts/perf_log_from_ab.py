#!/usr/bin/env python3
"""usage: perf_log_from_ab.py AB_LOG.jsonl TAG OUT_PERF_LOG OUT_C4_LOG [DATE]
Turns one ab_session.py row (decode short, decode 32k, 4 streams) into a perf.py style session for make_info.py perf.
Only metrics that the row holds are written. The three request values of a metric go into "samples" (card metric names)."""
import json, sys

ab, tag, perf_out, c4_out = sys.argv[1:5]
date = sys.argv[5] if len(sys.argv) > 5 else "2026-11-22"
row = next(r for r in map(json.loads, open(ab)) if r["tag"] == tag)
samples = {"decode": row["dshort_runs"], "decode_32k": row["d32k_runs"]}
with open(perf_out, "a") as f:
    f.write(json.dumps({"label": "loom-baseline", "kind": "loom", "time": date + " 00:00:00", "thinking": "off", "samples": samples}) + "\n")
    f.write(json.dumps({"session_end": tag}) + "\n")
with open(c4_out, "a") as f:
    for v in row["c4_runs"]:
        f.write(json.dumps({"time": date + " 00:00:00", "streams": 4, "thinking": "off", "agg_tps": v}) + "\n")
    f.write(json.dumps({"session_end": tag}) + "\n")
