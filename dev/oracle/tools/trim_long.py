#!/usr/bin/env python3
"""Trim a full oracle dump to the tensors the attention tests read.

usage: trim_long.py <src dump dir> <dst dir> <slots comma list>
Keeps proj_q, proj_r, conv_in (occurrence 0, 1), norm (attention input occurrences), Qcur, Kcur, Vcur, Rcur, kqv_out.
"""
import json, os, shutil, sys

src, dst, slots = sys.argv[1], sys.argv[2], [int(s) for s in sys.argv[3].split(",")]
only = sys.argv[4].split(",") if len(sys.argv) > 4 else None
m = json.load(open(f"{src}/manifest.json"))
glob = {5, 11, 17, 23}
keep = []
for t in m["tensors"]:
    if t["layer"] not in slots:
        continue
    b, o = t["base"], t["occurrence"]
    ok = b in ("proj_q", "proj_r", "Qcur", "Kcur", "Vcur", "Rcur", "kqv_out") or (b == "conv_in" and o < 2) or (b == "norm" and o < (3 if t["layer"] % 24 in glob or (t["layer"] >= 24 and (6 + (t["layer"] - 24) % 12) in glob) else 1))
    if ok and (only is None or b in only):
        keep.append(t)
os.makedirs(dst, exist_ok=True)
tot = 0
for t in keep:
    shutil.copy(f"{src}/{t['file']}", f"{dst}/{t['file']}")
    tot += os.path.getsize(f"{dst}/{t['file']}")
m["tensors"] = keep
m["mode"] = "trimmed"
json.dump(m, open(f"{dst}/manifest.json", "w"), indent=1)
print(len(keep), "tensors", tot / 1e6, "MB")
