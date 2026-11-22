#!/usr/bin/env python3
"""Markdown tables for docs/KERNELS_attn.md (run on the box in the work dir).

usage: mk_table.py tests|perf   reads results_<dev>.txt (attn_test.py) or perf_<dev>.txt (attn_perf.py)
"""
import re, sys

devs = ["gfx1201", "gfx1151"]


def tests():
    data = {}
    for d in devs:
        for line in open(f"results_{d}.txt"):
            m = re.match(r"(.+?) (q|r|k_act|v_act|out)\s+gfx\d+\s+(\S+)\s+\S+\s+(PASS|FAIL)", line)
            if m:
                label, field, v, res = m.groups()
                data.setdefault(label, {}).setdefault(d, {})[field] = (float(v), res)
    print("| case | q | r | k (F16) | v (F16) | out gfx1201 | out gfx1151 | result |")
    print("|---|---|---|---|---|---|---|---|")
    for label, per in data.items():
        worst = lambda f: max(per[d][f][0] for d in devs if d in per)
        ok = all(per[d][f][1] == "PASS" for d in devs if d in per for f in per[d])
        print(f"| {label} | {worst('q'):.1e} | {worst('r'):.1e} | {worst('k_act'):.1e} | {worst('v_act'):.1e} | "
              f"{per['gfx1201']['out'][0]:.1e} | {per['gfx1151']['out'][0]:.1e} | {'PASS' if ok else 'FAIL'} |")


def perf():
    rows = {}
    for d in devs:
        for line in open(f"perf_{d}.txt"):
            p = line.split()
            if len(p) in (2, 6) and p[0] != "kernel":
                rows.setdefault(p[0], {})[d] = p[1:]
    print("| run | R9700 us | floor us | x floor | 8060S us | floor us | x floor |")
    print("|---|---|---|---|---|---|---|")
    for name, per in rows.items():
        cells = []
        for d in devs:
            r = per.get(d)
            cells += ([r[0], r[3], r[4]] if len(r) == 5 else [r[0], "", ""]) if r else ["", "", ""]
        print(f"| {name} | " + " | ".join(cells) + " |")


{"tests": tests, "perf": perf}[sys.argv[1]]()
