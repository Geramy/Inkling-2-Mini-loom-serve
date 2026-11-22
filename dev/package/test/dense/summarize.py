"""Summarizes the result files of run_all.sh into a markdown table: cases, failures and the worst error of each test on each GPU.
Usage: python summarize.py [resultsdir]
"""
import glob
import os
import re
import sys

out = sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/inkling/inkF/results")
rows = {}
for path in sorted(glob.glob(os.path.join(out, "*.txt"))):
    arch, name = os.path.basename(path)[:-4].split("_", 1)
    cases = fails = 0
    worst_nmse = worst_oracle = 0.0
    for line in open(path):
        m = re.match(r"(PASS|FAIL)\s", line)
        if not m:
            continue
        cases += 1
        fails += m.group(1) == "FAIL"
        for v in re.findall(r"nmse=([0-9.e+-]+)", line) + re.findall(r"\b[a-z]:([0-9.e+-]+)/", line) + re.findall(r"(?<![a-z_])nmse_exact=([0-9.e+-]+)", line):
            worst_nmse = max(worst_nmse, float(v))
        for v in re.findall(r"oracle_relL2=([0-9.e+-]+)", line) + re.findall(r"/([0-9.e+-]+)(?: |$)", line) + re.findall(r"oracle_nmse=([0-9.e+-]+)", line):
            try:
                worst_oracle = max(worst_oracle, float(v))
            except ValueError:
                pass
    rows.setdefault(name, {})[arch] = (cases, fails, worst_nmse, worst_oracle)
print("| Test | gfx1151 cases / fail | gfx1201 cases / fail | worst nmse vs numpy (1151 / 1201) | worst oracle error (1151 / 1201) |")
print("|---|---|---|---|---|")
for name, d in rows.items():
    a, b = d.get("gfx1151", (0, 0, 0, 0)), d.get("gfx1201", (0, 0, 0, 0))
    print("| %s | %d / %d | %d / %d | %.1e / %.1e | %.1e / %.1e |" % (name, a[0], a[1], b[0], b[1], a[2], b[2], a[3], b[3]))
