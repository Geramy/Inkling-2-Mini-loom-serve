"""usage: m30_cmp.py A.json B.json   Compare greedy outputs case by case."""
import json, sys
a, b = (json.load(open(p)) for p in sys.argv[1:3])
for k in a:
    if k not in b:
        continue
    x, y = a[k]["text"], b[k]["text"]
    if x == y:
        print(k, "IDENTICAL", a[k]["tokens"])
    else:
        i = next((i for i in range(min(len(x), len(y))) if x[i] != y[i]), min(len(x), len(y)))
        print(k, "DIFF at char", i, "of", len(x), len(y), repr(x[max(0, i-20):i+30]), "|", repr(y[max(0, i-20):i+30]))
