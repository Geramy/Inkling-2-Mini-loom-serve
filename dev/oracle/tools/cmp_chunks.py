#!/usr/bin/env python3
"""Compare the oracle run in chunks (c0, c1, ...) with the single chunk run for the given tensor names.

usage: cmp_chunks.py <single dump dir> <chunked dump dir> <name> [<name> ...]   (names like kqv_out-0)
"""
import json, sys
import numpy as np


def load(d, name):
    m = json.load(open(f"{d}/manifest.json"))
    t = next(t for t in m["tensors"] if t["name"] == name and t["occurrence"] == 0)
    return np.fromfile(f"{d}/{t['file']}", dtype=np.float32).reshape(t["ne"][::-1])


one, many = sys.argv[1], sys.argv[2]
for name in sys.argv[3:]:
    a = load(one, name).reshape(-1, 3072)
    parts, c = [], 0
    while True:
        try:
            parts.append(load(f"{many}/c{c}", name).reshape(-1, 3072))
        except Exception:
            break
        c += 1
    b = np.concatenate(parts)
    e = ((a - b) ** 2).sum() / (a ** 2).sum()
    print(f"{name}: chunks {c} tokens {len(b)} NMSE {e:.3e} max abs {np.abs(a - b).max():.3e}")
