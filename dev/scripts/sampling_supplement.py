#!/usr/bin/env python3
"""Sampling checks that hold for a server whose default temperature is 1.0 (sampling_check.py compares a request without sampling fields to greedy).
usage: sampling_supplement.py PKG URL   (K = skills scripts directory in the environment)"""
import os
import sys

sys.path.insert(0, os.environ["K"])
import requests
from bench_common import Package

pkg, url = Package(sys.argv[1]), sys.argv[2]
msgs = [{"role": "user", "content": "Pick a random fruit: apple, banana, cherry or grape. Answer with just the fruit name in lowercase."}]


def ask(**kw):
    r = requests.post(url + "/v1/chat/completions", timeout=600, json={"model": pkg.model_id, "messages": msgs, "max_tokens": 16, **kw}).json()
    m = r["choices"][0]["message"]
    return (m.get("reasoning_content") or "") + (m.get("content") or "")


g = [ask(temperature=0) for _ in range(3)]
default = {ask() for _ in range(6)}
checks = [("temperature 0 repeats (3 runs equal)", len(set(g)) == 1),
          ("top_k 1 at temperature 1 equals temperature 0", ask(temperature=1.0, top_k=1, seed=7) == g[0]),
          ("top_p 1e-6 at temperature 1 equals temperature 0", ask(temperature=1.0, top_p=1e-6, seed=3) == g[0]),
          ("same seed reproduces", ask(temperature=1.0, seed=1234) == ask(temperature=1.0, seed=1234))]
bad = 0
for n, ok in checks:
    bad += not ok
    print("OK  " if ok else "FAIL", n)
print(f"INFO a request without sampling fields gave {len(default)} distinct of 6 answers (default temperature is not 0)")
sys.exit(1 if bad else 0)
