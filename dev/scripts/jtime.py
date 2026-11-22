#!/usr/bin/env python3
# usage: jtime.py <port> <dump_dir> [runs] [--tokens N]   times POST /v1/sequences for the prompt of the dump (new session each run); prints min and median in ms
import json, sys, time, urllib.request

port, dump = sys.argv[1], sys.argv[2]
runs = int(sys.argv[3]) if len(sys.argv) > 3 else 8
ids = json.load(open(f"{dump}/manifest.json"))["tokens"]
times = []
for i in range(runs + 2):
    body = {"model": "inkling-2-mini", "position": 0, "token_ids": ids, "options": {}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/sequences", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req) as resp:
        resp.read()
    if i >= 2:
        times.append((time.perf_counter() - t0) * 1000)
times.sort()
print(f"tokens {len(ids)} runs {runs} min {times[0]:.1f} ms median {times[len(times)//2]:.1f} ms")
