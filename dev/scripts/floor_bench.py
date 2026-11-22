#!/usr/bin/env python3
# usage: floor_bench.py <port> [n_tokens=1] [repeats=40]
# Sequential known-token requests against a sequence deployment. Prints the first (cold) request and the median and minimum of the warm ones, in ms.
import json, statistics, sys, time, urllib.request
port = sys.argv[1]
n = int(sys.argv[2]) if len(sys.argv) > 2 else 1
repeats = int(sys.argv[3]) if len(sys.argv) > 3 else 40
body = json.dumps({"model": "inkling-2-mini", "position": 0, "token_ids": [1000 + 17 * i for i in range(n)], "options": {}}).encode()
def once():
    t = time.perf_counter()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/sequences", body, {"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        resp.read()
    return (time.perf_counter() - t) * 1e3
cold = once()
warm = [once() for _ in range(repeats)]
print(f"tokens={n} cold={cold:.1f} ms warm median={statistics.median(warm):.2f} min={min(warm):.2f} max={max(warm):.2f} ms")
