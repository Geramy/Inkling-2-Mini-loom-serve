#!/usr/bin/env python3
# usage: bench_first.py <port>   first and second request latency per prompt size on a fresh server (JIT of the shape buckets on first use)
import http.client, json, sys, time

port = int(sys.argv[1])
S = "The history of computing is a story of steady abstraction, and every layer hides the one below it. "
for reps in (0, 6, 25, 100):
    row = []
    for _ in range(2):
        c = http.client.HTTPConnection("127.0.0.1", port, timeout=900)
        text = S * reps if reps else "Hello"
        t0 = time.perf_counter()
        c.request("POST", "/v1/chat/completions", json.dumps({"model": "inkling-2-mini", "max_tokens": 2, "temperature": 0, "messages": [{"role": "user", "content": text}]}), {"Content-Type": "application/json"})
        d = json.loads(c.getresponse().read())
        row.append((time.perf_counter() - t0, d["usage"]["prompt_tokens"]))
    print(f"prompt {row[0][1]} tokens: first {row[0][0]:.2f} s, second {row[1][0]:.3f} s")
