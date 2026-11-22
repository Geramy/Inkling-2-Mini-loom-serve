#!/usr/bin/env python3
# usage: bench_chat.py <port> [--prefill 128,512,2048] [--decode 0,8000] [--runs 5] [--new 64]   (run on the box under flock -x ~/hrx-pr1/gpu.lock)
# Prefill: wall time of a request with max_tokens 1 for a prompt of about N tokens (the first token is included). Decode: gaps between streamed tokens after a prompt of about N tokens, temperature 0.
import argparse, http.client, json, statistics, time

ap = argparse.ArgumentParser()
ap.add_argument("port", type=int)
ap.add_argument("--prefill", default="128,512,2048")
ap.add_argument("--decode", default="0,8000")
ap.add_argument("--runs", type=int, default=5)
ap.add_argument("--new", type=int, default=64)
a = ap.parse_args()
SENT = "The history of computing is a story of steady abstraction, and every layer hides the one below it. "


def body(n_reps, max_tokens, stream):
    text = (SENT * n_reps + "Summarize the text above in one sentence.") if n_reps else "Hello"
    return json.dumps({"model": "inkling-2-mini", "max_tokens": max_tokens, "temperature": 0, "stream": stream, "messages": [{"role": "user", "content": text}]})


def post(payload):
    c = http.client.HTTPConnection("127.0.0.1", a.port, timeout=900)
    c.request("POST", "/v1/chat/completions", payload, {"Content-Type": "application/json"})
    return c, c.getresponse()


def timed(n_reps, max_tokens):
    t0 = time.perf_counter()
    c, r = post(body(n_reps, max_tokens, False))
    d = json.loads(r.read())
    if "usage" not in d:
        raise SystemExit("request failed: " + json.dumps(d)[:400])
    t = time.perf_counter() - t0
    c.close()
    return t, d["usage"]["prompt_tokens"], d["usage"]["completion_tokens"]


def calibrate(target):
    if target <= 40:
        return 0
    reps = max(1, target // 20)
    for _ in range(4):
        _, p, _ = timed(reps, 1)
        reps = max(1, round(reps * (target - 17) / max(p - 17, 1)))
    return reps


def stream_gaps(n_reps):
    c, r = post(body(n_reps, a.new, True))
    stamps = []
    buf = b""
    while True:
        chunk = r.read1(65536) if hasattr(r, "read1") else r.read(1)
        if not chunk:
            break
        buf += chunk
        while b"\n\n" in buf:
            ev, buf = buf.split(b"\n\n", 1)
            if b'"delta":{"' in ev and b'"role"' not in ev:
                stamps.append(time.perf_counter())
    c.close()
    return stamps


print("prefill (tokens, median ms, tokens/s, min ms):")
for tgt in [int(x) for x in a.prefill.split(",") if x]:
    reps = calibrate(tgt)
    for _ in range(2):
        timed(reps, 1)
    ts = []
    for _ in range(a.runs):
        t, p, _ = timed(reps, 1)
        ts.append(t)
    m = statistics.median(ts)
    print(f"  prompt {p} tokens: {m * 1000:8.1f} ms  {p / m:8.1f} tok/s  min {min(ts) * 1000:8.1f} ms", flush=True)
print("decode (context tokens, median tokens/s over the gaps, runs):")
for tgt in [int(x) for x in a.decode.split(",") if x]:
    reps = calibrate(tgt)
    stream_gaps(reps)
    rates = []
    for _ in range(a.runs):
        st = stream_gaps(reps)
        gaps = [b - a_ for a_, b in zip(st, st[1:])]
        if gaps:
            rates.append(1.0 / statistics.median(gaps))
    _, p, _ = timed(reps, 1)
    if not rates:
        print(f"  context {p} tokens: no streamed tokens")
        continue
    print(f"  context {p} tokens: {statistics.median(rates):7.1f} tok/s (min {min(rates):.1f}, max {max(rates):.1f}, per token {1000 / statistics.median(rates):.2f} ms)", flush=True)
