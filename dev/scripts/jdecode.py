#!/usr/bin/env python3
# usage: jdecode.py <port> <dump_dir> [steps] [--ctx N]   times single token appends (decode steps) to a sequence session after a prompt of N tokens (the dump prompt repeated); prints min and median in ms
import base64, json, sys, time, urllib.request

port, dump = sys.argv[1], sys.argv[2]
steps = int(sys.argv[3]) if len(sys.argv) > 3 and not sys.argv[3].startswith("--") else 20
ctx = int(sys.argv[sys.argv.index("--ctx") + 1]) if "--ctx" in sys.argv else 0
ids = json.load(open(f"{dump}/manifest.json"))["tokens"]
prompt = (ids * (ctx // len(ids) + 1))[:ctx] if ctx else ids


def post(handle, position, tokens):
    body = {"model": "inkling-2-mini", "position": position, "token_ids": tokens, "options": {}}
    if handle:
        body["session"] = handle
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/sequences", json.dumps(body).encode(), {"Content-Type": "application/json"})
    event = None
    with urllib.request.urlopen(req) as resp:
        for raw in resp:
            line = raw.decode().rstrip("\n")
            if line.startswith("event:"):
                event = line[6:].strip()
            elif line.startswith("data:"):
                d = json.loads(line[5:])
                if event == "attached":
                    handle = d["session"]
                elif event == "done":
                    position = d["position"]
    return handle, position


handle, position = None, 0
for i in range(0, len(prompt), 512):
    handle, position = post(handle, position, prompt[i:i + 512])
times = []
for i in range(steps + 3):
    t0 = time.perf_counter()
    handle, position = post(handle, position, [1000 + i])
    if i >= 3:
        times.append((time.perf_counter() - t0) * 1000)
times.sort()
print(f"context {len(prompt)} steps {steps} min {times[0]:.2f} ms median {times[len(times) // 2]:.2f} ms")
