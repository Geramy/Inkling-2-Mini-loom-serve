#!/usr/bin/env python3
# usage: peek_capture.py <port> <n_tokens> [width]   prints the first values of the captured rows of a sequence request
import base64, json, sys, urllib.request
import numpy as np
port, n = sys.argv[1], int(sys.argv[2])
width = int(sys.argv[3]) if len(sys.argv) > 3 else 2304
body = {"model": "inkling-2-mini", "position": 0, "token_ids": [1000 + 17 * i for i in range(n)], "options": {}}
req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/sequences", json.dumps(body).encode(), {"Content-Type": "application/json"})
rows, event = [], None
with urllib.request.urlopen(req) as resp:
    for raw in resp:
        line = raw.decode().rstrip("\n")
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:") and event == "chunk":
            rows.append(np.frombuffer(base64.b64decode(json.loads(line[5:])["data"]), dtype=np.float32).reshape(-1, width))
out = np.concatenate(rows)
print(out.shape, out[0, :8])
