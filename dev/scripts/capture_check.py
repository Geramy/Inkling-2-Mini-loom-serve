#!/usr/bin/env python3
# usage: capture_check.py <port> <dump_dir> <name> [occurrence] [--model NAME] [--width N] [--tokens FILE]
# Sends the dump's prompt tokens to a sequence deployment (POST /v1/sequences), assembles the captured F32 rows
# and compares them with the oracle tensor <name> (by manifest name and occurrence). Exit 1 on FAIL.
import argparse, base64, json, sys, urllib.request
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("port")
ap.add_argument("dump")
ap.add_argument("name")
ap.add_argument("occurrence", nargs="?", type=int, default=0)
ap.add_argument("--model", default="inkling-2-mini")
ap.add_argument("--width", type=int, default=0)
ap.add_argument("--tokens", default="")
ap.add_argument("--nmse", type=float, default=1e-6)
ap.add_argument("--mul", default="", help="GGUF:TENSOR, multiplies each oracle row by this F32 vector (the oracle stores some norms without their weight)")
ap.add_argument("--chunk", type=int, default=0, help="send the prompt in appends of this many tokens")
a = ap.parse_args()

manifest = json.load(open(f"{a.dump}/manifest.json"))
ids = manifest["tokens"] if not a.tokens else [int(t) for t in open(a.tokens).read().split()]
entry = [t for t in manifest["tensors"] if t["name"] == a.name and t["occurrence"] == a.occurrence][0]
ref = np.fromfile(f"{a.dump}/{entry['file']}", dtype=np.float32).reshape(entry["ne"][::-1])
if a.mul:
    from gguf import GGUFReader
    path, tname = a.mul.rsplit(":", 1)
    wt = [t for t in GGUFReader(path, "r").tensors if t.name == tname][0]
    ref = ref * np.asarray(wt.data).astype(np.float32).reshape(-1)
width = a.width or ref.shape[-1]

def post(body, session=None):
    req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/sequences", json.dumps(body).encode(), {"Content-Type": "application/json"})
    rows, handle, position = [], session, body["position"]
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
                elif event == "chunk":
                    rows.append(np.frombuffer(base64.b64decode(d["data"]), dtype=np.float32).reshape(-1, width))
                elif event == "done":
                    position = d["position"]
    return handle, position, rows

step = a.chunk or len(ids)
handle, position, got = None, 0, []
for begin in range(0, len(ids), step):
    body = {"model": a.model, "position": position, "token_ids": ids[begin:begin + step], "options": {}}
    if handle:
        body["session"] = handle
    handle, position, rows = post(body, handle)
    got += rows
out = np.concatenate(got, axis=0)
if out.shape[0] != len(ids):
    print(f"FAIL {a.name}: got {out.shape[0]} rows for {len(ids)} tokens")
    sys.exit(1)
w = min(width, ref.shape[-1])
out, ref = out[:, :w].astype(np.float64), ref.reshape(len(ids), -1)[:, :w].astype(np.float64)
err = out - ref
nmse = float((err ** 2).sum() / max((ref ** 2).sum(), 1e-30))
cos = float((out * ref).sum() / max(np.linalg.norm(out) * np.linalg.norm(ref), 1e-30))
status = "PASS" if nmse < a.nmse else "FAIL"
print(f"{status} {a.name}#{a.occurrence} tokens={len(ids)} width={w} nmse={nmse:.3e} cos={cos:.8f} maxabs={np.abs(err).max():.3e}")
sys.exit(0 if status == "PASS" else 1)
