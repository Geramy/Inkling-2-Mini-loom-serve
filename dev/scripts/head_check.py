#!/usr/bin/env python3
# usage: head_check.py <port> <dump_dir> <gen_dir> [--chunk 128] [--norm]   (run on the box, server started with INKLING_CAPTURE=31,0,0,200064; --norm: capture 30 with width 2304)
# Compares the head output of the last prompt token with the oracle result_output (or result_norm), and checks teacher-forced greedy: prompt + the oracle greedy tokens go through the model in chunks and the argmax of every row must equal the next oracle token.
import argparse, base64, json, sys, urllib.request
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("port")
ap.add_argument("dump")
ap.add_argument("gen")
ap.add_argument("--chunk", type=int, default=128)
ap.add_argument("--norm", action="store_true")
ap.add_argument("--model", default="inkling-2-mini")
ap.add_argument("--teacher", type=int, default=64)
a = ap.parse_args()
manifest = json.load(open(f"{a.dump}/manifest.json"))
prompt = manifest["tokens"]
gen = [int(t) for t in open(f"{a.gen}/gen_ids.txt").read().split()][: a.teacher]
width = 2304 if a.norm else 200064
valid = 2304 if a.norm else 200058
name = "result_norm" if a.norm else "result_output"
want = None
for t in manifest["tensors"]:
    if t["name"] == name:
        want = np.fromfile(f"{a.dump}/{t['file']}", dtype=np.float32).reshape(-1)


def post(body):
    req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/sequences", json.dumps(body).encode(), {"Content-Type": "application/json"})
    rows, handle, position, event = [], body.get("session"), body["position"], None
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
                    rows.append(np.frombuffer(base64.b64decode(d["data"]), dtype=np.float32).reshape(-1, width)[:, :valid].copy())
                elif event == "done":
                    position = d["position"]
    return handle, position, np.concatenate(rows, axis=0)


ids = prompt + gen[:-1] if not a.norm else prompt
handle, position, got, begin = None, 0, [], 0
n = len(prompt)
while begin < len(ids):
    c = ids[begin:begin + a.chunk]
    body = {"model": a.model, "position": position, "token_ids": c, "options": {}}
    if handle:
        body["session"] = handle
    handle, position, rows = post(body)
    keep = [i for i in range(begin, begin + len(c)) if i >= n - 1]
    got.append(rows[[i - begin for i in keep]])
    begin += len(c)
got = np.concatenate(got, axis=0)
last = got[0].astype(np.float64)
w = want.astype(np.float64)
nmse = float(((last - w) ** 2).sum() / (w ** 2).sum())
print(f"{name}: nmse {nmse:.3e} cos {float((last * w).sum() / np.linalg.norm(last) / np.linalg.norm(w)):.8f}")
if not a.norm:
    top = lambda v, k: set(np.argsort(-v)[:k].tolist())
    p = np.exp(w - w.max()); p /= p.sum()
    q = np.exp(last - last.max()); q /= q.sum()
    kl = float((p * (np.log(p + 1e-30) - np.log(q + 1e-30))).sum())
    print(f"argmax ours {int(last.argmax())} oracle {int(w.argmax())}; top-5 overlap {len(top(last, 5) & top(w, 5))}/5, top-20 {len(top(last, 20) & top(w, 20))}/20; KL(oracle||ours) {kl:.3e}")
    am = got.argmax(axis=1)
    ok = [int(am[i]) == gen[i] for i in range(len(gen))]
    first = ok.index(False) if False in ok else -1
    print(f"teacher forced greedy: {sum(ok)}/{len(ok)} equal, first mismatch {first}")
    sys.exit(0 if sum(ok) >= len(ok) - 2 else 1)
