#!/usr/bin/env python3
# usage: mulmat_check.py <port> <gguf> [n_tokens=8] [--model NAME]
# Capture id 100 of a sequence deployment: PR1 Q8_0 mul_mat on blk.0.attn_q applied to the normalized embedding.
# The reference is computed with numpy from the same GGUF (Q8_0 weights, f64 accumulation).
import argparse, base64, json, sys, urllib.request
import numpy as np
from gguf import GGUFReader, GGMLQuantizationType
from gguf.quants import dequantize

ap = argparse.ArgumentParser()
ap.add_argument("port")
ap.add_argument("gguf")
ap.add_argument("n", nargs="?", type=int, default=8)
ap.add_argument("--model", default="inkling-2-mini")
ap.add_argument("--ids", default="")
ap.add_argument("--tol", type=float, default=1e-5)
a = ap.parse_args()

ids = [int(t) for t in a.ids.split(",")] if a.ids else [1000 + 17 * i for i in range(a.n)]
body = {"model": a.model, "position": 0, "token_ids": ids, "options": {}}
req = urllib.request.Request(f"http://127.0.0.1:{a.port}/v1/sequences", json.dumps(body).encode(), {"Content-Type": "application/json"})
rows, event = [], None
with urllib.request.urlopen(req) as resp:
    for raw in resp:
        line = raw.decode().rstrip("\n")
        if line.startswith("event:"):
            event = line[6:].strip()
        elif line.startswith("data:") and event == "chunk":
            rows.append(np.frombuffer(base64.b64decode(json.loads(line[5:])["data"]), dtype=np.float32).reshape(-1, 3072))
got = np.concatenate(rows, axis=0)

r = GGUFReader(a.gguf, "r")
T = {t.name: t for t in r.tensors}
emb = T["token_embd.weight"]
e = dequantize(np.asarray(emb.data)[ids], emb.tensor_type).astype(np.float64)
w = np.asarray(T["token_embd_norm.weight"].data).astype(np.float64).reshape(-1)
x0 = e / np.sqrt((e ** 2).mean(-1, keepdims=True) + 1e-6) * w
q = T["blk.0.attn_q.weight"]
wq = dequantize(np.asarray(q.data), q.tensor_type).astype(np.float64)
want = x0 @ wq.T
err = got.astype(np.float64) - want
nmse = float((err ** 2).sum() / (want ** 2).sum())
print("got", got[0,:5], "want", want[0,:5], "norm got", np.linalg.norm(got), "want", np.linalg.norm(want))
status = "PASS" if got.shape == want.shape and nmse < a.tol else "FAIL"
print(f"{status} mul_mat Q8_0 blk.0.attn_q tokens={len(ids)} shape={got.shape} nmse={nmse:.3e} maxabs={np.abs(err).max():.3e}")
sys.exit(0 if status == "PASS" else 1)
