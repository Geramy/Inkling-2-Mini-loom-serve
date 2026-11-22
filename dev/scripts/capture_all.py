#!/usr/bin/env python3
# usage: capture_all.py <port> <dump_dir> <visit> --gguf GGUF [--chunk N | --chunks a,b,c] [--tokens FILE] [--only name,name] [--tol 1e-4] [--model NAME]
# Sends the dump's prompt to a sequence deployment started with capture id 99 at <visit> (INKLING_CAPTURE=99,<visit>,0,<total width of dispatch/capture_layout.json>)
# and compares every captured tensor of the visit with the oracle (cache slot = visit). Visits 0..47 (visits >= 24 are controlled loop visits: the norm occurrences are k, v, q, ffn).
import argparse, base64, json, os, sys, urllib.request
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("port")
ap.add_argument("dump")
ap.add_argument("visit", type=int)
ap.add_argument("--gguf", required=True)
ap.add_argument("--chunk", type=int, default=0)
ap.add_argument("--chunks", default="")
ap.add_argument("--tokens", default="")
ap.add_argument("--only", default="")
ap.add_argument("--tol", type=float, default=2e-4)
ap.add_argument("--model", default="inkling-2-mini")
ap.add_argument("--layout", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "package", "dispatch", "capture_layout.json"))
ap.add_argument("--save", default="")
a = ap.parse_args()

layout = json.load(open(a.layout))
total = layout["total"]
manifest = json.load(open(f"{a.dump}/manifest.json"))
ids = manifest["tokens"] if not a.tokens else [int(t) for t in open(a.tokens).read().split()]
n = len(ids)
slot = a.visit
layer = slot if slot < 24 else 6 + (slot - 24) % 12
glob = layer % 6 == 5
ctl = slot >= 24

from gguf import GGUFReader
reader = GGUFReader(a.gguf, "r")
tensors = {t.name: t for t in reader.tensors}


def gw(name):
    return np.asarray(tensors[name].data).astype(np.float32).reshape(-1)


def gw16(name):
    u = np.asarray(tensors[name].data).view(np.uint16).astype(np.uint32) << 16
    return u.view(np.float32)


def oracle_rows(name, occ=0, ne_axis=None):
    for t in manifest["tensors"]:
        if t["layer"] == slot and t["name"] == f"{name}-{slot}" and t["occurrence"] == occ:
            arr = np.fromfile(f"{a.dump}/{t['file']}", dtype=np.float32).reshape(t["ne"][::-1])
            return arr
    return None


def refs():
    r = {}
    mix = oracle_rows("mix_out", 0)
    if mix is not None and not ctl:
        r["x"] = mix.reshape(n, -1, 2304)[:, 0, :]
    nrm = oracle_rows("norm", 2 if (glob or ctl) else 0)
    if nrm is not None:
        r["xn"] = nrm.reshape(n, -1) * gw(f"blk.{layer}.{'attn_q_inp_norm' if glob else 'attn_norm'}.weight")
    for key, name, occ in [("q_raw", "proj_q", 0), ("k_raw", "conv_in", 0), ("v_raw", "conv_in", 1), ("r_raw", "proj_r", 0), ("q", "Qcur", 0), ("k", "Kcur", 0), ("v", "Vcur", 0), ("r", "Rcur", 0), ("ab", "kqv_out", 0),
                           ("ob", "conv_in", 2), ("ffn_inp", "ffn_inp", 0), ("logits", "ffn_moe_logits", 0), ("y", "conv_in", 3), ("l_out", "l_out", 0)]:
        t = oracle_rows(name, occ)
        if t is not None:
            r[key] = t.reshape(n, -1)
    if "xn" in r and os.environ.get("NOGATE") is None:
        wg = gw16(f"blk.{layer}.attn_gate.weight").reshape(3072, 2304)
        r["gate"] = r["xn"].astype(np.float64) @ wg.astype(np.float64).T
        ws = gw16(f"blk.{layer}.attn_sink_proj.weight").reshape(24, 2304)
        r["sink"] = r["xn"].astype(np.float64) @ ws.astype(np.float64).T
    nf = oracle_rows("norm", 3 if (glob or ctl) else 1)
    if nf is not None:
        r["xn_ffn"] = nf.reshape(n, -1) * gw(f"blk.{layer}.ffn_norm.weight")
    return r


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
                    rows.append(np.frombuffer(base64.b64decode(d["data"]), dtype=np.float32).reshape(-1, total))
                elif event == "done":
                    position = d["position"]
    return handle, position, rows


if a.chunks:
    cuts = [int(x) for x in a.chunks.split(",")]
elif a.chunk:
    cuts = [a.chunk] * (n // a.chunk) + ([n % a.chunk] if n % a.chunk else [])
else:
    cuts = [n]
handle, position, got, begin = None, 0, [], 0
for c in cuts:
    body = {"model": a.model, "position": position, "token_ids": ids[begin:begin + c], "options": {}}
    if handle:
        body["session"] = handle
    handle, position, rows = post(body)
    got += rows
    begin += c
out = np.concatenate(got, axis=0)
if out.shape[0] != n:
    print(f"FAIL: got {out.shape[0]} rows for {n} tokens")
    sys.exit(1)
if a.save:
    np.save(a.save, out)
ref = refs()
gate_it = [i for i in layout["items"] if i["name"] == "gate"][0]
only = set(a.only.split(",")) if a.only else None
worst, bad = 0.0, []
print(f"visit {slot} layer {layer} {'global' if glob else 'local'} tokens {n} chunks {cuts}")
for it in layout["items"]:
    name = it["name"]
    if name not in ref or (only and name not in only) or (name == "x" and (slot == 0 or (slot < 24 and not glob))):
        continue
    got_t = out[:, it["col"]:it["col"] + it["width"]].astype(np.float64)
    want = ref[name].astype(np.float64)
    if name == "ab":
        gate_got = out[:, gate_it["col"]:gate_it["col"] + gate_it["width"]].astype(np.float64)
        want = want / (1.0 + np.exp(-gate_got))
    if want.shape[1] != it["width"]:
        print(f"{name:8s} shape mismatch oracle {want.shape} width {it['width']}")
        continue
    err = got_t - want
    nmse = float((err ** 2).sum() / max((want ** 2).sum(), 1e-30))
    cos = float((got_t * want).sum() / max(np.linalg.norm(got_t) * np.linalg.norm(want), 1e-30))
    first = int(np.argmax(np.abs(err).max(axis=1) > 1e-3 * max(np.abs(want).max(), 1e-30))) if (np.abs(err).max(axis=1) > 1e-3 * max(np.abs(want).max(), 1e-30)).any() else -1
    flag = "ok" if nmse < a.tol * {"y": 5.0, "l_out": 2.0}.get(name, 1.0) else "BAD"
    if flag == "BAD":
        bad.append(name)
    print(f"{name:8s} nmse {nmse:9.3e} cos {cos:.8f} maxabs {np.abs(err).max():9.3e} first_bad_token {first:3d} {flag}")
print("RESULT", "FAIL " + ",".join(bad) if bad else "PASS")
sys.exit(1 if bad else 0)
