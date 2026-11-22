#!/usr/bin/env python3
"""Expert kernels at prefill sizes with synthetic weights: portable (tile 8) and matrix-operation (tile 32) paths against a host reference.
Usage: test_prefill.py gfx1201|gfx1151 [n ...]"""
import sys, os
import numpy as np
from runner import Case
import refs

arch = sys.argv[1] if len(sys.argv) > 1 else "gfx1201"
ns = [int(a) for a in sys.argv[2:]] or [37, 308, 512]
work = os.path.expanduser(f"~/inkling/optP1/work/prefill_{arch}")
fails = 0


def make_weights(rng, wtype, E, k_rows, k):
    rb = {0: (k // 64) * 36, 1: (k // 32) * 34, 2: k * 2}[wtype]
    rows = E * k_rows
    if wtype == 0:
        return refs.random_nvfp4_rows(rng, rows, k)
    if wtype == 1:
        w = rng.integers(0, 256, size=(rows, k // 32, 34), dtype=np.uint8)
        w[..., :2] = np.float16(rng.uniform(0.001, 0.02, size=(rows, k // 32))).view(np.uint8).reshape(rows, k // 32, 2)
        return w.reshape(rows, rb)
    f = (rng.standard_normal((rows, k)) * 0.02).astype(np.float32)
    return (f.view(np.uint32) >> 16).astype(np.uint16).view(np.uint8).reshape(rows, rb)


def run(path, wtype, n, E, seed):
    rng = np.random.default_rng(seed)
    wgu = make_weights(rng, wtype, E, 2304, 2304)
    wdn = make_weights(rng, wtype, E, 2304, 1152)
    sgu = np.zeros(128, np.float32)
    sdn = np.zeros(128, np.float32)
    if wtype == 0:
        sgu[:] = rng.uniform(4e-5, 1.7e-4, 128)
        sdn[:] = rng.uniform(4e-5, 1.7e-4, 128)
    x = rng.standard_normal((n, 2304)).astype(np.float16)
    ids = rng.integers(0, E, size=(n, 8), dtype=np.int32)
    wts = rng.uniform(0.1, 2.0, size=(n, 8)).astype(np.float32)
    tile = 8 if path == "portable" else 64
    suffix = "" if path == "portable" else "_wmma"
    c = Case("prefill", os.path.join(work, f"{path}_{wtype}_{n}"))
    c.inp("x16", refs.to_slots(x))
    c.inp("ids", ids.reshape(-1))
    c.inp("wts", wts.reshape(-1))
    c.inp("wgu", wgu.reshape(-1))
    c.inp("wdn", wdn.reshape(-1))
    c.inp("sgu", sgu)
    c.inp("sdn", sdn)
    recs = min(n * 8, n + 128)
    c.zeros("tasks", (4 + recs * 66,), np.int32)
    c.zeros("act", (n * 8, 1152), np.float16)
    c.zeros("dout", (n * 8, 2304), np.float32)
    c.zeros("y", (n, 2304), np.float32)
    c.lit("n", n)
    c.lit("wtype", wtype)
    c.lit("tile", tile)
    c.launch("inkling_experts_routes", [], [("n", "index"), ("tile", "index"), ("ids", "ids"), ("tasks", "tasks")], ("", "%n: index, %tile: index, %ids: buffer, %tasks: buffer"))
    c.launch("inkling_experts_gate_up" + suffix, ["n"], [("n", "index"), ("wtype", "index"), ("x16", "x16"), ("tasks", "tasks"), ("wgu", "wgu"), ("sgu", "sgu"), ("act", "act")],
             ("%n: index", "%n: index, %wtype: index, %x: buffer, %tasks: buffer, %w: buffer, %scales: buffer, %act: buffer"))
    c.launch("inkling_experts_down" + suffix, ["n"], [("n", "index"), ("wtype", "index"), ("act", "act"), ("tasks", "tasks"), ("wdn", "wdn"), ("sdn", "sdn"), ("dout", "dout")],
             ("%n: index", "%n: index, %wtype: index, %act: buffer, %tasks: buffer, %w: buffer, %scales: buffer, %dout: buffer"))
    c.launch("inkling_experts_combine", ["n"], [("n", "index"), ("dout", "dout"), ("wts", "wts"), ("y", "y")], ("%n: index", "%n: index, %dout: buffer, %wts: buffer, %y: buffer"))
    for o in ("act", "dout", "y"):
        c.out(o)
    kern = ["routes.loom", "combine.loom"] + (["gate_up.loom", "down.loom"] if path == "portable" else ["gate_up_wmma.loom", "down_wmma.loom"])
    out = c.run(arch, kern)
    flat = ids.reshape(-1)
    act_ref = np.zeros((n * 8, 1152), np.float32)
    dout_ref = np.zeros((n * 8, 2304), np.float32)
    xf = x.astype(np.float32)
    act_k = refs.from_slots(out["act"].astype(np.float32))
    for e in np.unique(flat):
        Wg = refs.decode_rows(wtype, wgu.reshape(E, 2304, -1)[e], 2304).astype(np.float32)
        Wd = refs.decode_rows(wtype, wdn.reshape(E, 2304, -1)[e], 1152).astype(np.float32)
        rs = np.nonzero(flat == e)[0]
        g = (xf[rs // 8] @ Wg.T) * (sgu[e] if wtype == 0 else 1.0)
        a = refs.situ(g[:, :1152].astype(np.float64), g[:, 1152:].astype(np.float64)).astype(np.float32)
        act_ref[rs] = a
        # down input: the kernel's own act (f16) so the check isolates the down GEMM
        dout_ref[rs] = (act_k[rs] @ Wd.T) * (sdn[e] if wtype == 0 else 1.0)
    y_ref = (dout_ref.reshape(n, 8, 2304) * wts.reshape(n, 8, 1)).sum(1)
    return refs.rel_l2(act_k, act_ref), refs.rel_l2(out["dout"], dout_ref), refs.rel_l2(out["y"], y_ref)


for path in ("portable", "wmma"):
    for wtype in (0, 1, 2):
        for n in ns:
            if wtype != 0 and n > 64:
                continue
            E = 128 if (wtype == 0 and n >= 300) else 16
            a, d, y = run(path, wtype, n, E, 7 * n + wtype)
            ok = a < 3e-3 and d < 3e-3 and y < 3e-3
            fails += not ok
            print(f"prefill {arch} {path} wtype={wtype} n={n} E={E}: act={a:.2e} dout={d:.2e} y={y:.2e} {'PASS' if ok else 'FAIL'}", flush=True)
print("FAILS", fails)
sys.exit(1 if fails else 0)
