#!/usr/bin/env python3
"""gate_up + SiTU with synthetic weights. Usage: test_gate_up.py gfx1201|gfx1151"""
import sys, os
import numpy as np
from runner import Case
import refs

arch = sys.argv[1] if len(sys.argv) > 1 else "gfx1201"
work = os.path.expanduser(f"~/inkling/optP1/work/gate_up_{arch}")
EXP = 16
fails = 0


def run(wtype, n, seed):
    rng = np.random.default_rng(seed)
    rb = refs.ROW_BYTES[wtype]
    if wtype == 0:
        w = refs.random_nvfp4_rows(rng, EXP * 2304, 2304)
    elif wtype == 1:
        w = rng.integers(0, 256, size=(EXP * 2304, 72, 34), dtype=np.uint8)
        w[..., :2] = np.float16(rng.uniform(0.001, 0.02, size=(EXP * 2304, 72))).view(np.uint8).reshape(-1, 72, 2)
        w = w.reshape(EXP * 2304, rb)
    else:
        f = (rng.standard_normal((EXP * 2304, 2304)) * 0.02).astype(np.float32)
        w = (f.view(np.uint32) >> 16).astype(np.uint16).view(np.uint8).reshape(EXP * 2304, rb)
    scales = np.zeros(128, np.float32)
    scales[:] = rng.uniform(4e-5, 1.7e-4, 128) if wtype == 0 else 1.0
    x = rng.standard_normal((n, 2304)).astype(np.float16)
    ids = rng.integers(0, EXP, size=(n, 8), dtype=np.int32)
    if n >= 4:
        ids[1] = ids[0]
    c = Case("gate_up", os.path.join(work, f"t{wtype}_{n}"))
    c.inp("x", refs.to_slots(x))
    c.inp("ids", ids.reshape(-1))
    c.inp("w", w.reshape(-1))
    c.inp("scales", scales)
    recs = min(n * 8, n + 128)
    c.zeros("tasks", (4 + recs * 66,), np.int32)
    c.zeros("act", (n * 8, 1152), np.float16)
    c.lit("n", n)
    c.lit("wtype", wtype)
    c.lit("tile", 8)
    c.launch("inkling_experts_routes", [], [("n", "index"), ("tile", "index"), ("ids", "ids"), ("tasks", "tasks")],
             ("", "%n: index, %tile: index, %ids: buffer, %tasks: buffer"))
    c.launch("inkling_experts_gate_up", ["n"], [("n", "index"), ("wtype", "index"), ("x", "x"), ("tasks", "tasks"), ("w", "w"), ("scales", "scales"), ("act", "act")],
             ("%n: index", "%n: index, %wtype: index, %x: buffer, %tasks: buffer, %w: buffer, %scales: buffer, %act: buffer"))
    c.out("act")
    c.out("tasks")
    out = c.run(arch, ["routes.loom", "gate_up.loom"])
    act = refs.from_slots(out["act"].astype(np.float32))
    W = refs.decode_rows(wtype, w, 2304).reshape(EXP, 2304, 2304)
    ref = np.zeros((n * 8, 1152))
    xf = x.astype(np.float64)
    for r in range(n * 8):
        e = ids.reshape(-1)[r]
        gu = W[e].astype(np.float64) @ xf[r // 8] * (scales[e] if wtype == 0 else 1.0)
        ref[r] = refs.situ(gu[:1152], gu[1152:])
    err = refs.rel_l2(act, ref)
    mx = float(np.abs(act - ref).max() / max(np.abs(ref).max(), 1e-30))
    return err, mx


for wtype in (0, 1, 2):
    for n in (1, 3, 8, 37):
        err, mx = run(wtype, n, 100 * wtype + n)
        ok = err < 2e-3
        fails += not ok
        print(f"gate_up {arch} wtype={wtype} n={n}: rel_l2={err:.2e} max_rel={mx:.2e} {'PASS' if ok else 'FAIL'}")
sys.exit(1 if fails else 0)
