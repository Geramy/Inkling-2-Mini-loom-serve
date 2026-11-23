#!/usr/bin/env python3
"""inkling_router_decode (select and routes in one workgroup) against inkling_router_select followed by inkling_experts_routes and against a host reference.
Usage: test_router_decode.py gfx1201|gfx1151. Random logits with many ties, n = 1..8 rows. Ids, weights and the route table must equal the split kernels and the host reference."""
import sys, os
import numpy as np
from runner import Case

arch = sys.argv[1] if len(sys.argv) > 1 else "gfx1201"
work = os.path.expanduser(f"~/inkling/optD3/work/router_decode_{arch}")
fails = 0
TILE = 8


def check(label, ok):
    global fails
    fails += not ok
    print(f"{label}: {'PASS' if ok else 'FAIL'}")


def build(n, logits, bias, fused):
    c = Case("fused" if fused else "split", os.path.join(work, f"{'f' if fused else 's'}{n}"))
    recs = min(n * 8, n + 128)
    c.inp("logits", logits)
    c.inp("bias", bias)
    c.zeros("ids", (n, 8), np.int32)
    c.zeros("wts", (n, 8), np.float32)
    c.zeros("tasks", (4 + recs * 34,), np.int32)
    c.lit("n", n)
    c.lit("tile", TILE)
    if fused:
        c.launch("inkling_router_decode", [], [("n", "index"), ("tile", "index"), ("logits", "logits"), ("bias", "bias"), ("ids", "ids"), ("wts", "wts"), ("tasks", "tasks")],
                 ("", "%n: index, %tile: index, %logits: buffer, %bias: buffer, %ids: buffer, %wts: buffer, %tasks: buffer"))
    else:
        c.launch("inkling_router_select", ["n"], [("n", "index"), ("logits", "logits"), ("bias", "bias"), ("ids", "ids"), ("wts", "wts")],
                 ("%n: index", "%n: index, %logits: buffer, %bias: buffer, %ids: buffer, %wts: buffer"))
        c.launch("inkling_experts_routes", [], [("n", "index"), ("tile", "index"), ("ids", "ids"), ("tasks", "tasks")],
                 ("", "%n: index, %tile: index, %ids: buffer, %tasks: buffer"))
    for o in ("ids", "wts", "tasks"):
        c.out(o)
    return c.run(arch, ["router.loom", "routes.loom"])


def records(t, stride=34):
    count = int(t[0])
    out = []
    for r in range(count):
        base = 4 + r * stride
        e, live = int(t[base]), int(t[base + 1])
        out.append((e, live, tuple(sorted(int(v) for v in t[base + 2:base + 2 + live]))))
    return count, out


def ref_records(ids):
    """n <= 8: one record per used expert (an expert has at most one route per token), ordered by expert index."""
    by = {}
    for t in range(ids.shape[0]):
        for s in range(8):
            by.setdefault(int(ids[t, s]), []).append(t * 8 + s)
    recs = [(e, len(v), tuple(sorted(v))) for e, v in sorted(by.items())]
    return len(recs), recs


rng = np.random.default_rng(7)
for n in (1, 2, 3, 4, 5, 6, 7, 8):
    for kind in ("tied", "random"):
        if kind == "tied":
            logits = np.round(rng.standard_normal((n, 128)) * 2).astype(np.float32) / 2
            bias = np.round(rng.standard_normal(128)).astype(np.float32) / 4
        else:
            logits = (rng.standard_normal((n, 128)) * 3).astype(np.float32)
            bias = (rng.standard_normal(128) * 0.1).astype(np.float32)
        a = build(n, logits, bias, False)
        b = build(n, logits, bias, True)
        p = 1.0 / (1.0 + np.exp(-logits.astype(np.float64)))
        sel = p + bias.astype(np.float64)
        ref_ids = np.array([sorted(range(128), key=lambda e: (-sel[t, e], e))[:8] for t in range(n)], np.int32)
        ref_w = np.take_along_axis(p, ref_ids, axis=1)
        ref_w = ref_w / (ref_w.sum(axis=1, keepdims=True) + 1e-20) * 8.0
        check(f"n {n} {kind} split ids vs numpy (ties to the lower index)", np.array_equal(a["ids"], ref_ids))
        check(f"n {n} {kind} split weights vs numpy", float(np.abs(a["wts"] - ref_w).max()) < 1e-5)
        check(f"n {n} {kind} fused ids and weights equal the split kernels", np.array_equal(a["ids"], b["ids"]) and np.array_equal(a["wts"], b["wts"]))
        check(f"n {n} {kind} fused route table equals the split kernels", records(a["tasks"]) == records(b["tasks"]))
        check(f"n {n} {kind} fused route table vs numpy ids", records(b["tasks"]) == ref_records(ref_ids))
print("FAILS", fails)
sys.exit(1 if fails else 0)
