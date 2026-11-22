#!/usr/bin/env python3
"""Router and expert kernels against the oracle dumps (prompt A, 16 tokens) and a host reference from the GGUF bytes.
Usage: test_oracle.py gfx1201|gfx1151 [slots...]   (default slots: 0 9 19 27 39 20 21 22 23)
Dumps come from the Q8_K_XL run; the NVFP4 experts are identical in both files, layers 20..23 are BF16 (q8 file)."""
import sys, os
import numpy as np
from runner import Case
import refs, gguf_io as G

arch = sys.argv[1] if len(sys.argv) > 1 else "gfx1201"
slots = [int(s) for s in sys.argv[2:]] or [0, 9, 19, 27, 39, 20, 21, 22, 23]
work = os.path.expanduser(f"~/inkling/optP1/work/oracle_{arch}")
fails = 0
dump = G.Dump("Q8_K_XL_A")
ROWS = [int(r) for r in os.environ["ROWS"].split(",")] if os.environ.get("ROWS") else list(range(16))
N = len(ROWS)
PATHS = os.environ.get("PATHS", "portable,wmma").split(",")


def check(label, err, tol):
    global fails
    ok = err < tol
    fails += not ok
    print(f"{label}: {err:.3e} (tol {tol:g}) {'PASS' if ok else 'FAIL'}")


def router_case(slot):
    L = G.phys(slot)
    xn = dump.get(f"norm-{slot}", dump.last_occ(f"norm-{slot}")).reshape(16, 2304)
    x = (xn * np.asarray(G.reader("q8")[f"blk.{L}.ffn_norm.weight"].data).astype(np.float32)).astype(np.float32)
    x = x[ROWS]
    rw, bias = G.router_weights("q8", L)
    c = Case("router", os.path.join(work, f"router_{slot}"))
    c.inp("x", x)
    c.inp("rw", rw)
    c.inp("bias", bias)
    c.zeros("logits", (N, 128), np.float32)
    c.zeros("x16", (N, 2304), np.float16)
    c.zeros("ids", (N, 8), np.int32)
    c.zeros("wts", (N, 8), np.float32)
    c.lit("n", N)
    c.launch("inkling_router_logits", ["n"], [("n", "index"), ("x", "x"), ("rw", "rw"), ("logits", "logits"), ("x16", "x16")],
             ("%n: index", "%n: index, %x: buffer, %rw: buffer, %logits: buffer, %x16: buffer"))
    c.launch("inkling_router_select", ["n"], [("n", "index"), ("logits", "logits"), ("bias", "bias"), ("ids", "ids"), ("wts", "wts")],
             ("%n: index", "%n: index, %logits: buffer, %bias: buffer, %ids: buffer, %wts: buffer"))
    for o in ("logits", "x16", "ids", "wts"):
        c.out(o)
    out = c.run(arch, ["router.loom"])
    ref_logits = dump.get(f"ffn_moe_logits-{slot}").reshape(16, 128)[ROWS]
    check(f"slot {slot} L{L} router logits rel_l2", refs.rel_l2(out["logits"], ref_logits), 1e-6)
    check(f"slot {slot} L{L} x16 rel_l2", refs.rel_l2(refs.from_slots(out["x16"].astype(np.float32)), x), 1e-3)
    ref_ids = dump.get(f"ffn_moe_topk-{slot}").reshape(16, 8)[ROWS]
    ref_w = dump.get(f"ffn_moe_weights_scaled-{slot}").reshape(16, 8)[ROWS]
    ids, wts = out["ids"], out["wts"]
    rows_bad = sum(set(ids[t]) != set(ref_ids[t]) for t in range(N))
    check(f"slot {slot} L{L} router id rows that differ (of {N})", rows_bad, 1)
    a = np.take_along_axis(wts, np.argsort(ids, axis=1), axis=1)
    b = np.take_along_axis(ref_w, np.argsort(ref_ids, axis=1), axis=1)
    if rows_bad == 0:
        check(f"slot {slot} L{L} router weights rel_l2", refs.rel_l2(a, b), 1e-6)
    return x, ref_ids, ref_w


def expert_case(slot, kind, x, ids, wts, tag, path="portable"):
    L = G.phys(slot)
    wt, gu, dn, sc = G.expert_weights(kind, L)
    uniq = np.unique(ids)
    remap = np.zeros(128, np.int32)
    remap[uniq] = np.arange(len(uniq))
    cids = remap[ids].astype(np.int32)
    gu_c = np.ascontiguousarray(gu[uniq]).reshape(-1)
    dn_c = np.ascontiguousarray(dn[uniq]).reshape(-1)
    sgu = np.zeros(128, np.float32)
    sdn = np.zeros(128, np.float32)
    if sc is not None:
        sgu[: len(uniq)] = sc[0][uniq]
        sdn[: len(uniq)] = sc[1][uniq]
    x16 = refs.to_slots(x.astype(np.float16))
    tile, suf = (8, "") if path == "portable" else (64, "_wmma")
    c = Case("experts", os.path.join(work, f"experts_{tag}_{path}_{slot}_{N}"))
    c.inp("x16", x16)
    c.inp("ids", cids.reshape(-1))
    c.inp("wts", wts.astype(np.float32).reshape(-1))
    c.inp("wgu", gu_c)
    c.inp("wdn", dn_c)
    c.inp("sgu", sgu)
    c.inp("sdn", sdn)
    recs = min(N * 8, N + 128)
    c.zeros("tasks", (4 + recs * 66,), np.int32)
    c.zeros("act", (N * 8, 1152), np.float16)
    c.zeros("dout", (N * 8, 2304), np.float32)
    c.zeros("y", (N, 2304), np.float32)
    c.lit("n", N)
    c.lit("wtype", wt)
    c.lit("tile", tile)
    c.launch("inkling_experts_routes", [], [("n", "index"), ("tile", "index"), ("ids", "ids"), ("tasks", "tasks")],
             ("", "%n: index, %tile: index, %ids: buffer, %tasks: buffer"))
    c.launch("inkling_experts_gate_up" + suf, ["n"], [("n", "index"), ("wtype", "index"), ("x16", "x16"), ("tasks", "tasks"), ("wgu", "wgu"), ("sgu", "sgu"), ("act", "act")],
             ("%n: index", "%n: index, %wtype: index, %x: buffer, %tasks: buffer, %w: buffer, %scales: buffer, %act: buffer"))
    c.launch("inkling_experts_down" + suf, ["n"], [("n", "index"), ("wtype", "index"), ("act", "act"), ("tasks", "tasks"), ("wdn", "wdn"), ("sdn", "sdn"), ("dout", "dout")],
             ("%n: index", "%n: index, %wtype: index, %act: buffer, %tasks: buffer, %w: buffer, %scales: buffer, %dout: buffer"))
    c.launch("inkling_experts_combine", ["n"], [("n", "index"), ("dout", "dout"), ("wts", "wts"), ("y", "y")],
             ("%n: index", "%n: index, %dout: buffer, %wts: buffer, %y: buffer"))
    for o in ("act", "dout", "y"):
        c.out(o)
    out = c.run(arch, ["routes.loom", "combine.loom"] + (["gate_up.loom", "down.loom"] if path == "portable" else ["gate_up_wmma.loom", "down_wmma.loom"]))
    # host reference, f32 activations
    act_ref = np.zeros((N * 8, 1152))
    dout_ref = np.zeros((N * 8, 2304))
    flat = ids.reshape(-1)
    for e in uniq:
        Wg = refs.decode_rows(wt, gu[e], 2304).astype(np.float64)
        Wd = refs.decode_rows(wt, dn[e], 1152).astype(np.float64)
        s_gu = sc[0][e] if sc else 1.0
        s_dn = sc[1][e] if sc else 1.0
        for r in np.nonzero(flat == e)[0]:
            g = Wg @ x[r // 8].astype(np.float64) * s_gu
            act_ref[r] = refs.situ(g[:1152], g[1152:])
            dout_ref[r] = Wd @ act_ref[r] * s_dn
    y_ref = (dout_ref.reshape(N, 8, 2304) * wts.reshape(N, 8, 1)).sum(1)
    label = f"slot {slot} L{L} n={N} wtype {wt} {tag} {path}"
    check(f"{label} act vs host (f32 x)", refs.rel_l2(refs.from_slots(out["act"].astype(np.float32)), act_ref), 3e-3)
    check(f"{label} dout vs host", refs.rel_l2(out["dout"], dout_ref), 3e-3)
    check(f"{label} y vs host", refs.rel_l2(out["y"], y_ref), 3e-3)
    try:
        y_or = dump.get(f"conv_in-{slot}", 3).reshape(16, 2304)[ROWS]
        check(f"{label} y vs oracle conv_in(ffn)", refs.rel_l2(out["y"], y_or), 2e-2)
    except KeyError:
        pass


for slot in slots:
    x, ids, wts = router_case(slot)
    L = G.phys(slot)
    for path in PATHS:
        if L >= 20:
            expert_case(slot, "q8", x, ids, wts, "bf16-q8xl", path)
            expert_case(slot, "q4", x, ids, wts, "q8_0-q4xl-hostref", path)
        else:
            expert_case(slot, "q4", x, ids, wts, "nvfp4", path)
print("FAILS", fails)
sys.exit(1 if fails else 0)
