"""Tests of the loop message mix and controller kernels against the oracle (prompt A, slots 24..47). Usage: test_ctrl.py [gfx1201|gfx1151] [chunk]"""
import sys

import numpy as np

from harness import nmse, run_steps
from inkg import *
from test_mix import SENT, check, spans_buf

rows = []
TOKENS = 1
SRC = ["mix_states.loom", "conv_res.loom", "ctrl.loom"]


def weights_raw(g, ip, c):
    n = "blk.%d.loop_%s_ctrl_" % (ip, CTRL[c])
    out = []
    for k in ("down", "forget", "gain"):
        r = g.raw(n + k + ".weight")
        out.append(np.ascontiguousarray(r).reshape(-1).view(np.int16) if r.dtype != np.float32 else np.ascontiguousarray(r).reshape(-1))
    return out


def run_ctrl(device, kind, chunk, tokens, x, m, rx, rm, wsets, nw, nc, x_col0, x_col_step, m_col0, t, cap, bf16):
    """x [ncx][cap][E], m [4][cap][E], rx [ncx][cap], rm [4][cap]; wsets: list of 3 (down, forget, gain) raw arrays."""
    ws = list(wsets) + [wsets[-1]] * (3 - len(wsets))
    parts = np.full((3, cap, 4608 // chunk, 64), SENT, np.float32)
    zbuf = np.full((3, cap, 64), SENT, np.float32)
    nxc = x.shape[0]
    out = np.full((3, cap, E), SENT, np.float32)
    nwb = np.zeros((3, E), np.float32)
    nwb[:] = nw
    bufs = {"spans": spans_buf(t), "x": x, "m": m, "rx": rx, "rm": rm, "d0": ws[0][0], "d1": ws[1][0], "d2": ws[2][0],
            "f0": ws[0][1], "f1": ws[1][1], "f2": ws[2][1], "g0": ws[0][2], "g1": ws[1][2], "g2": ws[2][2],
            "part": parts, "z": zbuf, "out": out, "w": nwb, "nout": np.full((3, cap, E), SENT, np.float32)}
    cs = cap * E
    steps = [
        ("inkling_ctrl_down", (cap, nc), [x_col0, x_col_step, m_col0, cs], ["spans", "x", "m", "d0", "d1", "d2", "part"]),
        ("inkling_ctrl_z", (cap, nc), [x_col0, x_col_step, m_col0], ["spans", "rx", "rm", "part", "z"]),
        ("inkling_ctrl_apply", (cap, nc), [x_col0, x_col_step, m_col0, cs], ["spans", "x", "m", "f0", "f1", "f2", "g0", "g1", "g2", "z", "out"]),
        ("inkling_mix_norm", (cap, nc), [cs, E if kind == "global" else 0], ["spans", "out", "w", "nout"]),
    ]
    r = run_steps(SRC, bufs, steps, device, {"inkling.mix.ctrl_chunk": chunk, "inkling.mix.ctrl_tokens": tokens, "inkling.mix.ctrl_bf16": int(bf16)})
    return r["out"], r["nout"]


def main(device, chunk):
    d = Dump("Q8_K_XL_A")
    g = Gguf("Q8")
    t = d.n
    cap = t + 3
    for slot in range(24, 48):
        ip = phys(slot)
        glob = ip % 6 == 5
        prev = 17 if slot == 24 else (35 if slot == 36 else slot - 1)
        msgs = d.get("mix_out-%d" % prev, 2).reshape(t, 4, E)
        m = np.zeros((4, cap, E), np.float32)
        m[:, :t] = msgs.transpose(1, 0, 2)
        rm = np.zeros((4, cap), np.float32)
        rm[:, :t] = rms_inv(m[:, :t])[..., 0]
        # attention q, k, v
        q = g.f32("blk.%d.attn_res_query.weight" % ip)
        xm, _ = mix_ref(blocks_at(d, slot, False) + [slot_input(d, slot)], q)
        ncx = 3 if glob else 1
        x = np.zeros((ncx, cap, E), np.float32)
        x[:, :t] = xm.astype(np.float32)
        rx = np.zeros((ncx, cap), np.float32)
        rx[:, :t] = rms_inv(x[:, :t])[..., 0]
        wsets = [weights_raw(g, ip, c) for c in range(3)]
        out, nout = run_ctrl(device, "global" if glob else "local", chunk, TOKENS, x, m, rx, rm, wsets, np.ones(E, np.float32), 3, 0, 1 if glob else 0, 0, t, cap, True)
        for c in range(3):
            wd, wf, wg = ctrl_weights(g, ip, c)
            exp = ctrl_ref(x[c if glob else 0, :t], msgs[:, c], wd, wf, wg)
            check("slot %d attn ctrl %s vs numpy" % (slot, "qkv"[c]), out[c, :t], exp, 1e-11, rows)
            errs = [nmse(nout[c, :t], d.get("norm-%d" % slot, o)) for o in range(3)]
            e = min(errs)
            rows.append(("slot %d attn ctrl %s norm vs oracle" % (slot, "qkv"[c]), e, e < 1e-4))
            print("%-44s nmse %.3e (oracle occ %d) %s" % ("slot %d attn %s norm vs oracle" % (slot, "qkv"[c]), e, errs.index(e), "PASS" if e < 1e-4 else "FAIL"), flush=True)
        check("slot %d guard" % slot, out[:, t:], np.full((3, cap - t, E), SENT, np.float32), 0.0, rows)
        # ffn
        ffn_inp = d.get("ffn_inp-%d" % slot, 0)
        qf = g.f32("blk.%d.ffn_res_query.weight" % ip)
        xf, _ = mix_ref(blocks_at(d, slot, True) + [ffn_inp], qf)
        x = np.zeros((1, cap, E), np.float32)
        x[:, :t] = xf.astype(np.float32)
        rx = np.zeros((1, cap), np.float32)
        rx[:, :t] = rms_inv(x[:, :t])[..., 0]
        wf3 = weights_raw(g, ip, 3)
        nwf = g.f32("blk.%d.ffn_norm.weight" % ip)
        out, nout = run_ctrl(device, "local", chunk, TOKENS, x, m, rx, rm, [wf3], nwf, 1, 0, 0, 3, t, cap, True)
        wd, wf, wg = ctrl_weights(g, ip, 3)
        exp = ctrl_ref(x[0, :t], msgs[:, 3], wd, wf, wg)
        check("slot %d ffn ctrl vs numpy" % slot, out[0, :t], exp, 1e-11, rows)
        e = nmse(nout[0, :t] , rms_norm(exp, nwf))
        check("slot %d ffn ctrl+norm(ffn_norm) vs numpy" % slot, nout[0, :t], rms_norm(exp, nwf), 1e-11, rows)
        e = nmse(rms_norm(out[0, :t]), d.get("norm-%d" % slot, 3))
        rows.append(("slot %d ffn norm vs oracle" % slot, e, e < 1e-4))
        print("%-44s nmse %.3e %s" % ("slot %d ffn norm vs oracle" % slot, e, "PASS" if e < 1e-4 else "FAIL"), flush=True)


def q4(device, chunk):
    """F32 controller weights of the Q4_K_XL file: kernel vs numpy on the same weights."""
    d = Dump("Q8_K_XL_A")
    g = Gguf("Q4")
    t = d.n
    cap = t + 3
    for slot in (24, 29, 41, 47):
        ip = phys(slot)
        glob = ip % 6 == 5
        prev = 17 if slot == 24 else (35 if slot == 36 else slot - 1)
        msgs = d.get("mix_out-%d" % prev, 2).reshape(t, 4, E)
        m = np.zeros((4, cap, E), np.float32)
        m[:, :t] = msgs.transpose(1, 0, 2)
        rm = np.zeros((4, cap), np.float32)
        rm[:, :t] = rms_inv(m[:, :t])[..., 0]
        ncx = 3 if glob else 1
        x = np.zeros((ncx, cap, E), np.float32)
        x[:, :t] = np.random.default_rng(slot).standard_normal((ncx, t, E)) * 4
        rx = np.zeros((ncx, cap), np.float32)
        rx[:, :t] = rms_inv(x[:, :t])[..., 0]
        wsets = [weights_raw(g, ip, c) for c in range(3)]
        out, nout = run_ctrl(device, "global" if glob else "local", chunk, TOKENS, x, m, rx, rm, wsets, np.ones(E, np.float32), 3, 0, 1 if glob else 0, 0, t, cap, False)
        for c in range(3):
            wd, wf, wg = ctrl_weights(g, ip, c)
            exp = ctrl_ref(x[c if glob else 0, :t], msgs[:, c], wd, wf, wg)
            check("Q4 F32 slot %d ctrl %s chunk %d vs numpy" % (slot, "qkv"[c], chunk), out[c, :t], exp, 1e-11, rows)


if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else "gfx1201"
    chunk = int(sys.argv[2]) if len(sys.argv) > 2 else 128
    TOKENS = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    if len(sys.argv) > 3 and sys.argv[3] == "q4":
        q4(device, chunk)
    else:
        main(device, chunk)
    bad = [r for r in rows if not r[2]]
    print("TOTAL %d FAIL %d" % (len(rows), len(bad)))
    sys.exit(1 if bad else 0)
