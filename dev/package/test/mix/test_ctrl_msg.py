"""inkling_ctrl_down_msg + inkling_ctrl_apply_z (decode) against inkling_mix_states (message mix) + ctrl_down + ctrl_z + ctrl_apply on random data.
Usage: test_ctrl_msg.py [gfx1201|gfx1151] [chunk]. The message columns and the rms factors must be bit equal, the controller output must agree to 1e-12 (nmse)."""
import sys

import numpy as np

from harness import nmse, run_steps
from inkg import *
from test_mix import SENT, spans_buf

SRC = ["mix_states.loom", "conv_res.loom", "ctrl.loom"]
rows = []


def case(device, chunk, nc, x_col_step, m_col0, t, cap, bf16, seed):
    rng = np.random.default_rng(seed)
    cs = cap * E
    ncx = 3 if x_col_step else 1
    x = np.zeros((ncx, cap, E), np.float32)
    x[:, :t] = (rng.standard_normal((ncx, t, E)) * 2).astype(np.float32)
    rx = np.zeros((ncx, cap), np.float32)
    rx[:, :t] = rms_inv(x[:, :t])[..., 0]
    bank = np.zeros((2, cap, E), np.float32)
    bank[:, :t] = (rng.standard_normal((2, t, E)) * 1.5).astype(np.float32)
    query = (rng.standard_normal((4, E)) * 0.05).astype(np.float32)
    nwb = np.ones((4, E), np.float32)

    def w(shape, s):
        a = (rng.standard_normal(shape) * s).astype(np.float32)
        return a.reshape(-1).view(np.int16) if False else a.reshape(-1)

    sets = [(w((64, 4608), 0.02), w((E, 64), 0.1), w((E, 64), 0.1)) for _ in range(nc)]
    sets += [sets[-1]] * (3 - nc)
    if bf16:
        sets = [tuple(((a.view(np.uint32) >> 16).astype(np.uint16)).view(np.int16) for a in s) for s in sets]
    bufs = {"spans": spans_buf(t), "sel": np.zeros(4, np.int32), "x": x, "bank": bank, "query": query, "nw": nwb,
            "d0": sets[0][0], "d1": sets[1][0], "d2": sets[2][0], "f0": sets[0][1], "f1": sets[1][1], "f2": sets[2][1], "g0": sets[0][2], "g1": sets[1][2], "g2": sets[2][2],
            "rx": rx, "z": np.full((3, cap, 64), SENT, np.float32), "cs0": np.zeros(4, np.float32), "cs1": np.zeros(4, np.float32),
            "msgA": np.full((4, cap, E), SENT, np.float32), "rmA": np.full((4, cap), SENT, np.float32), "partA": np.full((3, cap, 4608 // chunk, 64), SENT, np.float32),
            "outA": np.full((3, cap, E), SENT, np.float32),
            "msgB": np.full((4, cap, E), SENT, np.float32), "rmB": np.full((4, cap), SENT, np.float32), "partB": np.full((3, cap, 4608 // chunk, 64), SENT, np.float32),
            "outB": np.full((3, cap, E), SENT, np.float32)}
    steps = [
        ("inkling_mix_states", (cap, 0, 4), [1, cs, cs, 0, 0, 0, 0, cs, 0, 0, 0], ["spans", "sel", "bank", "bank", "query", "nw", "msgA", "rmA", "cs0", "cs1"]),
        ("inkling_ctrl_down", (cap, nc), [0, x_col_step, m_col0, cs], ["spans", "x", "msgA", "d0", "d1", "d2", "partA"]),
        ("inkling_ctrl_z", (cap, nc), [0, x_col_step, m_col0], ["spans", "rx", "rmA", "partA", "z"]),
        ("inkling_ctrl_apply", (cap, nc), [0, x_col_step, m_col0, cs], ["spans", "x", "msgA", "f0", "f1", "f2", "g0", "g1", "g2", "z", "outA"]),
        ("inkling_ctrl_down_msg", (cap, nc), [0, x_col_step, m_col0, cs, 0, cs], ["spans", "x", "bank", "query", "d0", "d1", "d2", "msgB", "rmB", "partB"]),
        ("inkling_ctrl_apply_z", (cap, nc), [0, x_col_step, m_col0, cs], ["spans", "x", "msgB", "f0", "f1", "f2", "g0", "g1", "g2", "rx", "rmB", "partB", "outB"]),
    ]
    r = run_steps(SRC, bufs, steps, device, {"inkling.mix.ctrl_chunk": chunk, "inkling.mix.ctrl_tokens": 1, "inkling.mix.ctrl_bf16": int(bf16)})
    name = "nc %d m_col0 %d t %d cap %d chunk %d %s" % (nc, m_col0, t, cap, chunk, "bf16" if bf16 else "f32")
    for c in range(nc):
        col = m_col0 + c
        a, b = r["msgA"][col, :t], r["msgB"][col, :t]
        ok = np.array_equal(a, b)
        rows.append(("%s msg col %d bit equal" % (name, col), 0.0 if ok else nmse(b, a), ok))
        print("%-60s %s" % (rows[-1][0], "PASS" if ok else "FAIL (nmse %.3e)" % nmse(b, a)), flush=True)
        ok = np.array_equal(r["rmA"][col, :t], r["rmB"][col, :t])
        rows.append(("%s rinv_m col %d bit equal" % (name, col), 0.0, ok))
        print("%-60s %s" % (rows[-1][0], "PASS" if ok else "FAIL"), flush=True)
    e = nmse(r["outB"][:nc, :t], r["outA"][:nc, :t])
    rows.append(("%s controller output" % name, e, e < 1e-12))
    print("%-60s nmse %.3e %s" % (rows[-1][0], e, "PASS" if e < 1e-12 else "FAIL"), flush=True)
    guard = np.array_equal(r["outB"][:, t:], np.full((3, cap - t, E), SENT, np.float32))
    rows.append(("%s guard" % name, 0.0, guard))
    print("%-60s %s" % (rows[-1][0], "PASS" if guard else "FAIL"), flush=True)


if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else "gfx1201"
    chunk = int(sys.argv[2]) if len(sys.argv) > 2 else 256
    seed = 1
    for bf16 in (False, True):
        for t, cap in ((1, 1), (1, 8), (3, 4), (8, 8)):
            case(device, chunk, 3, 1, 0, t, cap, bf16, seed); seed += 1
            case(device, chunk, 3, 0, 0, t, cap, bf16, seed); seed += 1
            case(device, chunk, 1, 0, 3, t, cap, bf16, seed); seed += 1
    bad = [r for r in rows if not r[2]]
    print("TOTAL %d FAIL %d" % (len(rows), len(bad)))
    sys.exit(1 if bad else 0)
