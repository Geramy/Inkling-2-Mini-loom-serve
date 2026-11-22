"""Per-kernel timing vs traffic floor (exclusive GPU lock, batched dispatches). Usage: bench_mix.py [gfx1201|gfx1151] [only-regex]"""
import re
import sys

import numpy as np

from harness import bench_steps
from inkg import E
from test_mix import spans_buf

BW = {"gfx1201": 620e9, "gfx1151": 230e9}
ROW = 6656
import os
TOK = int(os.environ.get("TOK", "16"))
CHUNK = int(os.environ.get("CHUNK", "128"))


def timed(device, sources, bufs, steps, batch, config=None):
    r = bench_steps(sources, bufs, steps, device, config, extra=["--batch-size=%d" % batch])
    return r["work_items"][0]["measurement"]["operation_timing_ns"]["p50"] / 1000.0


def row(name, us, nbytes, device):
    floor = nbytes / BW[device] * 1e6
    print("%-52s %9.2f us  traffic %8.2f MB  floor %8.2f us  x%.1f" % (name, us, nbytes / 1e6, floor, us / floor), flush=True)


def main(device, only):
    rng = np.random.default_rng(0)

    def want(n):
        return re.search(only, n) is not None

    for t in (1, 512):
        batch = 100 if t == 1 else 10
        cap = t
        spans = spans_buf(t, [t])
        sel = np.zeros(4, np.int32)
        for nb, nq in ((3, 1), (3, 3), (7, 1), (1, 4)):
            name = "mix_states T=%d nb=%d nq=%d" % (t, nb, nq)
            if not want(name):
                continue
            bufs = {"spans": spans, "sel": sel, "blocks": rng.standard_normal((8, cap, E)).astype(np.float32),
                    "cur": rng.standard_normal((cap, E)).astype(np.float32), "q": rng.standard_normal((4, E)).astype(np.float32),
                    "nw": np.ones((4, E), np.float32), "out": np.zeros((nq, cap, E), np.float32), "rinv": np.zeros((4, cap), np.float32), "state": np.zeros(4, np.float32), "a": np.zeros(4, np.float32)}
            us = timed(device, ["mix_states.loom", "conv_res.loom"], bufs, [("inkling_mix_states", [cap, 0, nq], [nb, cap * E, cap * E, 0, 0, 0, 0, 0, 0, 0, 0],
                                                              ["spans", "sel", "blocks", "cur", "q", "nw", "out", "rinv", "state", "a"])], batch)
            row(name, us, (nb + 1) * t * E * 4 + nq * t * E * 4, device)
        for nc, bf16 in ((3, 0), (3, 1), (1, 0)):
            name = "ctrl_down+apply+norm T=%d nc=%d %s" % (t, nc, "bf16" if bf16 else "f32")
            if not want(name):
                continue
            wb = 2 if bf16 else 4
            dt = np.int16 if bf16 else np.float32
            ch = CHUNK
            bufs = {"spans": spans, "x": rng.standard_normal((nc, cap, E)).astype(np.float32), "m": rng.standard_normal((4, cap, E)).astype(np.float32),
                    "rx": np.ones((nc, cap), np.float32), "rm": np.ones((4, cap), np.float32),
                    "d": rng.integers(0, 1000, 294912).astype(dt) if bf16 else rng.standard_normal(294912).astype(dt),
                    "f": rng.integers(0, 1000, 294912).astype(dt) if bf16 else rng.standard_normal(294912).astype(dt),
                    "part": np.zeros((3, cap, 4608 // ch, 64), np.float32), "z": np.zeros((3, cap, 64), np.float32), "out": np.zeros((3, cap, E), np.float32),
                    "w": np.ones((3, E), np.float32), "nout": np.zeros((3, cap, E), np.float32)}
            cs = cap * E
            steps = [("inkling_ctrl_down", (cap, nc), [0, 1, 0, cs], ["spans", "x", "m", "d", "d", "d", "part"]),
                     ("inkling_ctrl_z", (cap, nc), [0, 1, 0], ["spans", "rx", "rm", "part", "z"]),
                     ("inkling_ctrl_apply", (cap, nc), [0, 1, 0, cs], ["spans", "x", "m", "f", "f", "f", "f", "f", "f", "z", "out"]),
                     ("inkling_mix_norm", (cap, nc), [cs, 0], ["spans", "out", "w", "nout"])]
            cfg = {"inkling.mix.ctrl_chunk": ch, "inkling.mix.ctrl_bf16": bf16, "inkling.mix.ctrl_tokens": 1 if t == 1 else TOK}
            us = [timed(device, ["ctrl.loom"], bufs, [s], batch, cfg) for s in steps]
            floors = [nc * 4608 * 64 * wb + t * nc * 2 * E * 4, t * nc * 36 * 64 * 4, nc * 2 * 2304 * 64 * wb + t * nc * 3 * E * 4, 2 * t * nc * E * 4]
            for n, u, f in zip(("down", "z", "apply", "norm"), us, floors):
                row("%s %s" % (name, n), u, f, device)
            row(name + " total", sum(us), sum(floors), device)
        for site in ("attn", "ffn"):
            name = "conv_res+commit %s T=%d" % (site, t)
            if not want(name):
                continue
            bufs = {"spans": spans, "state": rng.standard_normal(2 * 3 * ROW).astype(np.float32), "a": rng.standard_normal((cap, E)).astype(np.float32),
                    "w": rng.standard_normal((E, 4)).astype(np.float32), "cur": rng.standard_normal((cap, E)).astype(np.float32),
                    "blocks": np.zeros((8, cap, E), np.float32), "bank": np.zeros((2, cap, E), np.float32)}
            push = 1 if site == "attn" else 0
            s1 = ("inkling_conv_res", cap, [2048 if site == "attn" else 4352, 0, push, 3, cap * E, 0, 0], ["spans", "state", "a", "w", "cur", "blocks", "bank"])
            s2 = ("inkling_conv_commit", 1, [2048 if site == "attn" else 4352, 0], ["spans", "state", "a"])
            u1 = timed(device, ["conv_res.loom"], bufs, [s1], batch)
            u2 = timed(device, ["conv_res.loom"], bufs, [s2], batch)
            row(name + " conv_res", u1, t * E * 4 * (3 + push) + t * E * 16, device)
            row(name + " commit", u2, 3 * E * 4 * 2 + 3 * E * 4, device)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "gfx1201", sys.argv[2] if len(sys.argv) > 2 else ".")
