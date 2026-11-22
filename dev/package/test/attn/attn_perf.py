#!/usr/bin/env python3
"""Per-kernel timing of the attention kernels with loom-benchmark (dispatch_complete). Runs on the box.

usage: attn_perf.py [--dev gfx1151|gfx1201] [--only REGEX] [--batch 32]
Inputs are random (F16 K/V pool in [-1, 1]); only time is measured. Each benchmark runs one kernel launch;
floors use measured bandwidth R9700 620 GB/s, 8060S 230 GB/s.
"""
import argparse, json, os, re, subprocess, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import attn_test as T

BW = {"gfx1201": 620e9, "gfx1151": 230e9}
VENV = f"{T.HOME}/loom-serve/.venv-inkE/bin"


class Perf:
    def __init__(self):
        self.cases, self.names = [], []

    def add(self, name, build):
        c = T.Case(name)
        info = build(c)
        self.cases.append((c, info))
        self.names.append((name, info))


def common(c, d, glob, spans, rows_pages, state_halves, bps, max_splits, cap, kv_origin=0, slot_stride=8 << 20):
    rows = np.zeros(32 + rows_pages, np.int32)
    rows[2] = 0
    rows[8] = 128
    rows[32:] = np.arange(rows_pages)
    T.npy(f"{d}/rows.npy", rows)
    c.read("rows", f"{d}/rows.npy", rows, "i32")
    sp = T.spans_table(spans, 4)
    T.npy(f"{d}/spans.npy", sp)
    c.read("spans", f"{d}/spans.npy", sp, "i32")
    c.lit("seed", 7, "i64")
    c.lit("span_cap", 4)
    c.lit("global", int(glob))
    c.lit("conv_origin", T.CONV_ORIGIN, "offset")
    c.lit("kv_origin", kv_origin, "offset")
    c.lit("split_blocks", bps)
    c.lit("max_splits", max_splits)
    c.lit("max_blocks", rows_pages)
    c.lit("cap", cap)
    for k, v in (("mode0", 0), ("mode1", 1), ("kv8", 8), ("heads", 24), ("one", 1)):
        c.lit(k, v)


def rnd(c, var, shape, dt="f32", lo=-1.0, hi=1.0, seed=7):
    t = "tensor<" + "x".join(map(str, shape)) + "x" + dt + ">"
    c.lines.append(f"  %{var} = check.generate.random.uniform seed(%seed) range({float(lo)} to {float(hi)}) : {t}")
    c.vals[var] = t


def build_tile(kind, glob, ctx, n, bps, ms=None):
    """kind 'dec' (one decode token at position ctx-1) or 'pre' (n tokens starting at position ctx)."""
    def build(c):
        d = f"{T.FIX}/perf_{c.name}"
        pages = max(32, (ctx + n) // 64 + 2)
        nb = (ctx + n) // 64 + 1
        max_splits = ms or max(1, -(-(nb if glob else min(17, nb)) // bps))
        halves = max(pages * 131072, (T.SLOT0 + 2 * T.SLOT_STRIDE) // 2) if glob else (T.SLOT0 + 2 * T.SLOT_STRIDE) // 2
        if kind == "dec":
            spans, cap, pos0, ln = [(0, 1, T.SLOT0, ctx - 1, 0)], 1, ctx - 1, 1
        else:
            spans, cap, pos0, ln = [(0, n, T.SLOT0, ctx, 0)], n, ctx, n
        common(c, d, glob, spans, pages, halves, bps, max_splits, cap, kv_origin=0)
        rnd(c, "state", [halves], "f16")
        rnd(c, "q", [cap, 3072], lo=-3, hi=3)
        rnd(c, "r", [cap, 384], lo=-1, hi=1)
        rnd(c, "k_act", [cap, 1024], "f16")
        rnd(c, "v_act", [cap, 1024], "f16")
        rnd(c, "rel_bias", [1024, 16], lo=-0.5, hi=0.5)
        rnd(c, "sink", [cap, 24], lo=-2, hi=2)
        rnd(c, "gate", [cap, 3072])
        c.zeros("out", [cap, 3072])
        c.zeros("partial", [4 * 24 * max_splits * 136])
        c.lit("tiles", -(-ln // 16) + 4)
        tile = "cap span_cap {m} global split_blocks max_splits max_blocks kv_origin spans rows state q r k_act v_act rel_bias sink gate out partial"
        if kind == "dec":
            c.launch(*T.DECL["tile"], ["span_cap", "kv8", "max_splits"], tile.format(m="mode1").split())
        else:
            c.launch(*T.DECL["tile"], ["tiles", "heads", "one"], tile.format(m="mode0").split())
        keys = (ctx if glob else min(ctx, 1024)) if kind == "dec" else None
        return dict(kind=kind, glob=glob, ctx=ctx, n=n, bytes=4096 * keys if keys else None, splits=max_splits)
    return build


def build_merge(glob, ctx, bps):
    def build(c):
        d = f"{T.FIX}/perf_{c.name}"
        nb = ctx // 64 + 1
        max_splits = max(1, -(-(nb if glob else min(17, nb)) // bps))
        common(c, d, glob, [(0, 1, 0, ctx - 1, 0)], 32, 0, bps, max_splits, 1)
        rnd(c, "sink", [1, 24])
        rnd(c, "gate", [1, 3072])
        rnd(c, "partial", [4 * 24 * max_splits * 136])
        c.zeros("out", [1, 3072])
        c.launch(*T.DECL["merge"], ["span_cap"], "cap span_cap global split_blocks max_splits spans sink gate out partial".split())
        return dict(kind="merge", glob=glob, ctx=ctx, bytes=24 * max_splits * 136 * 4)
    return build


def build_prep(n, glob):
    def build(c):
        d = f"{T.FIX}/perf_{c.name}"
        common(c, d, glob, [(0, n, T.SLOT0, 0 if n > 1 else 700, 0)], 32, 0, 8, 1, n)
        c.zeros("state", [(T.SLOT0 + 2 * T.SLOT_STRIDE) // 2], "f16")
        for k, w in (("q_raw", 3072), ("k_raw", 1024), ("v_raw", 1024), ("r_raw", 384), ("dyn_k", 32), ("dyn_v", 32)):
            rnd(c, k, [n, w])
        for k, shape in (("w_q_clip", [24, 128]), ("w_rel_clip", [24, 16]), ("w_k_conv", [1024, 4]), ("w_v_conv", [1024, 4])):
            rnd(c, k, shape)
        c.zeros("q", [n, 3072])
        c.zeros("r", [n, 384])
        c.zeros("k_act", [n, 1024], "f16")
        c.zeros("v_act", [n, 1024], "f16")
        c.launch(*T.DECL["prep"], ["cap"], "cap span_cap global conv_origin spans state q_raw k_raw v_raw r_raw dyn_k dyn_v w_q_clip w_rel_clip w_k_conv w_v_conv q r k_act v_act".split())
        return dict(kind="prep", glob=glob, n=n, bytes=n * (3072 + 1024 * 2 + 384 + 64) * 4 + n * (3072 + 384) * 4 + n * 2048 * 2)
    return build


def build_commit_append(what, n, glob):
    def build(c):
        d = f"{T.FIX}/perf_{c.name}"
        common(c, d, glob, [(0, n, T.SLOT0, 0, 0)], 64, 0, 8, 1, n)
        c.zeros("state", [(T.SLOT0 + 2 * T.SLOT_STRIDE) // 2], "f16")
        rnd(c, "k_raw", [n, 1024])
        rnd(c, "v_raw", [n, 1024])
        rnd(c, "k_act", [n, 1024], "f16")
        rnd(c, "v_act", [n, 1024], "f16")
        if what == "commit":
            c.launch(*T.DECL["conv"], ["span_cap"], "span_cap cap conv_origin spans state k_raw v_raw".split())
            b = 2048 * 3 * 4 * 2 + n * 2048 * 4 * 0 + min(n, 3) * 2048 * 4
        else:
            c.launch(*T.DECL["append"], ["cap"], "cap span_cap global max_blocks kv_origin spans rows state k_act v_act".split())
            b = n * 4096 * 2
        return dict(kind=what, glob=glob, n=n, bytes=b)
    return build


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="gfx1201")
    ap.add_argument("--only", default="")
    ap.add_argument("--batch", type=int, default=32)
    a = ap.parse_args()
    perf = Perf()
    for ctx in (1024, 4096, 16384, 32768):
        for bps in (1, 2, 4, 8, 16):
            perf.add(f"dec_g_{ctx}_b{bps}", build_tile("dec", True, ctx, 1, bps))
    for bps in (1, 2, 4, 8):
        perf.add(f"dec_l_1024_b{bps}", build_tile("dec", False, 1500, 1, bps))
    perf.add("dec_g_1024_b4_grid128", build_tile("dec", True, 1024, 1, 4, 128))
    perf.add("dec_g_4096_b4_grid128", build_tile("dec", True, 4096, 1, 4, 128))
    for ctx in (1024, 4096, 8192, 16384, 32768):
        perf.add(f"dec_g_{ctx}_dyn", build_tile("dec", True, ctx, 1, 1, 64))
    perf.add("dec_l_1024_b1_grid17", build_tile("dec", False, 1500, 1, 1, 17))
    perf.add("merge_g_32768", build_merge(True, 32768, 8))
    perf.add("merge_l", build_merge(False, 1500, 8))
    perf.add("pre_l_512_at0", build_tile("pre", False, 0, 512, 8))
    perf.add("pre_l_512_at1024", build_tile("pre", False, 1024, 512, 8))
    perf.add("pre_g_512_at0", build_tile("pre", True, 0, 512, 8))
    perf.add("pre_g_512_at4096", build_tile("pre", True, 4096, 512, 8))
    perf.add("prep_1", build_prep(1, False))
    perf.add("prep_512", build_prep(512, False))
    perf.add("commit_1", build_commit_append("commit", 1, False))
    perf.add("commit_512", build_commit_append("commit", 512, False))
    perf.add("append_g_1", build_commit_append("append", 1, True))
    perf.add("append_g_512", build_commit_append("append", 512, True))
    perf.add("append_l_512", build_commit_append("append", 512, False))
    rows = []
    for c, info in perf.cases:
        if not re.search(a.only, c.name):
            continue
        c.lines.append("  check.return_placeholder")
        txt = c.text().replace("\n  check.return_placeholder", "")
        txt += f"\ncheck.benchmark<@{c.name}> @{c.name}_bench\n"
        src = f"{T.ROOT}/build/perf_{c.name}.loom"
        open(src, "w").write(txt)
        merged = f"{T.ROOT}/build/perf_{c.name}_{a.dev}.loom"
        r = subprocess.run([f"{VENV}/loom-link", src] + T.SRC + ["--mode=merge", "--to=text", f"--output={merged}"], capture_output=True, text=True, cwd=T.ROOT)
        if r.returncode:
            print(c.name, "link failed", r.stderr[-800:])
            continue
        outj = f"{T.ROOT}/build/perf_{c.name}_{a.dev}.json"
        r = subprocess.run(["flock", "-x", f"{T.HOME}/hrx-pr1/gpu.lock", f"{VENV}/loom-benchmark", merged, f"--device={T.DEVICES[a.dev]}",
                            f"--benchmark=@{c.name}_bench", "--measure=dispatch_complete", f"--batch-size={a.batch}", f"--output={outj}"],
                           capture_output=True, text=True, cwd=T.ROOT)
        if r.returncode:
            print(c.name, "benchmark failed", r.stdout[-500:], r.stderr[-800:])
            continue
        j = json.load(open(outj))
        rows.append((c.name, info, j))
    print(f"{'kernel run':24s} {'time us':>9s} {'bytes':>12s} {'GB/s':>8s} {'floor us':>9s} {'x floor':>8s}")
    for name, info, j in rows:
        us = j["work_items"][0]["measurement"]["mean_physical_dispatch_duration_ns"] / 1e3
        b = info.get("bytes")
        if b:
            fl = b / BW[a.dev] * 1e6
            print(f"{name:24s} {us:9.2f} {b:12d} {b / us / 1e3:8.1f} {fl:9.2f} {us / fl:8.1f}")
        else:
            print(f"{name:24s} {us:9.2f}")
    json.dump([(n, i, j["work_items"][0]["measurement"]["mean_physical_dispatch_duration_ns"]) for n, i, j in rows], open(f"{T.ROOT}/build/perf_{a.dev}.json", "w"))


if __name__ == "__main__":
    main()
