"""Per-shape timing of the dense kernels against traffic floors. Runs on the box; every measurement holds the GPU lock exclusively.
Usage: python bench_dense.py ARCH [filter] [--ring-mb N]
Timing: loom-benchmark dispatch_complete, batch 64, warmed, stable stop; the input ring is >= 200 MB so weights come from DRAM, not from cache.
"""
import json
import sys

import numpy as np

import test_gemm as tg
import test_gemv as tv
import test_norm_embed as tn
from dtest import *

BW = {"gfx1201": 620.0, "gfx1151": 230.0}  # measured GB/s
PEAK_TFLOPS = {"gfx1201": 190.0, "gfx1151": 59.0}  # nominal dense F16 matrix rate, only for the efficiency column


def p50_us(res):
    wi = res["work_items"][0]
    return wi["measurement"]["operation_timing_ns"]["p50"] / 1000.0


def entries():
    rng = np.random.default_rng(1)
    rn = lambda n, k: rng.standard_normal((n, k)).astype(np.float32)
    e = []

    def gemv(label, which, wname, n, family="", bytes_=None):
        tv.FAMILY = family
        c, *_ = tv.gemv_case(label, which, wname, rn(n, load_weight(which, wname)[2][0]))
        tv.FAMILY = ""
        raw = tensor_raw(which, wname)[0]
        e.append((label, c, raw.nbytes, 0.0, 0.0))

    def gemm(label, sym, which, wname, n, cap):
        k = tensor_raw(which, wname)[2][0]
        c, ref, rows, *_ = tg.gemm_case(sym, label, which, wname, rn(n, k), cap)
        nn = tensor_raw(which, wname)[2][1]
        e.append((label, c, tensor_raw(which, wname)[0].nbytes, 2.0 * n * k * nn, n))

    for n in (1, 4, 8):
        gemv("gemv_q8_0 attn_q 3072x2304 n=%d" % n, "q4", "blk.0.attn_q.weight", n)
        gemv("gemv_bf16 attn_q 3072x2304 n=%d" % n, "q8", "blk.0.attn_q.weight", n)
    for label, which, wname in (("gemv_q8_0 attn_q 3072x2304 n=1 [cols=1 build]", "q4", "blk.0.attn_q.weight"), ("gemv_bf16 attn_q 3072x2304 n=1 [cols=1 build]", "q8", "blk.0.attn_q.weight"),
                                ("gemv_q8_0 lm_head 200058x2304 n=1 [cols=1 build]", "q4", "output.weight"), ("gemv_bf16 lm_head 200058x2304 n=1 [cols=1 build]", "q8", "output.weight")):
        gemv(label, which, wname, 1)
        e[-1][1].config = dict(e[-1][1].config, **{"inkling.dense.gemv.cols": 1})
    gemv("gemv_q8_0 attn_k 1024x2304 n=1", "q4", "blk.2.attn_k.weight", 1)
    gemv("gemv_bf16 attn_k 1024x2304 n=1", "q8", "blk.2.attn_k.weight", 1)
    gemv("gemv_q8_0 attn_rel 384x2304 n=1", "q4", "blk.2.attn_rel.weight", 1)
    gemv("gemv_bf16 attn_rel 384x2304 n=1", "q8", "blk.2.attn_rel.weight", 1)
    gemv("gemv_q8_0 attn_o 2304x3072 n=1", "q4", "blk.0.attn_output.weight", 1)
    gemv("gemv_bf16 attn_o 2304x3072 n=1", "q8", "blk.0.attn_output.weight", 1)
    gemv("gemv_bf16 sink 24x2304 n=1", "q8", "blk.0.attn_sink_proj.weight", 1)
    gemv("gemv_split_bf16 sink 24x2304 n=1", "q8", "blk.0.attn_sink_proj.weight", 1, family="split_")
    gemv("gemv_bf16 conv_proj 32x2304 n=1", "q8", "blk.5.attn_k_conv_proj.weight", 1)
    gemv("gemv_split_bf16 conv_proj 32x2304 n=1", "q8", "blk.5.attn_k_conv_proj.weight", 1, family="split_")
    gemv("gemv_f32 router 128x2304 n=1", "q4", "blk.2.ffn_gate_inp.weight", 1)
    gemv("gemv_split_f32 router 128x2304 n=1", "q4", "blk.2.ffn_gate_inp.weight", 1, family="split_")
    gemv("gemv_split_f32 ctrl_down 64x4608 n=1", "q4", "blk.6.loop_q_ctrl_down.weight", 1, family="split_")
    gemv("gemv_split_bf16 ctrl_down 64x4608 n=1", "q8", "blk.6.loop_q_ctrl_down.weight", 1, family="split_")
    gemv("gemv_split_f32 ctrl_gain 2304x64 n=1", "q4", "blk.6.loop_q_ctrl_gain.weight", 1, family="split_")
    gemv("gemv_split_bf16 ctrl_gain 2304x64 n=1", "q8", "blk.6.loop_q_ctrl_gain.weight", 1, family="split_")
    gemv("gemv_rows_f32 ctrl_gain 2304x64 n=1", "q4", "blk.6.loop_q_ctrl_gain.weight", 1, family="rows_")
    gemv("gemv_rows_bf16 ctrl_gain 2304x64 n=1", "q8", "blk.6.loop_q_ctrl_gain.weight", 1, family="rows_")
    gemv("gemv_rows_f32 ctrl_down 64x4608 n=1", "q4", "blk.6.loop_q_ctrl_down.weight", 1, family="rows_")
    gemv("gemv_rows_bf16 ctrl_down 64x4608 n=1", "q8", "blk.6.loop_q_ctrl_down.weight", 1, family="rows_")
    gemv("gemv_rows_bf16 sink 24x2304 n=1", "q8", "blk.0.attn_sink_proj.weight", 1, family="rows_")
    gemv("gemv_rows_bf16 conv_proj 32x2304 n=1", "q8", "blk.5.attn_k_conv_proj.weight", 1, family="rows_")
    gemv("gemv_rows_f32 router 128x2304 n=1", "q4", "blk.2.ffn_gate_inp.weight", 1, family="rows_")
    for n in (1, 4, 8):
        gemv("gemv_q8_0 lm_head 200058x2304 n=%d" % n, "q4", "output.weight", n)
        gemv("gemv_bf16 lm_head 200058x2304 n=%d" % n, "q8", "output.weight", n)
    for n, cap in ((16, 16), (128, 128), (308, 320), (512, 512)):
        gemm("gemm_q8_0 attn_q n=%d" % n, "inkling_gemm_q8_0", "q4", "blk.0.attn_q.weight", n, cap)
        gemm("gemm_wide_q8_0 attn_q n=%d" % n, "inkling_gemm_wide_q8_0", "q4", "blk.0.attn_q.weight", n, cap)
        gemm("gemm_bf16 attn_q n=%d" % n, "inkling_gemm_bf16", "q8", "blk.0.attn_q.weight", n, cap)
        gemm("gemm_wmma_bf16 attn_q n=%d" % n, "inkling_gemm_wmma_bf16", "q8", "blk.0.attn_q.weight", n, cap)
    gemm("gemm_q8_0 attn_o n=512", "inkling_gemm_q8_0", "q4", "blk.0.attn_output.weight", 512, 512)
    gemm("gemm_bf16 attn_o n=512", "inkling_gemm_bf16", "q8", "blk.0.attn_output.weight", 512, 512)
    gemm("gemm_wmma_bf16 attn_o n=512", "inkling_gemm_wmma_bf16", "q8", "blk.0.attn_output.weight", 512, 512)
    gemm("gemm_wmma_bf16 lm_head n=16", "inkling_gemm_wmma_bf16", "q8", "output.weight", 16, 16)
    gemm("gemm_bf16 sink 24x2304 n=512", "inkling_gemm_bf16", "q8", "blk.0.attn_sink_proj.weight", 512, 512)
    gemm("gemm_q8_0 lm_head n=16", "inkling_gemm_q8_0", "q4", "output.weight", 16, 16)
    for label, case in (("rms_norm 2304 n=1", tn.norm_case("rn1", rn(1, 2304), np.ones(2304, np.float32), np.zeros((1, 2304)))),
                        ("rms_norm 2304 n=512", tn.norm_case("rn512", rn(512, 2304), np.ones(2304, np.float32), np.zeros((512, 2304)))),
                        ("rms_norm_heads 8x128 n=512", tn.norm_case("rh512", rn(512, 1024), None, np.zeros((512, 1024)), kind="heads"))):
        c = case[0]
        c.name = label.replace(" ", "_").replace("=", "")
        x = c.inputs["x"]
        e.append((label, c, 2 * x.nbytes, 0.0, 0.0))
    toks = np.arange(1, 513, dtype=np.int32) * 391
    raw8, t8, ne8, _ = load_weight("q8", "token_embd.weight")
    raw4, t4, ne4, _ = load_weight("q4", "token_embd.weight")
    for n in (1, 512):
        c = tn.embed_case("embed_bf16_n%d" % n, "bf16", raw8, ne8[1], toks[:n], np.zeros((n, 2304)))[0]
        e.append(("embed_bf16 gather+norm n=%d" % n, c, n * 4608, 0.0, 0.0))
        c = tn.embed_case("embed_q8_0_n%d" % n, "q8_0", raw4, ne4[1], toks[:n], np.zeros((n, 2304)))[0]
        e.append(("embed_q8_0 gather+norm n=%d" % n, c, n * 2448, 0.0, 0.0))
    return e


def main():
    arch = sys.argv[1]
    flt = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("--") else ""
    rows = []
    for label, case, nbytes, flops, n in entries():
        if flt and flt not in label:
            continue
        case.name = "".join(ch if ch.isalnum() else "_" for ch in case.name)
        res, err = bench_case(case, arch, "/tmp/inkF_b/%s/%s" % (arch, case.name), ring_mb=200)
        if res is None:
            print("FAIL %-8s %-40s %s" % (arch, label, err.replace("\n", " ")[:300]), flush=True)
            continue
        us = p50_us(res)
        floor_us = nbytes / (BW[arch] * 1e3)
        gbs = nbytes / (us * 1e3)
        line = "%-8s %-42s time %9.1f us  bytes %7.2f MB  floor %8.1f us  achieved %6.1f GB/s (%3.0f%% of %d)" % (arch, label, us, nbytes / 1e6, floor_us, gbs, 100 * gbs / BW[arch], BW[arch])
        if flops:
            tf = flops / (us * 1e6)
            line += "  %6.2f TFLOP/s (%2.0f%% of nominal %d)" % (tf, 100 * tf / PEAK_TFLOPS[arch], PEAK_TFLOPS[arch])
        print(line, flush=True)
        rows.append(dict(arch=arch, label=label, us=us, bytes=nbytes, floor_us=floor_us, gbs=gbs, flops=flops))
    json.dump(rows, open("bench_%s%s.json" % (arch, "_" + flt.replace(" ", "_") if flt else ""), "w"), indent=1)


if __name__ == "__main__":
    main()
