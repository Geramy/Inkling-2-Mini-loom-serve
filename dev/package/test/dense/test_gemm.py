"""GEMM (up to 512 rows) tests for Q8_0 weights (matrix ops): numpy reference on the exact weights, oracle comparison for prompt A.
Usage: python test_gemm.py ARCH [filter]
"""
import sys

import numpy as np

from dtest import *

LIBS = ["lib/inkling/dense_common.loom", "lib/inkling/dense_float.loom", "lib/inkling/dense_q8_0.loom", "lib/inkling/dense_abi.loom",
        "lib/inkling/dense_mm_wmma.loom", "lib/inkling/dense_wmma.loom", "lib/inkling/dense_targets.loom", "kernel/dense/gemm_q8_0.loom", "kernel/dense/gemm_f32.loom", "kernel/dense/gemm_bf16_wmma.loom"]
CONFIG = {}
DECL = """kernel.decl @{sym}(%n_rows: index, %cols_cap: index) launch(%control: buffer, %count_origin: offset, %k_size: index, %n_rows: index, %w_origin: offset, %x_origin: offset, %x_stride: index, %d_origin: offset, %d_stride: index, %act: index, %cap: f32, %weight: buffer, %input: buffer, %output: buffer)"""
BODY = """  %n_rows = check.literal value({n}) : index
  %cols_cap = check.literal value({cap_rows}) : index
  %k = check.literal value({k}) : index
  %zero = check.literal value(0) : offset
  %act = check.literal value({act}) : index
  %cap = check.literal value({cap}) : f32
  kernel.launch @{sym}[%n_rows, %cols_cap](%ctl, %zero, %k, %n_rows, %zero, %zero, %k, %zero, %n_rows, %act, %cap, %w, %x, %y) : [index, index](tensor<4xi32>, offset, index, index, offset, offset, index, offset, index, index, f32, {wt}, {xt}, {yt})"""


def gemm_case(sym, name, which, wname, x, cap_rows, act=0, oracle_y=None, tol=5e-6, otol=2e-2):
    raw, ttype, ne, wdeq = load_weight(which, wname)
    k, n = ne[0], ne[1]
    rows = x.shape[0]
    xp = np.zeros((cap_rows, k), np.float32)
    xp[:rows] = x
    xp[rows:] = 7.0  # garbage beyond the live rows must not leak into the output
    ref = x.astype(np.float64) @ wdeq.astype(np.float64).T
    if act == 1:
        ref = 1 / (1 + np.exp(-ref))
    y0 = np.zeros((cap_rows, n), np.float32)
    ctl = np.array([rows, 0, 0, 0], dtype=np.int32)
    w = raw.view(np.int8)
    body = BODY.format(sym=sym, n=n, k=k, act=act, cap=30.0, cap_rows=cap_rows, wt=tensor_type(w), xt=tensor_type(xp), yt=tensor_type(y0))
    c = Case(name, LIBS, {"ctl": ctl, "w": w, "x": xp}, {"y": y0}, body, preamble=DECL.format(sym=sym), config=CONFIG)
    return c, ref, rows, oracle_y, tol, otol


def build_cases(sym):
    rng = np.random.default_rng(11)
    rn = lambda n, k: rng.standard_normal((n, k)).astype(np.float32)
    cases = []
    tag = "q8_0" if sym.endswith("q8_0") else sym.split("_")[-1]
    if tag == "q8_0":
        for n, cap in ((1, 64), (16, 64), (100, 128), (308, 320), (512, 512)):
            cases.append(gemm_case(sym, "q8_0_attn_q_n%d" % n, "q4", "blk.0.attn_q.weight", rn(n, 2304), cap))
        cases.append(gemm_case(sym, "q8_0_attn_o_n77", "q4", "blk.0.attn_output.weight", rn(77, 3072), 128))
        cases.append(gemm_case(sym, "q8_0_gate_sig_n33", "q4", "blk.3.attn_gate.weight", rn(33, 2304), 64, act=1))
        cases.append(gemm_case(sym, "q8_0_kv_1024_n200", "q4", "blk.2.attn_k.weight", rn(200, 2304), 256))
        cases.append(gemm_case(sym, "q8_0_rel_384_n130", "q4", "blk.2.attn_rel.weight", rn(130, 2304), 256))
        xa = oracle("A", "norm-0", 0) * load_weight("q8", "blk.0.attn_norm.weight")[3][0]
        # Q4 weights against the BF16 oracle: the difference is the weight quantization.
        cases.append(gemm_case(sym, "q8_0_attn_q_oracle_n16", "q4", "blk.0.attn_q.weight", xa, 64, oracle_y=oracle("A", "proj_q-0", 0), otol=3e-2))
        cases.append(gemm_case(sym, "q8_0_attn_k_oracle_n16", "q4", "blk.0.attn_k.weight", xa, 64, oracle_y=oracle("A", "conv_in-0", 0), otol=3e-2))
        return cases
    tol = 5e-6 if "wmma" in sym else 1e-9
    if tag == "bf16":
        w = "q8"
        names = dict(q="blk.0.attn_q.weight", o="blk.0.attn_output.weight", gate="blk.3.attn_gate.weight", small="blk.0.attn_sink_proj.weight",
                     conv="blk.5.attn_k_conv_proj.weight", down="blk.6.loop_q_ctrl_down.weight", gain="blk.6.loop_q_ctrl_gain.weight")
    else:
        w = "q4"
        names = dict(small="blk.0.ffn_gate_inp.weight", down="blk.6.loop_q_ctrl_down.weight", gain="blk.6.loop_q_ctrl_gain.weight")
    if "q" in names:
        for n, cap in ((1, 64), (9, 16), (16, 64), (100, 128), (308, 320), (512, 512)):
            cases.append(gemm_case(sym, "%s_attn_q_n%d" % (tag, n), w, names["q"], rn(n, 2304), cap, tol=tol))
        cases.append(gemm_case(sym, "%s_attn_o_n77" % tag, w, names["o"], rn(77, 3072), 128, tol=tol))
        cases.append(gemm_case(sym, "%s_gate_sig_n33" % tag, w, names["gate"], rn(33, 2304), 64, act=1, tol=tol))
        cases.append(gemm_case(sym, "%s_conv_32_n20" % tag, w, names["conv"], rn(20, 2304), 32, tol=tol))
        xa = oracle("A", "norm-0", 0) * load_weight("q8", "blk.0.attn_norm.weight")[3][0]
        cases.append(gemm_case(sym, "%s_attn_q_oracle_n16" % tag, w, names["q"], xa, 64, oracle_y=oracle("A", "proj_q-0", 0), tol=tol))
    cases.append(gemm_case(sym, "%s_small_rows_n20" % tag, w, names["small"], rn(20, 2304), 32, tol=tol))
    cases.append(gemm_case(sym, "%s_ctrl_down_4608_n12" % tag, w, names["down"], rn(12, 4608), 16, tol=tol))
    cases.append(gemm_case(sym, "%s_ctrl_gain_k64_n12" % tag, w, names["gain"], rn(12, 64), 16, tol=tol))
    return cases


def main():
    arch = sys.argv[1]
    flt = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else ""
    sym = "inkling_" + (sys.argv[3] if len(sys.argv) > 3 else "gemm_q8_0")
    rep = Report()
    for c, ref, rows, oy, tol, otol in build_cases(sym):
        if flt and flt not in c.name:
            continue
        out, err = run_case(c, arch, "/tmp/inkF_t/%s/%s_%s" % (arch, sym, c.name))
        if out is None:
            rep.add(sym + ":" + c.name, arch, False, "", err.replace("\n", " ")[:700])
            continue
        y = out["y"]
        e = nmse(y[:rows], ref)
        pad_ok = bool(np.all(y[rows:] == 0.0))
        m = "nmse=%.2e pad_untouched=%s" % (e, pad_ok)
        ok = e < tol and pad_ok
        if oy is not None:
            ro = rel_l2(y[:rows], oy)
            m += " oracle_relL2=%.2e" % ro
            ok = ok and ro < otol
        rep.add(sym + ":" + c.name, arch, ok, m)
    rep.finish()


if __name__ == "__main__":
    main()
