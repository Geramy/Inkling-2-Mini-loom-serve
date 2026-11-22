"""GEMV (1..8 rows) tests: real GGUF weights, numpy reference on the exact weight values, oracle comparison where the oracle has the tensor.
Usage: python test_gemv.py ARCH [filter|-] [split]     (split selects the split-K kernels)
"""
import sys

import numpy as np

from dtest import *

KINDS = {"Q8_0": "q8_0", "BF16": "bf16", "F32": "f32"}
LIBS_GEMV = ["lib/inkling/dense_common.loom", "lib/inkling/dense_float.loom", "lib/inkling/dense_q8_0.loom", "lib/inkling/dense_abi.loom",
             "lib/inkling/dense_vec.loom", "kernel/dense/gemv.loom", "kernel/dense/gemv_split.loom"]
CONFIG = {}

PER_UNIT = {"gfx1151": 4}.get(sys.argv[1], 2)  # the value the dense_vec.loom providers give for the device
DECL = """kernel.decl @inkling_gemv_{kind}({workload}) launch(%control: buffer, %count_origin: offset, %k_size: index, %n_rows: index, %w_origin: offset, %x_origin: offset, %x_stride: index, %d_origin: offset, %d_stride: index, %act: index, %cap: f32, %weight: buffer, %input: buffer, %output: buffer)"""

BODY = """{pre}  %n_rows = check.literal value({n}) : index
  %k = check.literal value({k}) : index
  %zero = check.literal value(0) : offset
  %act = check.literal value({act}) : index
  %cap = check.literal value({cap}) : f32
  kernel.launch @inkling_gemv_{kind}[{wargs}](%ctl, %zero, %k, %n_rows, %zero, %zero, %k, %zero, %n_rows, %act, %cap, %w, %x, %y) : [{wtypes}](tensor<4xi32>, offset, index, index, offset, offset, index, offset, index, index, f32, {wt}, {xt}, {yt})"""

FAMILY = ""


def gemv_case(name, which, wname, x, act=0, cap=30.0, oracle_y=None, tol=1e-9, otol=2e-2):
    raw, ttype, ne, wdeq = load_weight(which, wname)
    k, n = ne[0], ne[1]
    rows = x.shape[0]
    w = raw.view(np.int8)
    ref = x.astype(np.float64) @ wdeq.astype(np.float64).T
    if act == 1:
        ref = 1 / (1 + np.exp(-ref))
    elif act == 2:
        ref = cap * np.tanh(ref / cap)
    y0 = np.zeros((rows, n), np.float32)
    ctl = np.array([rows, 0, 0, 0], dtype=np.int32)
    kind = FAMILY + KINDS[ttype]
    if FAMILY:
        workload, wargs, wtypes = "%n_rows: index", "%n_rows", "index"
    else:
        workload, wargs, wtypes = "%n_rows: index, %k_size: index, %per_unit: index", "%n_rows, %k, %per_unit", "index, index, index"
    pre = "" if FAMILY else "  %%per_unit = check.literal value(%d) : index\n" % PER_UNIT
    body = BODY.format(pre=pre, n=n, k=k, act=act, cap=cap, kind=kind, wargs=wargs, wtypes=wtypes, wt=tensor_type(w), xt=tensor_type(x), yt=tensor_type(y0))
    c = Case(name, LIBS_GEMV, {"ctl": ctl, "w": w, "x": x}, {"y": y0}, body, preamble=DECL.format(kind=kind, workload=workload), config=CONFIG)
    return c, ref, oracle_y, tol, otol


def xs_cases(rng, xa, xk5, xo):
    rn = lambda n, k: rng.standard_normal((n, k)).astype(np.float32)
    c = []
    for n in (1, 3, 8):
        c.append(gemv_case("q8_0_attn_q_n%d" % n, "q4", "blk.0.attn_q.weight", rn(n, 2304)))
        c.append(gemv_case("bf16_attn_q_n%d" % n, "q8", "blk.0.attn_q.weight", rn(n, 2304)))
    c.append(gemv_case("bf16_attn_q_oracle", "q8", "blk.0.attn_q.weight", xa, oracle_y=oracle("A", "proj_q-0", 0)[:8]))
    c.append(gemv_case("bf16_attn_k_oracle", "q8", "blk.0.attn_k.weight", xa, oracle_y=oracle("A", "conv_in-0", 0)[:8]))
    c.append(gemv_case("bf16_attn_v_oracle", "q8", "blk.0.attn_v.weight", xa, oracle_y=oracle("A", "conv_in-0", 1)[:8]))
    c.append(gemv_case("bf16_attn_rel_oracle", "q8", "blk.0.attn_rel.weight", xa, oracle_y=oracle("A", "proj_r-0", 0)[:8]))
    c.append(gemv_case("bf16_attn_o_oracle", "q8", "blk.0.attn_output.weight", xo, oracle_y=oracle("A", "conv_in-0", 2)[:8]))
    c.append(gemv_case("q8_0_attn_o", "q4", "blk.0.attn_output.weight", rn(2, 3072)))
    c.append(gemv_case("q8_0_gate_sigmoid", "q4", "blk.3.attn_gate.weight", rn(3, 2304), act=1))
    c.append(gemv_case("bf16_sink_24", "q8", "blk.0.attn_sink_proj.weight", xa))
    c.append(gemv_case("bf16_conv_proj_32", "q8", "blk.5.attn_k_conv_proj.weight", xk5))
    c.append(gemv_case("f32_router_128", "q8", "blk.0.ffn_gate_inp.weight", xa))
    c.append(gemv_case("q8_0_lm_head_softcap", "q4", "output.weight", rn(2, 2304) * 0.1, act=2, cap=30.0))
    return c


def split_cases(rng, xa, xk5, xo):
    rn = lambda n, k: rng.standard_normal((n, k)).astype(np.float32)
    c = []
    for n in (1, 3, 8):
        c.append(gemv_case("bf16_sink_24_n%d" % n, "q8", "blk.0.attn_sink_proj.weight", rn(n, 2304)))
        c.append(gemv_case("bf16_conv_proj_32_n%d" % n, "q8", "blk.5.attn_v_conv_proj.weight", rn(n, 2304)))
        c.append(gemv_case("bf16_ctrl_down_4608_n%d" % n, "q8", "blk.6.loop_k_ctrl_down.weight", rn(n, 4608)))
        c.append(gemv_case("bf16_ctrl_gain_k64_n%d" % n, "q8", "blk.7.loop_ffn_ctrl_gain.weight", rn(n, 64)))
        c.append(gemv_case("f32_router_128_n%d" % n, "q4", "blk.2.ffn_gate_inp.weight", rn(n, 2304)))
        c.append(gemv_case("f32_ctrl_down_4608_n%d" % n, "q4", "blk.6.loop_q_ctrl_down.weight", rn(n, 4608)))
        c.append(gemv_case("f32_ctrl_forget_k64_n%d" % n, "q4", "blk.9.loop_v_ctrl_forget.weight", rn(n, 64)))
    c.append(gemv_case("bf16_sink_24_oracle", "q8", "blk.0.attn_sink_proj.weight", xa))
    c.append(gemv_case("bf16_conv_proj_32_input", "q8", "blk.5.attn_k_conv_proj.weight", xk5))
    c.append(gemv_case("bf16_sink_24_sigmoid", "q8", "blk.0.attn_sink_proj.weight", rn(2, 2304), act=1))
    c.append(gemv_case("bf16_attn_q_oracle", "q8", "blk.0.attn_q.weight", xa, oracle_y=oracle("A", "proj_q-0", 0)[:8]))
    return c


def main():
    global FAMILY
    arch = sys.argv[1]
    flt = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else ""
    FAMILY = {"split": "split_", "rows": "rows_"}.get(sys.argv[3] if len(sys.argv) > 3 else "", "")
    rep = Report()
    rng = np.random.default_rng(7)
    wn = lambda nm: load_weight("q8", nm)[3][0]
    xa = (oracle("A", "norm-0", 0) * wn("blk.0.attn_norm.weight"))[:8]  # attn input of layer 0, prompt A (oracle norm tensors have no weight)
    xk5 = (oracle("A", "norm-5", 0) * wn("blk.5.attn_k_inp_norm.weight"))[:8]
    gate0 = 1 / (1 + np.exp(-(xa.astype(np.float64) @ load_weight("q8", "blk.0.attn_gate.weight")[3].astype(np.float64).T)))
    xo = (oracle("A", "kqv_out-0", 0)[:8] * gate0).astype(np.float32)
    cases = (split_cases if FAMILY else xs_cases)(rng, xa, xk5, xo)
    for c, ref, oy, tol, otol in cases:
        if flt and flt not in c.name:
            continue
        out, err = run_case(c, arch, "/tmp/inkF_t/%s/%s%s" % (arch, FAMILY, c.name))
        if out is None:
            rep.add(FAMILY + c.name, arch, False, "", err.replace("\n", " ")[:600])
            continue
        e = nmse(out["y"], ref)
        m = "nmse=%.2e" % e
        ok = e < tol
        if oy is not None:
            ro = rel_l2(out["y"], oy)
            m += " oracle_relL2=%.2e" % ro
            ok = ok and ro < otol
        rep.add(FAMILY + c.name, arch, ok, m)
    rep.finish()


if __name__ == "__main__":
    main()
