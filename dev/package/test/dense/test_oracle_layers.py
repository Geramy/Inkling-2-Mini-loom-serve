"""Projection outputs of whole layers against the oracle (prompt A, 16 rows): q, k, v, rel, gate+o (through kqv_out), router logits.
BF16 weights (Q8_K_XL file) use inkling_gemm_bf16 and inkling_gemm_f32 (router); Q8_0 weights (Q4_K_XL file) use inkling_gemm_q8_0 and are compared with the larger tolerance of a different quantization.
Usage: python test_oracle_layers.py ARCH [filter]
"""
import sys

import numpy as np

from dtest import *

LIBS = ["lib/inkling/dense_common.loom", "lib/inkling/dense_float.loom", "lib/inkling/dense_q8_0.loom", "lib/inkling/dense_abi.loom",
        "lib/inkling/dense_mm_wmma.loom", "lib/inkling/dense_wmma.loom", "lib/inkling/dense_targets.loom", "kernel/dense/gemm_q8_0.loom", "kernel/dense/gemm_f32.loom"]
ARGS = "%control: buffer, %count_origin: offset, %k_size: index, %n_rows: index, %w_origin: offset, %x_origin: offset, %x_stride: index, %d_origin: offset, %d_stride: index, %act: index, %cap: f32, %weight: buffer, %input: buffer, %output: buffer"
GLOBAL = (5, 11, 17, 23)
ROWS, CAP = 16, 16


def wvec(name):
    return load_weight("q8", name)[3][0]


def inputs_for(layer):
    """q, k, v inputs of a layer (the oracle norm tensors have no weight)."""
    if layer in GLOBAL:
        xq = oracle("A", "norm-%d" % layer, 2) * wvec("blk.%d.attn_q_inp_norm.weight" % layer)
        xk = oracle("A", "norm-%d" % layer, 0) * wvec("blk.%d.attn_k_inp_norm.weight" % layer)
        xv = oracle("A", "norm-%d" % layer, 1) * wvec("blk.%d.attn_v_inp_norm.weight" % layer)
        ffn_occ = 3
    else:
        xq = xk = xv = oracle("A", "norm-%d" % layer, 0) * wvec("blk.%d.attn_norm.weight" % layer)
        ffn_occ = 1
    xf = oracle("A", "norm-%d" % layer, ffn_occ) * wvec("blk.%d.ffn_norm.weight" % layer)
    return [a.astype(np.float32) for a in (xq, xk, xv, xf)]


def layer_case(layer, which):
    """One case with every projection of the layer. Returns the case and the (name, exact reference, oracle reference) list."""
    q8 = which == "q4"
    sym = "inkling_gemm_q8_0" if q8 else "inkling_gemm_bf16"
    xq, xk, xv, xf = inputs_for(layer)
    ops = []  # (key, weight name, input array, activation, oracle ref)
    ops.append(("q", "blk.%d.attn_q.weight" % layer, xq, 0, oracle("A", "proj_q-%d" % layer, 0)))
    ops.append(("k", "blk.%d.attn_k.weight" % layer, xk, 0, oracle("A", "conv_in-%d" % layer, 0)))
    ops.append(("v", "blk.%d.attn_v.weight" % layer, xv, 0, oracle("A", "conv_in-%d" % layer, 1)))
    ops.append(("r", "blk.%d.attn_rel.weight" % layer, xq, 0, oracle("A", "proj_r-%d" % layer, 0)))
    gate = load_weight(which, "blk.%d.attn_gate.weight" % layer)[3]
    sig = 1 / (1 + np.exp(-(xq.astype(np.float64) @ gate.astype(np.float64).T)))
    xo = (oracle("A", "kqv_out-%d" % layer, 0) * sig).astype(np.float32)
    ops.append(("g", "blk.%d.attn_gate.weight" % layer, xq, 1, None))
    ops.append(("o", "blk.%d.attn_output.weight" % layer, xo, 0, oracle("A", "conv_in-%d" % layer, 2)))
    inputs, outputs, body, preamble, refs = {"ctl": np.array([ROWS, 0, 0, 0], np.int32)}, {}, [], [], []
    decls = set()
    body.append("  %zero = check.literal value(0) : offset\n  %cap = check.literal value(30.0) : f32\n  %cols = check.literal value(" + str(CAP) + ") : index")
    for key, wname, x, act, oref in ops:
        raw, ttype, ne, wdeq = load_weight(which, wname)
        k, n = ne[0], ne[1]
        ref = x.astype(np.float64) @ wdeq.astype(np.float64).T
        if act == 1:
            ref = 1 / (1 + np.exp(-ref))
        xp = np.zeros((CAP, k), np.float32)
        xp[:ROWS] = x
        inputs["w_" + key], inputs["x_" + key] = raw.view(np.int8), xp
        outputs["y_" + key] = np.zeros((CAP, n), np.float32)
        body.append("  %%n_%s = check.literal value(%d) : index\n  %%k_%s = check.literal value(%d) : index\n  %%a_%s = check.literal value(%d) : index" % (key, n, key, k, key, act))
        body.append("  kernel.launch @%s[%%n_%s, %%cols](%%ctl, %%zero, %%k_%s, %%n_%s, %%zero, %%zero, %%k_%s, %%zero, %%n_%s, %%a_%s, %%cap, %%w_%s, %%x_%s, %%y_%s) : [index, index](tensor<4xi32>, offset, index, index, offset, offset, index, offset, index, index, f32, %s, %s, %s)" % (
            sym, key, key, key, key, key, key, key, key, key, tensor_type(inputs["w_" + key]), tensor_type(xp), tensor_type(outputs["y_" + key])))
        refs.append((key, ref, oref))
    # Router in F32 (the Q8_K_XL file has the router in F32 for every layer).
    if not q8:
        raw, ttype, ne, wdeq = load_weight(which, "blk.%d.ffn_gate_inp.weight" % layer)
        k, n = ne[0], ne[1]
        xp = np.zeros((CAP, k), np.float32)
        xp[:ROWS] = xf
        inputs["w_l"], inputs["x_l"] = raw.view(np.int8), xp
        outputs["y_l"] = np.zeros((CAP, n), np.float32)
        body.append("  %%n_l = check.literal value(%d) : index\n  %%k_l = check.literal value(%d) : index\n  %%a_l = check.literal value(0) : index" % (n, k))
        body.append("  kernel.launch @inkling_gemm_f32[%%n_l, %%cols](%%ctl, %%zero, %%k_l, %%n_l, %%zero, %%zero, %%k_l, %%zero, %%n_l, %%a_l, %%cap, %%w_l, %%x_l, %%y_l) : [index, index](tensor<4xi32>, offset, index, index, offset, offset, index, offset, index, index, f32, %s, %s, %s)" % (
            tensor_type(inputs["w_l"]), tensor_type(xp), tensor_type(outputs["y_l"])))
        refs.append(("l", xf.astype(np.float64) @ wdeq.astype(np.float64).T, oracle("A", "ffn_moe_logits-%d" % layer, 0)))
    pre = "\n".join("kernel.decl @%s(%%n_rows: index, %%cols_cap: index) launch(%s)" % (s, ARGS) for s in ("inkling_gemm_q8_0", "inkling_gemm_bf16", "inkling_gemm_f32"))
    return Case("layer%d_%s" % (layer, "q8_0" if q8 else "bf16"), LIBS, inputs, outputs, "\n".join(body), preamble=pre), refs


def main():
    arch = sys.argv[1]
    flt = sys.argv[2] if len(sys.argv) > 2 else ""
    rep = Report()
    plan = [(l, "q8") for l in (0, 1, 2, 3, 4, 5, 6, 8, 11, 12, 17, 18, 23)] + [(l, "q4") for l in (0, 5, 17, 23)]
    for layer, which in plan:
        name = "layer%d_%s" % (layer, "q8_0" if which == "q4" else "bf16")
        if flt and flt not in name:
            continue
        c, refs = layer_case(layer, which)
        out, err = run_case(c, arch, "/tmp/inkF_t/%s/%s" % (arch, name))
        if out is None:
            rep.add(name, arch, False, "", err.replace("\n", " ")[:600])
            continue
        parts, ok = [], True
        for key, ref, oref in refs:
            y = out["y_" + key][:ROWS]
            e = nmse(y, ref)
            tol = 1e-5 if which == "q4" else 1e-9
            good = e < tol
            s = "%s:%.1e" % (key, e)
            if oref is not None:
                ro = rel_l2(y, oref.reshape(ROWS, -1))
                good = good and ro < (5e-2 if which == "q4" else 6e-3)
                s += "/%.1e" % ro
            ok = ok and good
            parts.append(s)
        rep.add(name, arch, ok, "nmse_exact/oracle_relL2 " + " ".join(parts))
    rep.finish()


if __name__ == "__main__":
    main()
