"""RMS norm, per-head norm and embedding gather tests against the oracle (prompt A) and numpy.
Usage: python test_norm_embed.py ARCH [filter]
"""
import sys

import numpy as np

from dtest import *

LIBS = ["lib/inkling/dense_common.loom", "lib/inkling/dense_float.loom", "lib/inkling/dense_q8_0.loom", "lib/inkling/dense_abi.loom",
        "kernel/dense/norm.loom", "kernel/dense/embed.loom"]
DECLS = """kernel.decl @inkling_rms_norm(%rows: index) launch(%control: buffer, %count_origin: offset, %width: index, %x_origin: offset, %x_stride: index, %d_origin: offset, %d_stride: index, %epsilon: f32, %input: buffer, %weight: buffer, %output: buffer)
kernel.decl @inkling_rms_norm_plain(%rows: index) launch(%control: buffer, %count_origin: offset, %width: index, %x_origin: offset, %x_stride: index, %d_origin: offset, %d_stride: index, %epsilon: f32, %input: buffer, %output: buffer)
kernel.decl @inkling_rms_norm_cols(%rows: index) launch(%control: buffer, %count_origin: offset, %width: index, %origin: offset, %col_step: offset, %epsilon: f32, %input: buffer, %w0: buffer, %w1: buffer, %w2: buffer, %output: buffer)
kernel.decl @inkling_rms_norm_heads(%rows: index) launch(%control: buffer, %count_origin: offset, %width: index, %group: index, %x_origin: offset, %x_stride: index, %d_origin: offset, %d_stride: index, %epsilon: f32, %input: buffer, %output: buffer)
kernel.decl @inkling_embed_q8_0(%rows: index) launch(%control: buffer, %count_origin: offset, %vocab: index, %tokens: buffer, %tokens_origin: offset, %d_origin: offset, %d_stride: index, %epsilon: f32, %embedding: buffer, %norm_weight: buffer, %output: buffer)
kernel.decl @inkling_embed_bf16(%rows: index) launch(%control: buffer, %count_origin: offset, %vocab: index, %tokens: buffer, %tokens_origin: offset, %d_origin: offset, %d_stride: index, %epsilon: f32, %embedding: buffer, %norm_weight: buffer, %output: buffer)"""

PRE = """  %rows = check.literal value({cap}) : index
  %zero = check.literal value(0) : offset
  %width = check.literal value({width}) : index
  %stride = check.literal value({stride}) : index
  %eps = check.literal value(1.0e-6) : f32
"""
TYPES = "tensor<4xi32>, offset, index, offset, index, offset, index, f32"


def norm_case(name, x, w, ref, kind="rms_norm", rows_live=None, stride=None, group=128, tol=1e-10, oracle_ref=None):
    rows, width = x.shape
    live = rows if rows_live is None else rows_live
    stride = stride or width
    xp = np.zeros((rows, stride), np.float32)
    xp[:, :width] = x
    y0 = np.zeros((rows, stride), np.float32)
    ctl = np.array([live, 0, 0, 0], np.int32)
    ins = {"ctl": ctl, "x": xp}
    if kind == "rms_norm":
        ins["w"] = w
        call = "kernel.launch @inkling_rms_norm[%rows](%ctl, %zero, %width, %zero, %stride, %zero, %stride, %eps, %x, %w, %y) : [index](" + TYPES + ", {xt}, {wt}, {yt})"
    elif kind == "plain":
        call = "kernel.launch @inkling_rms_norm_plain[%rows](%ctl, %zero, %width, %zero, %stride, %zero, %stride, %eps, %x, %y) : [index](" + TYPES + ", {xt}, {yt})"
    else:
        call = ("%%group = check.literal value(%d) : index\n  " % group) + "kernel.launch @inkling_rms_norm_heads[%rows](%ctl, %zero, %width, %group, %zero, %stride, %zero, %stride, %eps, %x, %y) : [index](tensor<4xi32>, offset, index, index, offset, index, offset, index, f32, {xt}, {yt})"
    body = PRE.format(cap=rows, width=width, stride=stride) + "  " + call.format(xt=tensor_type(xp), wt=tensor_type(w) if w is not None else "", yt=tensor_type(y0))
    return Case(name, LIBS, ins, {"y": y0}, body, preamble=DECLS), ref, live, width, tol, oracle_ref


def cols_case(name, x3, ws, ref3, tol=1e-10):
    """x3 [3][rows][width]: the three columns of the row block, one weight each."""
    _, rows, width = x3.shape
    live = rows
    xp = np.ascontiguousarray(x3.reshape(3 * rows, width), np.float32)
    y0 = np.zeros_like(xp)
    ctl = np.array([live, 0, 0, 0], np.int32)
    ins = {"ctl": ctl, "x": xp, "w0": ws[0], "w1": ws[1], "w2": ws[2]}
    body = PRE.format(cap=rows, width=width, stride=width) + "  %%step = check.literal value(%d) : offset\n  " % (rows * width * 4) + "kernel.launch @inkling_rms_norm_cols[%rows](%ctl, %zero, %width, %zero, %step, %eps, %x, %w0, %w1, %w2, %y) : [index](" + TYPES.replace("offset, index, offset, index, offset, index, f32", "offset, index, offset, offset, f32") + ", " + ", ".join(tensor_type(a) for a in (xp, ws[0], ws[1], ws[2], y0)) + ")"
    return Case(name, LIBS, ins, {"y": y0}, body, preamble=DECLS), ref3.reshape(3 * rows, width), 3 * rows, width, tol, None


def embed_case(name, kind, table_raw, vocab, tokens, ref, tol=1e-9, oracle_ref=None, cap=None):
    rows = len(tokens)
    cap = cap or rows
    w = load_weight("q8", "token_embd_norm.weight")[3][0]
    tok = np.zeros(cap, np.int32)
    tok[:rows] = tokens
    ctl = np.array([rows, 0, 0, 0], np.int32)
    y0 = np.zeros((cap, 2304), np.float32)
    tab = table_raw.view(np.int8)
    body = """  %rows = check.literal value({cap}) : index
  %zero = check.literal value(0) : offset
  %vocab = check.literal value({vocab}) : index
  %stride = check.literal value(2304) : index
  %eps = check.literal value(1.0e-6) : f32
  kernel.launch @inkling_embed_{kind}[%rows](%ctl, %zero, %vocab, %tok, %zero, %zero, %stride, %eps, %tab, %w, %y) : [index](tensor<4xi32>, offset, index, {tt}, offset, offset, index, f32, {bt}, {wt}, {yt})""".format(
        cap=cap, vocab=vocab, kind=kind, tt=tensor_type(tok), bt=tensor_type(tab), wt=tensor_type(w), yt=tensor_type(y0))
    return Case(name, LIBS, {"ctl": ctl, "tok": tok, "tab": tab, "w": w}, {"y": y0}, body, preamble=DECLS), ref, rows, 2304, tol, oracle_ref


def rms(x, w=None, eps=1e-6):
    x = x.astype(np.float64)
    y = x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)
    return y if w is None else y * w.astype(np.float64)


def cases():
    out = []
    wq = lambda nm: load_weight("q8", nm)[3][0]
    tokw = wq("token_embd_norm.weight")
    cur0 = (oracle("A", "norm") * tokw).astype(np.float32)
    an0 = wq("blk.0.attn_norm.weight")
    out.append(norm_case("rms_weighted_attn_norm0", cur0, an0, rms(cur0, an0), oracle_ref=oracle("A", "norm-0", 0) * an0))
    out.append(norm_case("rms_plain_cur0", cur0, None, rms(cur0), kind="plain", oracle_ref=oracle("A", "norm-0", 0)))
    mix0 = oracle("A", "mix_out-0", 0).reshape(16, 2304)
    fn0 = wq("blk.0.ffn_norm.weight")
    out.append(norm_case("rms_weighted_ffn_norm0", mix0, fn0, rms(mix0, fn0), oracle_ref=oracle("A", "norm-0", 1) * fn0))
    mix5 = oracle("A", "mix_out-5", 0)  # [16][3][2304]
    for j, nm in enumerate(("q", "k", "v")):
        w5 = wq("blk.5.attn_%s_inp_norm.weight" % nm)
        xj = np.ascontiguousarray(mix5[:, j, :])
        out.append(norm_case("rms_weighted_global_%s_inp_norm5" % nm, xj, w5, rms(xj, w5), oracle_ref=oracle("A", "norm-5", {0: 2, 1: 0, 2: 1}[j]) * w5))
    vconv = oracle("A", "conv_out-0", 1)  # [16][1024]
    out.append(norm_case("rms_heads_v_norm0", vconv, None, rms(vconv.reshape(16, 8, 128)).reshape(16, 1024), kind="heads",
                         oracle_ref=oracle("A", "Vcur-0", 0).reshape(16, 1024)))
    ws5 = [wq("blk.5.attn_%s_inp_norm.weight" % nm) for nm in ("q", "k", "v")]
    x3 = np.ascontiguousarray(np.moveaxis(mix5, 1, 0))
    out.append(cols_case("rms_cols_global_inp_norm5", x3, ws5, np.stack([rms(x3[j], ws5[j]) for j in range(3)])))
    rng = np.random.default_rng(5)
    xc = (rng.standard_normal((3, 9, 2304)) * 2).astype(np.float32)
    out.append(cols_case("rms_cols_shared_weight_n9", xc, [an0] * 3, np.stack([rms(xc[j], an0) for j in range(3)])))
    xr = (rng.standard_normal((40, 2304)) * 3).astype(np.float32)
    out.append(norm_case("rms_weighted_n40_live25_stride2312", xr, an0, rms(xr, an0), rows_live=25, stride=2312))
    out.append(norm_case("rms_plain_n1", xr[:1], None, rms(xr[:1]), kind="plain"))
    out.append(norm_case("rms_heads_n7_group128", xr[:7, :1024], None, rms(xr[:7, :1024].reshape(7, 8, 128)).reshape(7, 1024), kind="heads"))
    toks = oracle_tokens("A")
    raw8, t8, ne8, deq8 = load_weight("q8", "token_embd.weight")
    out.append(embed_case("embed_bf16_full_table_n16", "bf16", raw8, ne8[1], toks, rms(deq8[toks], tokw), oracle_ref=oracle("A", "norm") * tokw))
    raw4, t4, ne4, deq4 = load_weight("q4", "token_embd.weight")
    out.append(embed_case("embed_q8_0_full_table_n16", "q8_0", raw4, ne4[1], toks, rms(deq4[toks], tokw), oracle_ref=None))
    tk = np.array([200057, 0, 1, 199999, 201023, 12345, 77777, 150000, 99999], np.int32)
    out.append(embed_case("embed_q8_0_edge_ids_n9_cap16", "q8_0", raw4, ne4[1], tk, rms(deq4[tk], tokw), cap=16))
    out.append(embed_case("embed_bf16_edge_ids_n9_cap16", "bf16", raw8, ne8[1], tk, rms(deq8[tk], tokw), cap=16))
    return out


def main():
    arch = sys.argv[1]
    flt = sys.argv[2] if len(sys.argv) > 2 else ""
    rep = Report()
    for c, ref, live, width, tol, oref in cases():
        if flt and flt not in c.name:
            continue
        out, err = run_case(c, arch, "/tmp/inkF_t/%s/%s" % (arch, c.name))
        if out is None:
            rep.add(c.name, arch, False, "", err.replace("\n", " ")[:600])
            continue
        y = out["y"]
        e = nmse(y[:live, :width], ref[:live])
        pad_ok = bool(np.all(y[live:] == 0)) and bool(np.all(y[:live, width:] == 0))
        m = "nmse=%.2e untouched_ok=%s" % (e, pad_ok)
        ok = e < tol and pad_ok
        if oref is not None:
            ro = nmse(y[:live, :width], oref.reshape(live, width))
            m += " oracle_nmse=%.2e" % ro
            ok = ok and ro < 1e-9
        rep.add(c.name, arch, ok, m)
    rep.finish()


if __name__ == "__main__":
    main()
