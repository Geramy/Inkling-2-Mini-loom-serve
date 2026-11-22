"""Output head tests: the chain of the real dispatch (head_select, head_norm, lm_head with soft cap 30) against numpy and the oracle, and lm_head against the oracle result_output.
The dispatch launches the selection and norm kernels with a launch size that is a compile-time constant (spans capacity from the config); head.loom needs this for its bounds
proofs and does not compile without it. A check.case cannot give a kernel a constant launch argument, so the tests link a copy of head.loom in which the two lines that
read the launch size are replaced by the constant (head_source). No other line of the kernel changes.
Usage: python test_head.py ARCH [filter]
"""
import os
import sys

import numpy as np

from dtest import *

LIBS = ["lib/inkling/dense_common.loom", "lib/inkling/dense_float.loom", "lib/inkling/dense_q8_0.loom", "lib/inkling/dense_abi.loom",
        "lib/inkling/dense_vec.loom", "lib/inkling/dense_mm_wmma.loom", "lib/inkling/dense_wmma.loom", "lib/inkling/dense_targets.loom",
        "lib/vendor/packet.loom", "kernel/dense/norm.loom", "kernel/dense/head.loom", "kernel/dense/gemv.loom", "kernel/dense/gemm_q8_0.loom", "kernel/dense/gemm_f32.loom",
        "kernel/dense/gemm_bf16_wmma.loom"]
LAUNCH_SIZE_LINES = ["%cap = index.assume %spans_cap [range(%spans_cap, 1, 2147483646)] : index", "%cap = index.assume %spans_cap [range(%spans_cap, 1, 65536)] : index"]
VOCAB, STRIDE, HIDDEN = 200058, 200064, 2304
LM_ARGS = "%control: buffer, %count_origin: offset, %k_size: index, %n_rows: index, %w_origin: offset, %x_origin: offset, %x_stride: index, %d_origin: offset, %d_stride: index, %act: index, %cap: f32, %weight: buffer, %input: buffer, %output: buffer"
DECLS = """kernel.decl @inkling_head_select(%spans_cap: index) launch(%spans_cap: index, %all_rows: index, %metadata: buffer, %sel: buffer, %sel_origin: offset)
kernel.decl @inkling_head_norm(%spans_cap: index) launch(%sel: buffer, %sel_origin: offset, %spans_cap: index, %width: index, %x_origin: offset, %x_stride: index, %d_origin: offset, %d_stride: index, %epsilon: f32, %logit_valid: index, %logit_stride: index, %logits: buffer, %logits_origin: offset, %input: buffer, %weight: buffer, %output: buffer)
kernel.decl @inkling_gemv_q8_0(%n_rows: index, %k_size: index, %per_unit: index) launch({a})
kernel.decl @inkling_gemv_bf16(%n_rows: index, %k_size: index, %per_unit: index) launch({a})
kernel.decl @inkling_gemm_q8_0(%n_rows: index, %cols_cap: index) launch({a})
kernel.decl @inkling_gemm_wmma_bf16(%n_rows: index, %cols_cap: index) launch({a})""".format(a=LM_ARGS)
PER_UNIT_LINE = "  %%per_unit = check.literal value(%d) : index\n" % {"gfx1151": 4}.get(sys.argv[1], 2)
LM_TYPES = "{st}, offset, index, index, offset, offset, index, offset, index, index, f32, {wt}, {xt}, {lt}"


def head_source(cap):
    """Path of a copy of head.loom whose launch size is the constant cap."""
    text = open(os.path.join(PKG, "kernel/dense/head.loom"), encoding="utf-8").read()
    for line in LAUNCH_SIZE_LINES:
        assert text.count(line) == 1, line
        text = text.replace(line, "%%cap = index.constant %d : index" % cap)
    path = "/tmp/inkF_t/%s/head_cap%d.loom" % (sys.argv[1], cap)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w", encoding="utf-8").write(text)
    return path


def lm_launch(kind, path, t):
    """The lm_head launch as the dispatch hook writes it: gemv for at most 8 selected rows, gemm above."""
    name = "inkling_gemv_%s" % kind if path == "gemv" else ("inkling_gemm_wmma_bf16" if kind == "bf16" else "inkling_gemm_q8_0")
    work, wtypes = ("[%n_rows, %width, %per_unit]", "[index, index, index]") if path == "gemv" else ("[%n_rows, %cap]", "[index, index]")
    return "  kernel.launch @%s%s(%%sel, %%zero, %%width, %%n_rows, %%zero, %%zero, %%width, %%zero, %%lstride, %%act, %%softcap, %%w, %%xs, %%logits) : %s(%s)" % (
        name, work, wtypes, LM_TYPES.format(**t))


def chain_body(cap, kind, path, t):
    return """{pu}  %all_rows = check.literal value(0) : index
  %cap = check.literal value({cap}) : index
  %zero = check.literal value(0) : offset
  %width = check.literal value({hidden}) : index
  %eps = check.literal value(1.0e-6) : f32
  %valid = check.literal value({vocab}) : index
  %lstride = check.literal value({stride}) : index
  %n_rows = check.literal value({vocab}) : index
  %act = check.literal value(2) : index
  %softcap = check.literal value(30.0) : f32
  kernel.launch @inkling_head_select[%cap](%cap, %all_rows, %meta, %sel, %zero) : [index](index, index, {mt}, {st}, offset)
  kernel.launch @inkling_head_norm[%cap](%sel, %zero, %cap, %width, %zero, %width, %zero, %width, %eps, %valid, %lstride, %logits, %zero, %cur, %wn, %xs) : [index]({st}, offset, index, index, offset, index, offset, index, f32, index, index, {lt}, offset, {ct}, {wnt}, {xt})
{lm}""".format(cap=cap, pu=PER_UNIT_LINE if path == "gemv" else "", hidden=HIDDEN, vocab=VOCAB, stride=STRIDE, lm=lm_launch(kind, path, t), **t)


def plan(lengths, flags, pk):
    """Packet words: header (live tokens, spans, chosen count), 24 bytes per span, then the selection list (the rows to emit); pk is the packet capacity."""
    first = 3 + 6 * pk
    meta = np.zeros(first + pk, np.int32)
    off, k = 0, 0
    for n, f in zip(lengths, flags):
        if f & 1:
            meta[first + k] = off + n - 1
            k += 1
        off += n
    meta[0], meta[1], meta[2] = off, len(lengths), k
    return meta, off


def head_case(name, kind, which, lengths, flags, cap, path, tol, pk=None, seed=3, last_row=None):
    """kind: bf16 or q8_0 weights; path: gemv or gemm; cap: the span capacity; pk: the packet capacity (default cap)."""
    pk = pk or cap
    rng = np.random.default_rng(seed)
    meta, total = plan(lengths, flags, pk)
    cur = (rng.standard_normal((total, HIDDEN)) * 2.5).astype(np.float32)
    w_out = load_weight(which, "output_norm.weight")[3][0]
    if last_row == "oracle":
        cur[-1] = oracle("A", "result_norm").reshape(HIDDEN) / w_out
    raw, ttype, ne, wdeq = load_weight(which, "output.weight")
    sel_rows, sel_spans, off = [], [], 0
    for n, f in zip(lengths, flags):
        if f & 1:
            sel_spans.append(len(sel_rows))
            sel_rows.append(off + n - 1)
        off += n
    xn = cur[sel_rows].astype(np.float64)
    xn = xn / np.sqrt((xn * xn).mean(-1, keepdims=True) + 1e-6) * w_out
    z = xn @ wdeq[:VOCAB].astype(np.float64).T
    ref = 30.0 * np.tanh(z / 30.0)
    sel0 = np.zeros(4 + 2 * cap, np.int32)
    xs0 = np.zeros((cap, HIDDEN), np.float32)
    logits0 = np.zeros((cap, STRIDE), np.float32)
    w = raw.view(np.int8)
    t = dict(mt=tensor_type(meta), st=tensor_type(sel0), lt=tensor_type(logits0), ct=tensor_type(cur), wnt=tensor_type(w_out), xt=tensor_type(xs0), wt=tensor_type(w))
    files = [head_source(cap) if f == "kernel/dense/head.loom" else f for f in LIBS]
    c = Case(name, files, {"meta": meta, "cur": cur, "wn": w_out, "w": w}, {"sel": sel0, "xs": xs0, "logits": logits0}, chain_body(cap, kind, path, t), preamble=DECLS,
             config={"inkling.packet_capacity": pk})
    return c, dict(n_sel=len(sel_rows), spans=sel_spans, rows=sel_rows, ref=ref, tol=tol, oracle=last_row == "oracle")


def check_head(c, info, out):
    sel, logits = out["sel"], out["logits"]
    n = info["n_sel"]
    ok = int(sel[0]) == n and list(sel[4:4 + 2 * n:2]) == info["spans"] and list(sel[5:5 + 2 * n:2]) == info["rows"]
    got = logits[:n, :VOCAB]
    e = nmse(got, info["ref"])
    pad = bool(np.all(np.isneginf(logits[:n, VOCAB:])))
    rest = bool(np.all(logits[n:] == 0))
    top = bool(np.all(np.argmax(got, -1) == np.argmax(info["ref"], -1)))
    extra = ""
    if info["oracle"]:
        ref_out = oracle("A", "result_output").reshape(-1)
        rn = nmse(out["xs"][n - 1:n], oracle("A", "result_norm").reshape(1, HIDDEN))
        ro = rel_l2(got[n - 1], ref_out)
        top5 = set(np.argsort(-got[n - 1])[:5]) == set(np.argsort(-ref_out)[:5])
        extra = " result_norm_nmse=%.1e result_output_relL2=%.1e top5_equal=%s" % (rn, ro, top5)
        top = top and rn < 1e-9 and ro < info["oracle_tol"] and top5 and int(np.argmax(got[n - 1])) == int(np.argmax(ref_out))
    return ok and pad and rest and top and e < info["tol"], "sel_ok=%s pad_inf=%s rest_zero=%s argmax_equal=%s nmse=%.2e%s" % (ok, pad, rest, top, e, extra)


def lm_oracle_case(name, kind, which, path, tol):
    x = oracle("A", "result_norm").reshape(1, HIDDEN).astype(np.float32)
    ref = oracle("A", "result_output").reshape(1, VOCAB)
    raw, ttype, ne, wdeq = load_weight(which, "output.weight")
    z = x.astype(np.float64) @ wdeq[:VOCAB].astype(np.float64).T
    exact = 30.0 * np.tanh(z / 30.0)
    cap = 8
    xp = np.zeros((cap, HIDDEN), np.float32)
    xp[:1] = x
    ctl = np.array([1, 0, 0, 0], np.int32)
    y0 = np.zeros((cap, STRIDE), np.float32)
    w = raw.view(np.int8)
    t = dict(st=tensor_type(ctl), wt=tensor_type(w), xt=tensor_type(xp), lt=tensor_type(y0))
    per_unit = PER_UNIT_LINE if path == "gemv" else ""
    lm = lm_launch(kind, path, t).replace("%sel", "%ctl").replace("%xs", "%xp").replace("%logits", "%y")
    body = """{pu}  %zero = check.literal value(0) : offset
  %width = check.literal value({hidden}) : index
  %lstride = check.literal value({stride}) : index
  %n_rows = check.literal value({vocab}) : index
  %cap = check.literal value({cap}) : index
  %act = check.literal value(2) : index
  %softcap = check.literal value(30.0) : f32
{lm}""".format(pu=per_unit, hidden=HIDDEN, stride=STRIDE, vocab=VOCAB, cap=cap, lm=lm)
    c = Case(name, LIBS, {"ctl": ctl, "w": w, "xp": xp}, {"y": y0}, body, preamble=DECLS)
    return c, exact, ref, tol


def main():
    arch = sys.argv[1]
    flt = sys.argv[2] if len(sys.argv) > 2 else ""
    rep = Report()
    sel3 = ([5, 1, 3, 1, 4], [1, 0, 1, 1, 0])
    sel12 = ([2] * 14, [1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1])
    cases = [
        head_case("head_bf16_gemv_nsel3", "bf16", "q8", *sel3, 8, "gemv", 1e-9),
        head_case("head_q8_0_gemv_nsel3", "q8_0", "q4", *sel3, 8, "gemv", 1e-9),
        head_case("head_bf16_gemv_nsel1_decode", "bf16", "q8", [1], [1], 4, "gemv", 1e-9),
        head_case("head_bf16_gemv_packet512", "bf16", "q8", *sel3, 8, "gemv", 1e-9, pk=512),
        head_case("head_q8_0_gemv_packet512", "q8_0", "q4", *sel3, 8, "gemv", 1e-9, pk=512),
        head_case("head_bf16_gemm_nsel12", "bf16", "q8", *sel12, 16, "gemm", 5e-6),
        head_case("head_q8_0_gemm_nsel12", "q8_0", "q4", *sel12, 16, "gemm", 1e-5),
        head_case("head_oracle_chain_bf16_gemv", "bf16", "q8", [16], [1], 4, "gemv", 1e-9, last_row="oracle"),
        head_case("head_oracle_chain_q8_0_gemv", "q8_0", "q4", [16], [1], 4, "gemv", 1e-9, last_row="oracle"),
        head_case("head_oracle_chain_bf16_spans", "bf16", "q8", [5, 1, 3, 16], [1, 0, 1, 1], 8, "gemv", 1e-9, pk=512, last_row="oracle"),
        head_case("head_oracle_chain_q8_0_spans", "q8_0", "q4", [5, 1, 3, 16], [1, 0, 1, 1], 8, "gemv", 1e-9, pk=512, last_row="oracle"),
    ]
    for c, info in cases:
        if flt and flt not in c.name:
            continue
        info["oracle_tol"] = 3e-2 if "q8_0" in c.name else 5e-3
        out, err = run_case(c, arch, "/tmp/inkF_t/%s/%s" % (arch, c.name))
        if out is None:
            rep.add(c.name, arch, False, "", err.replace("\n", " ")[:700])
            continue
        ok, m = check_head(c, info, out)
        rep.add(c.name, arch, ok, m)
    for name, kind, which, path, tol in (("lm_head_oracle_bf16_gemv", "bf16", "q8", "gemv", 5e-3), ("lm_head_oracle_q8_0_gemv", "q8_0", "q4", "gemv", 2e-2),
                                         ("lm_head_oracle_bf16_gemm", "bf16", "q8", "gemm", 5e-3), ("lm_head_oracle_q8_0_gemm", "q8_0", "q4", "gemm", 2e-2)):
        if flt and flt not in name:
            continue
        c, exact, ref, tol = lm_oracle_case(name, kind, which, path, tol)
        out, err = run_case(c, arch, "/tmp/inkF_t/%s/%s" % (arch, name))
        if out is None:
            rep.add(name, arch, False, "", err.replace("\n", " ")[:700])
            continue
        y = out["y"][:1, :VOCAB]
        e = nmse(y, exact)
        ro = rel_l2(y, ref)
        t1 = int(np.argmax(y)) == int(np.argmax(ref))
        top5 = set(np.argsort(-y[0])[:5]) == set(np.argsort(-ref[0])[:5])
        rep.add(name, arch, ro < tol and t1, "nmse_vs_exact=%.2e oracle_relL2=%.2e argmax_equal=%s top5_equal=%s" % (e, ro, t1, top5))
    rep.finish()


if __name__ == "__main__":
    main()
