#!/usr/bin/env python3
"""Writes the hook implementations package/dispatch/hooks_{dense,attn,mix,experts}.loom (docs/DISPATCH.md, docs/INTEGRATION_LOG.md). Run with python3 from anywhere."""
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "package", "dispatch")
BUFS = "parameters span_table token_map metadata origins arena cur blocks bank x xn small q k v r gate ab ob y".split()
SIG = ", ".join("%%%s: buffer" % b for b in BUFS)
ARGS = ", ".join("%" + b for b in BUFS)
TYPES20 = ", ".join(["buffer"] * 20)
E = 2304
EPS = "9.9999999999999995e-07"


class Body:
    def __init__(self):
        self.header = []
        self.known = {}
        self.lines = []
        self.n = 0

    def once(self, key, line):
        if key not in self.known:
            self.known[key] = True
            self.header.append("  " + line)
        return key

    def i(self, v):
        return self.once("%%i%d" % v, "%%i%d = index.constant %d : index" % (v, v))

    def o(self, v):
        return self.once("%%o%d" % v, "%%o%d = index.constant %d : offset" % (v, v))

    def f(self, name, v):
        return self.once("%" + name, "%%%s = scalar.constant %s : f32" % (name, v))

    def cfg(self, name, key):
        return self.once("%" + name, "%%%s = config.get @%s : index" % (name, key))

    def cap(self):
        return self.cfg("cap", "inkling.tokens")

    def span_cap(self):
        return self.cfg("span_cap", "inkling.spans")

    def ctx(self):
        return self.cfg("ctx", "inkling.context")

    def pool(self):
        return self.cfg("pool", "inkling.pool")

    def dense(self):
        return self.cfg("dense", "inkling.dense_type")

    def op(self, name, text):
        return self.once("%" + name, "%%%s = %s" % (name, text))

    def mul(self, name, a, b):
        return self.op(name, "index.mul %s, %s : index" % (a, b))

    def add(self, name, a, b):
        return self.op(name, "index.add %s, %s : index" % (a, b))

    def addo(self, name, a, b):
        return self.op(name, "index.add %s, %s : offset" % (a, b))

    def div(self, name, a, b):
        return self.op(name, "index.div %s, %s : index" % (a, b))

    def rem(self, name, a, b):
        return self.op(name, "index.rem %s, %s : index" % (a, b))

    def cmp(self, name, pred, a, b):
        return self.op(name, "index.cmp %s, %s, %s : index" % (pred, a, b))

    def scale(self, name, a, b):
        return self.op(name, "index.scale %s, %s : index, offset -> offset" % (a, b))

    def flag(self, name, cond):
        return self.op(name, "scf.select %s, %s, %s : index" % (cond, self.i(1), self.i(0)))

    def cE(self):
        return self.mul("cE", self.cap(), self.i(E))

    def colbytes(self):
        return self.scale("colbytes", self.cap(), self.o(E * 4))

    def col(self, c):
        if c == 0:
            return self.o(0)
        return self.scale("col%d" % c, self.i(c), self.colbytes())

    def emit(self, text):
        self.lines.append("  " + text)

    def tmp(self, p):
        self.n += 1
        return "%%%s%d" % (p, self.n)

    def weight(self, name, vt=None, indexed=True):
        vt = vt or WV[name.replace("blk.{}.", "")]
        w, wb = self.tmp("w"), self.tmp("wb")
        suffix = "[%layer]" if indexed else ""
        self.emit('%s = command.parameter %%parameters, "%s"%s : view<%s>' % (w, name, suffix, vt))
        self.emit("%s = command.parameter.base %s : view<%s> -> buffer" % (wb, w, vt))
        return wb

    def pick(self, cond, arms, indexed=True):
        """arms: (then, else) lists of (name, view type); returns the buffer chosen by cond"""
        r = self.tmp("pick")
        self.emit("%s = scf.if %s -> (buffer) {" % (r, cond))
        for i, (name, vt) in enumerate(arms):
            if i:
                self.emit("} else {")
            wb = self.weight(name, vt, indexed)
            self.emit("  scf.yield %s : buffer" % wb)
        self.emit("}")
        return r


WV = {
    "attn_norm.weight": "2304xf32", "attn_q_inp_norm.weight": "2304xf32", "attn_k_inp_norm.weight": "2304xf32", "attn_v_inp_norm.weight": "2304xf32",
    "ffn_norm.weight": "2304xf32", "ffn_res_query.weight": "2304xf32", "attn_q_clip.weight": "3072xf32", "attn_rel_clip.weight": "384xf32",
    "attn_k_conv.weight": "4096xf32", "attn_v_conv.weight": "4096xf32", "attn_rel_bias.weight": "16384xf32", "attn_post_conv.weight": "9216xf32",
    "ffn_post_conv.weight": "9216xf32", "ffn_gate_inp.weight": "294912xf32", "exp_probs_b.bias": "128xf32", "ffn_gate_up_exps.scale": "128xf32",
    "ffn_down_exps.scale": "128xf32", "attn_sink_proj.weight": "24x2304xbf16", "attn_k_conv_proj.weight": "32x2304xbf16", "attn_v_conv_proj.weight": "32x2304xbf16",
}


def klaunch(b, kernel, work, args):
    w = "[%s]" % ", ".join(work) if work else ""
    wt = "[%s]" % ", ".join(["index"] * len(work)) if work else ""
    b.emit("kernel.launch @%s%s(%s) : %s(%s)" % (kernel, w, ", ".join(a for a, _ in args), wt, ", ".join(t for _, t in args)))


def hook(name, b, sig=SIG, params="%layer: index, %visit: index, %pass: index"):
    return "command.def public target(@inkling_execution) @inkling_hook_%s(%s) launch(%s) {\n%s\n%s\n  command.return\n}\n\n" % (name, params, sig, "\n".join(b.header), "\n".join(b.lines))


def lines(*ls):
    return list(ls)


# ---------------------------------------------------------------- dense

DENSE_T = ["buffer", "offset", "index", "index", "offset", "offset", "index", "offset", "index", "index", "f32", "buffer", "buffer", "buffer"]


def gemm_call(b, kernel, work, k, n, xbuf, xcol, obuf, dstride, act, wb, ctrl="%metadata"):
    args = [ctrl, b.o(0), b.i(k), b.i(n), b.o(0), xcol, b.i(k), b.o(0), b.i(dstride), b.i(act), b.f("c30", "30.0"), wb, xbuf, obuf]
    klaunch(b, kernel, work, list(zip(args, DENSE_T)))


def linear(b, wname, k, n, xbuf, xcol, obuf, dstride, act=0, kind="dense", ctrl="%metadata", rows=None, indexed=True, nw=None):
    """Projection of the live rows: GEMV for stage capacity up to 8 rows, GEMM above. ctrl holds the live count, rows is the row capacity, nw the weight rows."""
    cap = rows or b.cap()
    nw = nw or n
    small = b.tmp("small")
    b.emit("%s = index.cmp ule, %s, %s : index" % (small, cap, b.i(8)))
    n_i, k_i = b.i(n), b.i(k)
    pu = b.op("per_unit", "template.apply<@inkling.dense.gemv.xs_wg_per_unit>() : () -> (index)")
    if kind == "dense":
        is_q8 = b.tmp("isq8")
        b.emit("%s = index.cmp eq, %s, %s : index" % (is_q8, b.dense(), b.i(0)))
        wb = b.pick(is_q8, [(wname, "%dx%dx34xi8" % (nw, k // 32)), (wname, "%dx%dxbf16" % (nw, k))], indexed)
        b.emit("scf.if %s {" % small)
        b.emit("  scf.if %s {" % is_q8)
        gemm_call(b, "inkling_gemv_q8_0", [n_i, k_i, pu], k, n, xbuf, xcol, obuf, dstride, act, wb, ctrl)
        b.emit("  } else {")
        gemm_call(b, "inkling_gemv_bf16", [n_i, k_i, pu], k, n, xbuf, xcol, obuf, dstride, act, wb, ctrl)
        b.emit("  }")
        b.emit("} else {")
        b.emit("  scf.if %s {" % is_q8)
        gemm_call(b, "inkling_gemm_q8_0", [n_i, cap], k, n, xbuf, xcol, obuf, dstride, act, wb, ctrl)
        b.emit("  } else {")
        gemm_call(b, "inkling_gemm_wmma_bf16", [n_i, cap], k, n, xbuf, xcol, obuf, dstride, act, wb, ctrl)
        b.emit("  }")
        b.emit("}")
    else:
        wb = b.weight(wname, "%dx%dxbf16" % (n, k))
        b.emit("scf.if %s {" % small)
        gemm_call(b, "inkling_gemv_split_bf16", [n_i], k, n, xbuf, xcol, obuf, dstride, act, wb, ctrl)
        b.emit("} else {")
        gemm_call(b, "inkling_gemm_bf16", [n_i, cap], k, n, xbuf, xcol, obuf, dstride, act, wb, ctrl)
        b.emit("}")


def rms_norm(b, wname, xbuf, xcol, obuf, ocol, indexed=True):
    wb = b.weight(wname, None, indexed)
    cap = b.cap()
    args = ["%metadata", b.o(0), b.i(E), xcol, b.i(E), ocol, b.i(E), b.f("eps", EPS), xbuf, wb, obuf]
    types = ["buffer", "offset", "index", "offset", "index", "offset", "index", "f32", "buffer", "buffer", "buffer"]
    klaunch(b, "inkling_rms_norm", [cap], list(zip(args, types)))


def rms_norm_cols(b, wnames, xbuf, obuf):
    """One launch for the columns 0 to 2 of the packed row block, with one weight for each column."""
    made = {}
    for n in wnames:
        if n not in made:
            made[n] = b.weight(n)
    ws = [made[n] for n in wnames]
    args = ["%metadata", b.o(0), b.i(E), b.o(0), b.colbytes(), b.f("eps", EPS), xbuf] + ws + [obuf]
    types = ["buffer", "offset", "index", "offset", "offset", "f32", "buffer", "buffer", "buffer", "buffer", "buffer"]
    klaunch(b, "inkling_rms_norm_cols", [b.cap()], list(zip(args, types)))


def is_global(b):
    return b.op("g", "func.call pure @inkling_layer_is_global(%layer) : (index) -> (i1)")


def embed_hook():
    b = Body()
    cap = b.cap()
    is_q8 = b.cmp("isq8", "eq", b.dense(), b.i(0))
    nw = b.weight("token_embd_norm.weight", "2304xf32", False)
    wb = b.pick(is_q8, [("token_embd.weight", "201024x72x34xi8"), ("token_embd.weight", "201024x2304xbf16")], False)
    eps = b.f("eps", EPS)
    args = ["%metadata", b.o(0), b.i(201024), "%tokens", b.o(0), b.o(0), b.i(E), eps, wb, nw, "%cur"]
    types = ["buffer", "offset", "index", "buffer", "offset", "offset", "index", "f32", "buffer", "buffer", "buffer"]
    b.emit("scf.if %s {" % is_q8)
    klaunch(b, "inkling_embed_q8_0", [cap], list(zip(args, types)))
    b.emit("} else {")
    klaunch(b, "inkling_embed_bf16", [cap], list(zip(args, types)))
    b.emit("}")
    return hook("embed", b, "%parameters: buffer, %metadata: buffer, %tokens: buffer, %embd: buffer, %cur: buffer", "")


def head_hook():
    b = Body()
    cap, sc = b.cap(), b.span_cap()
    capid = b.cfg("capid", "inkling.capture_id")
    is30 = b.cmp("is30", "eq", capid, b.i(30))
    is31 = b.cmp("is31", "eq", capid, b.i(31))
    every = b.op("every", "scalar.ori %s, %s : i1" % (is30, is31))
    all_rows = b.flag("all_rows", every)
    rows = b.op("sel_rows", "scf.select %s, %s, %s : index" % (every, cap, sc))
    sel_bytes = b.addo("sel_bytes", b.op("sel_words", "index.scale %s, %s : index, offset -> offset" % (rows, b.o(8))), b.o(16))
    logit_bytes = b.scale("logit_bytes", rows, b.o(800256))
    b.emit("%%sel = buffer.alloca<global> align(256) %s : buffer" % sel_bytes)
    b.emit("%%logits = buffer.alloca<global> align(256) %s : buffer" % logit_bytes)
    b.emit("%%scratch = buffer.alloca<global> align(256) %s : buffer" % b.scale("scratch_bytes", rows, b.o(2560)))
    query = b.weight("output_res_query.weight", "2304xf32", False)
    mix_states(b, b.i(4), b.i(1), 0, query, "%x")
    klaunch(b, "inkling_head_select", [rows], list(zip([rows, all_rows, "%metadata", "%sel", b.o(0)], ["index", "index", "buffer", "buffer", "offset"])))
    nw = b.weight("output_norm.weight", "2304xf32", False)
    args = ["%sel", b.o(0), rows, b.i(E), b.o(0), b.i(E), b.o(0), b.i(E), b.f("eps", EPS), b.i(200058), b.i(200064), "%logits", b.o(0), "%x", nw, "%xn"]
    types = ["buffer", "offset", "index", "index", "offset", "index", "offset", "index", "f32", "index", "index", "buffer", "offset", "buffer", "buffer", "buffer"]
    klaunch(b, "inkling_head_norm", [rows], list(zip(args, types)))
    linear(b, "output.weight", E, 200058, "%xn", b.o(0), "%logits", 200064, act=2, ctrl="%sel", rows=rows, indexed=False, nw=201024)
    for cid, src, width in ((30, "%xn", E), (31, "%logits", 200064)):
        b.emit("scf.if %s {" % b.cmp("cap%d" % cid, "eq", capid, b.i(cid)))
        klaunch(b, "inkling_capture_rows", [], list(zip([b.i(width), b.i(0), b.i(0), "%metadata", src, "%residual"], ["index", "index", "index", "buffer", "buffer", "buffer"])))
        b.emit("}")
    sample_args = [rows, b.i(200064), "%metadata", "%logits"]
    sample_types = ["index", "index", "buffer", "buffer"]
    klaunch(b, "inkling_sample_stats", [rows], list(zip(sample_args + ["%scratch"], sample_types + ["buffer"])))
    klaunch(b, "inkling_sample_draw", [rows], list(zip(sample_args + ["%scratch"], sample_types + ["buffer"])))
    klaunch(b, "inkling_sample_rows", [rows], list(zip(sample_args + ["%outputs", "%scratch"], sample_types + ["buffer", "buffer"])))
    return hook("head", b, SIG + ", %outputs: buffer, %residual: buffer", "")


def attn_group(b, g, ctl):
    """One launch for the projections of an attention visit (inkling_gemv_attn_group): q, rel, gate, k, v, sink and, on global visits, the two dynamic conv projections."""
    q8 = lambda name, n: b.weight(name, "%dx%dx34xi8" % (n, E // 32))
    ws = [q8("blk.{}.attn_q.weight", 3072), q8("blk.{}.attn_rel.weight", 384), q8("blk.{}.attn_gate.weight", 3072)]
    sink = b.weight("blk.{}.attn_sink_proj.weight")
    kv_col = b.op("kv_col", "scalar.ori %s, %s : i1" % (g, ctl))
    xk = b.op("xo_k", "scf.select %s, %s, %s : offset" % (kv_col, b.col(1), b.o(0)))
    xv = b.op("xo_v", "scf.select %s, %s, %s : offset" % (kv_col, b.col(2), b.o(0)))
    pu = b.op("per_unit", "template.apply<@inkling.dense.group.wg_per_unit>() : () -> (index)")

    def launch(conv_on, wk, wv, kc, vc):
        args = [conv_on, "%metadata", b.o(0), b.i(E), b.o(0), xk, xv] + ws + [wk, wv, sink, kc, vc, "%xn", "%q", "%r", "%gate", "%k", "%v", "%small", "%ob", "%y"]
        types = ["index", "buffer", "offset", "index", "offset", "offset", "offset"] + ["buffer"] * 17
        klaunch(b, "inkling_gemv_attn_group", [conv_on, pu], list(zip(args, types)))

    b.emit("scf.if %s {" % g)
    wk, wv = q8("blk.{}.attn_k.weight", 1024), q8("blk.{}.attn_v.weight", 1024)
    launch(b.i(1), wk, wv, b.weight("blk.{}.attn_k_conv_proj.weight"), b.weight("blk.{}.attn_v_conv_proj.weight"))
    b.emit("} else {")
    wk, wv = q8("blk.{}.attn_k.weight", 1024), q8("blk.{}.attn_v.weight", 1024)
    launch(b.i(0), wk, wv, sink, sink)
    b.emit("}")


def dense_file():
    out = []
    out.append("""config.decl @inkling.dense_type : %value: index where [range(%value, 0, 1)]

template.decl @inkling.dense.gemv.xs_wg_per_unit() -> (index)

template.decl @inkling.dense.group.wg_per_unit() -> (index)

// Dense area hooks (embedding, head, projections and norms).
""".replace("@SIG@", SIG))
    out.append(embed_hook())
    out.append(head_hook())

    # norm_in
    b = Body()
    g = is_global(b)
    is0 = b.cmp("is0", "eq", "%layer", b.i(0))
    b.emit("scf.if %s {" % is0)
    rms_norm(b, "blk.{}.attn_norm.weight", "%cur", b.o(0), "%xn", b.o(0))
    b.emit("} else {")
    b.emit("scf.if %s {" % g)
    rms_norm_cols(b, ["blk.{}.attn_q_inp_norm.weight", "blk.{}.attn_k_inp_norm.weight", "blk.{}.attn_v_inp_norm.weight"], "%x", "%xn")
    b.emit("} else {")
    b.emit("scf.if %s {" % controlled(b))
    rms_norm_cols(b, ["blk.{}.attn_norm.weight"] * 3, "%x", "%xn")
    b.emit("}")
    b.emit("}")
    b.emit("}")
    out.append(hook("norm_in", b))
    # proj
    b = Body()
    g = is_global(b)
    xn = "%xn"
    use_group = b.op("use_group", "scalar.andi %s, %s : i1" % (b.cmp("grp_small", "ule", b.cap(), b.i(8)), b.cmp("grp_q8", "eq", b.dense(), b.i(0))))
    b.emit("scf.if %s {" % use_group)
    attn_group(b, g, controlled(b))
    b.emit("} else {")
    linear(b, "blk.{}.attn_q.weight", E, 3072, xn, b.o(0), "%q", 3072)
    linear(b, "blk.{}.attn_rel.weight", E, 384, xn, b.o(0), "%r", 384)
    linear(b, "blk.{}.attn_gate.weight", E, 3072, xn, b.o(0), "%gate", 3072)
    linear(b, "blk.{}.attn_sink_proj.weight", E, 24, xn, b.o(0), "%small", 24, kind="bf16")
    b.emit("scf.if %s {" % g)
    linear(b, "blk.{}.attn_k.weight", E, 1024, xn, b.col(1), "%k", 1024)
    linear(b, "blk.{}.attn_v.weight", E, 1024, xn, b.col(2), "%v", 1024)
    linear(b, "blk.{}.attn_k_conv_proj.weight", E, 32, xn, b.col(1), "%ob", 32, kind="bf16")
    linear(b, "blk.{}.attn_v_conv_proj.weight", E, 32, xn, b.col(2), "%y", 32, kind="bf16")
    b.emit("} else {")
    b.emit("scf.if %s {" % controlled(b))
    linear(b, "blk.{}.attn_k.weight", E, 1024, xn, b.col(1), "%k", 1024)
    linear(b, "blk.{}.attn_v.weight", E, 1024, xn, b.col(2), "%v", 1024)
    b.emit("} else {")
    linear(b, "blk.{}.attn_k.weight", E, 1024, xn, b.o(0), "%k", 1024)
    linear(b, "blk.{}.attn_v.weight", E, 1024, xn, b.o(0), "%v", 1024)
    b.emit("}")
    b.emit("}")
    b.emit("}")
    out.append(hook("proj", b))
    # oproj
    b = Body()
    linear(b, "blk.{}.attn_output.weight", 3072, E, "%ab", b.o(0), "%ob", E)
    out.append(hook("oproj", b))
    # norm_ffn
    b = Body()
    b.emit("scf.if %s {" % controlled(b))
    rms_norm(b, "blk.{}.ffn_norm.weight", "%x", b.o(0), "%xn", b.o(0))
    b.emit("}")
    out.append(hook("norm_ffn", b))
    return "".join(out)


# ---------------------------------------------------------------- mix

def mix_states(b, nb, nq, mode, query, out, blocks="%blocks", cur="%cur", rinv="%small", offs=None, norm_w="%xn", commit=None):
    """commit: (workgroup count, site, conv visit, state buffer, source buffer) of a conv state commit that rides in this launch."""
    cap, cE = b.cap(), b.cE()
    z = b.i(0)
    boff, coff, ooff = offs or (z, z, z)
    nc, csite, cvisit, cstate, csrc = commit or (z, z, z, "%arena", "%ob")
    args = [cap, nc, nq, nb, cE, cE, b.i(mode), b.i(0), b.i(0), boff, coff, ooff, csite, cvisit, "%span_table", "%xn", blocks, cur, query, norm_w, out, rinv, cstate, csrc]
    klaunch(b, "inkling_mix_states", [cap, nc, nq], list(zip(args, ["index"] * 14 + ["buffer"] * 10)))


def controlled(b):
    inl = b.op("in_loop", "func.call pure @inkling_layer_in_loop(%layer) : (index) -> (i1)")
    later = b.cmp("later", "ugt", "%pass", b.i(0))
    return b.op("controlled", "scalar.andi %s, %s : i1" % (inl, later))


def bank_offs(b, state):
    """Element offsets of bank state `state` of the set that the message mix of this pass reads."""
    cE = b.cE()
    prev = b.op("prev_pass", "index.sub %%pass, %s : index" % b.i(1))
    return b.mul("bank_set", b.mul("bank_set0", prev, b.i(2)), cE) if state == 0 else b.add("bank_cur", b.mul("bank_set", b.mul("bank_set0", prev, b.i(2)), cE), cE)


def ctrl_weights(b, kinds):
    """Buffers of the controller weights (F32 for the Q4_K_XL file, BF16 for the Q8_K_XL file) for the controller kinds, as (down, forget, gain) lists."""
    f32 = b.cmp("ctrl_f32", "eq", b.dense(), b.i(0))
    down, forget, gain = [], [], []
    for k in kinds:
        for lst, part, shape in ((down, "down", "64x4608"), (forget, "forget", "2304x64"), (gain, "gain", "2304x64")):
            name = "blk.{}.loop_%s_ctrl_%s.weight" % (k, part)
            lst.append(b.pick(f32, [(name, shape + "xf32"), (name, shape + "xbf16")]))
    while len(down) < 3:
        down.append(down[0]); forget.append(forget[0]); gain.append(gain[0])
    return down, forget, gain


def ctrl_hook(b, nc, kinds, m_col0, g):
    """Message mix of this pass, then the controller chain: reads the mix in xn, writes the controlled columns to x."""
    cap, cE = b.cap(), b.cE()
    chunk = b.cfg("chunk", "inkling.mix.ctrl_chunk")
    msg_bytes = b.scale("msg_bytes", cap, b.o(4 * E * 4))
    rinv_bytes = b.scale("rinv_bytes", cap, b.o(16))
    parts = b.mul("parts", b.mul("capnc", cap, b.i(nc)), b.div("nchunk", b.i(4608), chunk))
    part_bytes = b.scale("part_bytes", parts, b.o(256))
    z_bytes = b.scale("z_bytes", b.mul("capnc2", cap, b.i(nc)), b.o(256))
    b.emit("%%msg = buffer.alloca<global> align(256) %s : buffer" % msg_bytes)
    b.emit("%%rinv_m = buffer.alloca<global> align(256) %s : buffer" % rinv_bytes)
    b.emit("%%part = buffer.alloca<global> align(256) %s : buffer" % part_bytes)
    b.emit("%%z = buffer.alloca<global> align(256) %s : buffer" % z_bytes)
    q = b.weight("blk.{}.loop_msg_query.weight", "4x2304xf32")
    boff, coff = bank_offs(b, 0), bank_offs(b, 1)
    down, forget, gain = ctrl_weights(b, kinds)
    step = b.flag("xstep", g) if g else b.i(0)
    ncv = b.i(nc)
    spans = "%span_table"
    xc0, mc0 = b.i(0), b.i(m_col0)
    b.emit("scf.if %s {" % b.cmp("decode_rows", "ule", cap, b.i(8)))
    args = [cap, ncv, xc0, step, mc0, cE, boff, coff, spans, "%xn", "%bank", q] + down + ["%msg", "%rinv_m", "%part"]
    klaunch(b, "inkling_ctrl_down_msg", [cap, ncv], list(zip(args, ["index"] * 8 + ["buffer"] * 10)))
    args = [cap, ncv, xc0, step, mc0, cE, spans, "%xn", "%msg"] + forget + gain + ["%small", "%rinv_m", "%part", "%x"]
    klaunch(b, "inkling_ctrl_apply_z", [cap, ncv], list(zip(args, ["index"] * 6 + ["buffer"] * 13)))
    b.emit("} else {")
    mix_states(b, b.i(1), b.i(4), 0, q, "%msg", "%bank", "%bank", "%rinv_m", (boff, coff, b.i(0)))
    args = [cap, ncv, xc0, step, mc0, cE, spans, "%xn", "%msg"] + down + ["%part"]
    klaunch(b, "inkling_ctrl_down", [cap, ncv], list(zip(args, ["index"] * 6 + ["buffer"] * 7)))
    args = [cap, ncv, xc0, step, mc0, spans, "%small", "%rinv_m", "%part", "%z"]
    klaunch(b, "inkling_ctrl_z", [cap, ncv], list(zip(args, ["index"] * 5 + ["buffer"] * 5)))
    args = [cap, ncv, xc0, step, mc0, cE, spans, "%xn", "%msg"] + forget + gain + ["%z", "%x"]
    klaunch(b, "inkling_ctrl_apply", [cap, ncv], list(zip(args, ["index"] * 6 + ["buffer"] * 11)))
    b.emit("}")


def conv_visit(b):
    return b.add("convv", b.mul("convv0", "%visit", b.i(79872)), b.i(167772160))


def conv_site(b, site, push, push_block, bank_on, bank_origin, a, wname, commit=True):
    cap, cE = b.cap(), b.cE()
    wb = b.weight(wname)
    cv = conv_visit(b)
    args = [cap, b.i(site), cv, push, push_block, cE, bank_on, bank_origin, "%span_table", "%arena", a, wb, "%cur", "%blocks", "%bank"]
    klaunch(b, "inkling_conv_res", [cap], list(zip(args, ["index"] * 8 + ["buffer"] * 7)))
    if commit:
        conv_commit(b, site, a)


def conv_commit(b, site, a):
    sc = b.span_cap()
    args = [sc, b.i(site), conv_visit(b), "%span_table", "%arena", a]
    klaunch(b, "inkling_conv_commit", [sc], list(zip(args, ["index"] * 3 + ["buffer"] * 3)))


def prev_commit(b, is0):
    """Conv state commit of the FFN site of the visit that runs just before this one (layers 0 to 17, then the loop passes 6 to 17, then 18 to 23): (workgroup count, conv visit)."""
    i1, i6, i17 = b.i(1), b.i(6), b.i(17)
    later = b.cmp("later", "ugt", "%pass", b.i(0))
    is6 = b.cmp("is6", "eq", "%layer", i6)
    wrap6 = b.op("wrap6", "scalar.andi %s, %s : i1" % (is6, later))
    is18 = b.cmp("is18", "eq", "%layer", b.i(18))
    wrap = b.op("wrap", "scalar.ori %s, %s : i1" % (wrap6, is18))
    layer_pos = b.op("layer_pos", "scf.select %s, %s, %s : index" % (is0, i1, "%layer"))
    lm1 = b.op("lm1", "index.sub %s, %s : index" % (layer_pos, i1))
    pl = b.op("pl", "scf.select %s, %s, %s : index" % (wrap, i17, lm1))
    pass_pos = b.op("pass_pos", "scf.select %s, %s, %s : index" % (later, "%pass", i1))
    pm1 = b.op("pm1", "index.sub %s, %s : index" % (pass_pos, i1))
    pp0 = b.op("pp0", "scf.select %s, %s, %s : index" % (wrap6, pm1, "%pass"))
    pp = b.op("pp", "scf.select %s, %s, %s : index" % (is18, b.i(2), pp0))
    pv = b.op("pv", "func.call pure @inkling_visit_index(%s, %s) : (index, index) -> (index)" % (pl, pp))
    pconv = b.add("pconv", b.mul("pconv0", pv, b.i(79872)), b.i(167772160))
    pass0 = b.cmp("pass0", "eq", "%pass", b.i(0))
    skip = b.op("skip", "scalar.andi %s, %s : i1" % (is6, pass0))
    nc = b.op("nc", "scf.select %s, %s, %s : index" % (skip, b.i(0), b.span_cap()))
    return nc, pconv


def mix_file():
    out = ["// Mix area hooks: attention and FFN residual mixes and the post convolutions. The loop message mix runs inside the controller hooks (mix_msg is a no-op).\n\n"]
    # mix_attn
    b = Body()
    g = is_global(b)
    nq = b.op("nq", "scf.select %s, %s, %s : index" % (g, b.i(3), b.i(1)))
    l5 = b.add("l5", "%layer", b.i(5))
    nb = b.div("nb", l5, b.i(6))
    is0 = b.cmp("is0", "eq", "%layer", b.i(0))
    b.emit("scf.if %s {" % is0)
    b.emit("} else {")
    ctl = controlled(b)
    l6 = b.cmp("is6", "eq", "%layer", b.i(6))
    restore = b.op("restore", "scalar.andi %s, %s : i1" % (ctl, l6))
    nc, pconv = prev_commit(b, is0)
    nc_main = b.op("nc_main", "scf.select %s, %s, %s : index" % (restore, b.i(0), nc))
    b.emit("scf.if %s {" % restore)
    mix_states(b, b.i(0), b.i(1), 0, "%xn", "%cur", "%blocks", "%blocks", "%small", (b.i(0), b.mul("slot7", b.i(7), b.cE()), b.i(0)), commit=(b.span_cap(), b.i(4352), pconv, "%arena", "%y"))
    b.emit("}")
    wb = b.pick(g, [("blk.{}.attn_res_query.weight", "6912xf32"), ("blk.{}.attn_res_query.weight", "2304xf32")])
    commit = (nc_main, b.i(4352), pconv, "%arena", "%y")
    b.emit("scf.if %s {" % ctl)
    mix_states(b, nb, nq, 0, wb, "%xn", commit=commit)
    b.emit("} else {")
    b.emit("scf.if %s {" % g)
    mix_states(b, nb, b.i(3), 0, wb, "%x", commit=commit)
    b.emit("} else {")
    wq = b.weight("blk.{}.attn_res_query.weight", "2304xf32")
    wn = b.weight("blk.{}.attn_norm.weight")
    mix_states(b, nb, b.i(1), 1, wq, "%xn", norm_w=wn, commit=commit)
    b.emit("}")
    b.emit("}")
    b.emit("}")
    out.append(hook("mix_attn", b))
    # post_attn
    b = Body()
    push_phase = b.rem("phase", "%layer", b.i(6))
    is_push = b.cmp("is_push", "eq", push_phase, b.i(0))
    push = b.flag("push", is_push)
    push_block = b.div("push_block", "%layer", b.i(6))
    is12 = b.cmp("is12", "eq", "%layer", b.i(12))
    first2 = b.cmp("first2", "ult", "%pass", b.i(2))
    bank_flag = b.op("bank_and", "scalar.andi %s, %s : i1" % (is12, first2))
    bank_on = b.flag("bank_on", bank_flag)
    bank_origin = b.mul("bank_origin", b.mul("pass2", "%pass", b.i(2)), b.cE())
    conv_site(b, 2048, push, push_block, bank_on, bank_origin, "%ob", "blk.{}.attn_post_conv.weight", commit=False)
    out.append(hook("post_attn", b))
    # mix_ffn
    b = Body()
    l6 = b.div("l6", "%layer", b.i(6))
    nb = b.add("nbf", l6, b.i(1))
    wb = b.weight("blk.{}.ffn_res_query.weight")
    commit = (b.span_cap(), b.i(2048), conv_visit(b), "%arena", "%ob")
    b.emit("scf.if %s {" % controlled(b))
    mix_states(b, nb, b.i(1), 0, wb, "%xn", commit=commit)
    b.emit("} else {")
    wn = b.weight("blk.{}.ffn_norm.weight")
    mix_states(b, nb, b.i(1), 1, wb, "%xn", norm_w=wn, commit=commit)
    b.emit("}")
    out.append(hook("mix_ffn", b))
    # post_ffn
    b = Body()
    is17 = b.cmp("is17", "eq", "%layer", b.i(17))
    first2 = b.cmp("first2", "ult", "%pass", b.i(2))
    bank_flag = b.op("bank_and", "scalar.andi %s, %s : i1" % (is17, first2))
    bank_on = b.flag("bank_on", bank_flag)
    set_base = b.mul("set_base", b.mul("pass2", "%pass", b.i(2)), b.cE())
    bank_origin = b.add("bank_origin", set_base, b.cE())
    conv_site(b, 4352, b.i(0), b.i(0), bank_on, bank_origin, "%y", "blk.{}.ffn_post_conv.weight", commit=False)
    b.emit("scf.if %s {" % b.cmp("is5", "eq", "%layer", b.i(5)))
    mix_states(b, b.i(0), b.i(1), 0, "%xn", "%blocks", "%blocks", "%cur", "%small", (b.i(0), b.i(0), b.mul("slot7", b.i(7), b.cE())), commit=(b.span_cap(), b.i(4352), conv_visit(b), "%arena", "%y"))
    b.emit("} else {")
    b.emit("scf.if %s {" % b.cmp("is23", "eq", "%layer", b.i(23)))
    conv_commit(b, 4352, "%y")
    b.emit("}")
    b.emit("}")
    out.append(hook("post_ffn", b))
    b = Body()
    ctrl_hook(b, 3, ["q", "k", "v"], 0, is_global(b))
    out.append(hook("ctrl_qkv", b))
    b = Body()
    ctrl_hook(b, 1, ["ffn"], 3, None)
    out.append(hook("ctrl_ffn", b))
    # stubs
    for name in ["mix_msg"]:
        out.append("command.def public @inkling_hook_%s(%%layer: index, %%visit: index, %%pass: index) launch(%s) {\n  command.return\n}\n\n" % (name, SIG))
    return "".join(out)


# ---------------------------------------------------------------- attn

def attn_file():
    out = ["""config.decl @inkling.pool : %value: index where [range(%value, 0, 2147483647)]

config.decl @inkling.context : %value: index where [range(%value, 1, 2147483647)]

target.decl @inkling_rdna3

target.decl @inkling_rdna4

// Smallest decode split in 64-key blocks. The kernels raise it so that a visit has at most 64 splits (docs/OPT_D4.md).
template.decl @inkling.tune.attn.split_blocks(%context: index, %global: index) -> (index)

template.def<@inkling.tune.attn.split_blocks> priority(-1) @inkling_attn_split_blocks_default(%context: index, %global: index) -> (index) {
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c4 = index.constant 4 : index
  %is_global = index.cmp ne, %global, %c0 : index
  %value = scf.select %is_global, %c4, %c1 : index
  template.return %value : index
}

"""]
    for arch, alias in [("rdna3", "gfx1151"), ("rdna4", "gfx1201")]:
        out.append("""template.def<@inkling.tune.attn.split_blocks> target(@inkling_%s) @inkling_attn_split_blocks_%s(%%context: index, %%global: index) -> (index) {
  %%value = index.constant 1 : index
  template.return %%value : index
}

""" % (arch, alias))

    def common(b):
        g = is_global(b)
        gl = b.flag("gl", g)
        lv = b.op("lv", "index.sub %%visit, %s : index" % b.div("gv", "%visit", b.i(6)))
        pool = b.pool()
        kvg = b.scale("kvg", b.mul("gvpool", "%gv", pool), b.o(4096))
        kvl = b.scale("kvl", lv, b.o(4194304))
        kv = b.op("kv_origin", "scf.select %s, %s, %s : offset" % (g, kvg, kvl))
        cv = b.addo("conv_origin", b.scale("cv0", "%visit", b.o(79872)), b.o(167772160))
        return g, gl, kv, cv

    # attn_prep
    b = Body()
    g, gl, kv, cv = common(b)
    cap, sc = b.cap(), b.span_cap()
    qc = b.weight("blk.{}.attn_q_clip.weight")
    rc = b.weight("blk.{}.attn_rel_clip.weight")
    kc = b.weight("blk.{}.attn_k_conv.weight")
    vc = b.weight("blk.{}.attn_v_conv.weight")
    args = [cap, sc, gl, cv, "%span_table", "%arena", "%q", "%k", "%v", "%r", "%ob", "%y", qc, rc, kc, vc, "%q", "%r", "%xn", "%x"]
    types = ["index", "index", "index", "offset"] + ["buffer"] * 16
    klaunch(b, "inkling_attn_prep", [cap], list(zip(args, types)))
    out.append(hook("attn_prep", b))

    # attn
    b = Body()
    g, gl, kv, cv = common(b)
    cap, sc, ctx = b.cap(), b.span_cap(), b.ctx()
    split = b.op("split", "template.apply<@inkling.tune.attn.split_blocks>(%s, %s) : (index, index) -> (index)" % (ctx, gl))
    max_blocks = b.div("max_blocks", ctx, b.i(64))
    vis_g = b.add("vis_g", max_blocks, b.i(1))
    vis = b.op("vis", "scf.select %s, %s, %s : index" % (g, vis_g, b.i(17)))
    max_splits0 = b.div("max_splits0", b.add("vis_up", vis, b.op("splitm1", "index.sub %s, %s : index" % (split, b.i(1)))), split)
    max_splits = b.op("max_splits", "index.min %s, %s : index" % (max_splits0, b.i(64)))
    tile_bound = b.add("tile_bound", b.div("cap16", b.add("cap15", cap, b.i(15)), b.i(16)), sc)
    partial_n = b.mul("partial_n", sc, max_splits)
    partial_bytes = b.scale("partial_bytes", partial_n, b.o(24 * 136 * 4))
    b.emit("%%partial = buffer.alloca<global> align(256) %s : buffer" % partial_bytes)
    rel = b.weight("blk.{}.attn_rel_bias.weight")
    tile_t = ["index"] * 7 + ["offset"] + ["buffer"] * 12

    def tile(mode, work):
        args = [cap, sc, b.i(mode), gl, split, max_splits, max_blocks, kv, "%span_table", "%origins", "%arena", "%q", "%r", "%xn", "%x", rel, "%small", "%gate", "%ab", "%partial"]
        klaunch(b, "inkling_attn_tile", work, list(zip(args, tile_t)))

    tile(0, [tile_bound, b.i(24), b.i(1)])
    tile(1, [sc, b.i(8), max_splits])
    total = b.add("post_total", b.mul("post_sc", sc, b.i(40)), cap)
    args = [cap, sc, gl, split, max_splits, max_blocks, kv, cv, "%span_table", "%origins", "%arena", "%small", "%gate", "%ab", "%partial", "%xn", "%x", "%k", "%v"]
    klaunch(b, "inkling_attn_post", [total], list(zip(args, ["index"] * 6 + ["offset"] * 2 + ["buffer"] * 11)))
    out.append(hook("attn", b))
    return "".join(out)


# ---------------------------------------------------------------- experts

def experts_file():
    out = ["// Expert hooks. Router results live in shared buffers: logits in small ([cap][128]), ids in ab, weights in ob, f16 router input in gate (docs/DISPATCH.md aliasing table).\n\n"]
    out.append("""target.decl @inkling_rdna3

// Subgroup size of the decode down kernel (docs/OPT_W64.md).
template.decl @inkling.tune.experts.down_wave() -> (index)

template.def<@inkling.tune.experts.down_wave> priority(-1) @inkling_experts_down_wave_default() -> (index) {
  %value = index.constant 32 : index
  template.return %value : index
}

template.def<@inkling.tune.experts.down_wave> target(@inkling_rdna3) @inkling_experts_down_wave_gfx1151() -> (index) {
  %value = index.constant 64 : index
  template.return %value : index
}

""")
    # router
    b = Body()
    cap = b.cap()
    rw = b.weight("blk.{}.ffn_gate_inp.weight")
    bias = b.weight("blk.{}.exp_probs_b.bias")
    klaunch(b, "inkling_router_logits", [cap], list(zip([cap, "%xn", rw, "%small", "%gate"], ["index"] + ["buffer"] * 4)))
    out.append(hook("router", b))
    # experts
    b = Body()
    cap = b.cap()
    wtype = b.op("wtype", "func.call pure @inkling_expert_wtype(%layer) : (index) -> (index)")
    is_fp4 = b.cmp("is_fp4", "eq", wtype, b.i(0))
    n8 = b.mul("n8", cap, b.i(8))
    nbound = b.add("nbound", cap, b.i(128))
    records = b.op("records", "index.min %s, %s : index" % (n8, nbound))
    tasks_bytes = b.op("tasks_bytes", "index.scale %s, %s : index, offset -> offset" % (records, b.o(264)))
    tasks_total = b.op("tasks_total", "index.add %s, %s : offset" % (tasks_bytes, b.o(16)))
    act_bytes = b.scale("act_bytes", n8, b.o(2304))
    dout_bytes = b.scale("dout_bytes", n8, b.o(9216))
    b.emit("%%tasks = buffer.alloca<global> align(256) %s : buffer" % tasks_total)
    b.emit("%%act = buffer.alloca<global> align(256) %s : buffer" % act_bytes)
    b.emit("%%dout = buffer.alloca<global> align(256) %s : buffer" % dout_bytes)
    bias = b.weight("blk.{}.exp_probs_b.bias")
    gu, dn, gus, dns = b.tmp("gu"), b.tmp("dn"), b.tmp("gus"), b.tmp("dns")
    is_q8 = b.cmp("is_q8e", "eq", b.dense(), b.i(0))
    b.emit("%s, %s, %s, %s = scf.if %s -> (buffer, buffer, buffer, buffer) {" % (gu, dn, gus, dns, is_fp4))
    a1 = b.weight("blk.{}.ffn_gate_up_exps.weight", "382205952xi8")
    a2 = b.weight("blk.{}.ffn_down_exps.weight", "191102976xi8")
    a3 = b.weight("blk.{}.ffn_gate_up_exps.scale")
    a4 = b.weight("blk.{}.ffn_down_exps.scale")
    b.emit("  scf.yield %s, %s, %s, %s : buffer, buffer, buffer, buffer" % (a1, a2, a3, a4))
    b.emit("} else {")
    b.emit("%s, %s = scf.if %s -> (buffer, buffer) {" % ("%q8gu", "%q8dn", is_q8))
    c1 = b.weight("blk.{}.ffn_gate_up_exps.weight", "721944576xi8")
    c2 = b.weight("blk.{}.ffn_down_exps.weight", "360972288xi8")
    b.emit("  scf.yield %s, %s : buffer, buffer" % (c1, c2))
    b.emit("} else {")
    d1 = b.weight("blk.{}.ffn_gate_up_exps.weight", "1358954496xi8")
    d2 = b.weight("blk.{}.ffn_down_exps.weight", "679477248xi8")
    b.emit("  scf.yield %s, %s : buffer, buffer" % (d1, d2))
    b.emit("}")
    b.emit("  scf.yield %%q8gu, %%q8dn, %s, %s : buffer, buffer, buffer, buffer" % (bias, bias))
    b.emit("}")
    wide = b.cmp("wide", "uge", cap, b.i(48))

    def chain(tile, gu_k, dn_k, routes=True):
        if routes:
            klaunch(b, "inkling_experts_routes", [], list(zip([cap, b.i(tile), "%ab", "%tasks"], ["index", "index", "buffer", "buffer"])))
        klaunch(b, gu_k, [cap], list(zip([cap, wtype, "%gate", "%tasks", gu, gus, "%act"], ["index", "index"] + ["buffer"] * 5)))
        if isinstance(dn_k, tuple):
            wave = b.op("down_wave", "template.apply<@inkling.tune.experts.down_wave>() : () -> (index)")
            on = b.cmp("down_w64", "eq", wave, b.i(64))
            b.emit("scf.if %s {" % on)
            klaunch(b, dn_k[0], [cap], list(zip([cap, wtype, "%act", "%tasks", dn, dns, "%dout"], ["index", "index"] + ["buffer"] * 5)))
            b.emit("} else {")
            klaunch(b, dn_k[1], [cap], list(zip([cap, wtype, "%act", "%tasks", dn, dns, "%dout"], ["index", "index"] + ["buffer"] * 5)))
            b.emit("}")
            return
        klaunch(b, dn_k, [cap], list(zip([cap, wtype, "%act", "%tasks", dn, dns, "%dout"], ["index", "index"] + ["buffer"] * 5)))

    decode = b.cmp("decode_rows", "ule", cap, b.i(8))
    b.emit("scf.if %s {" % decode)
    klaunch(b, "inkling_router_decode", [], list(zip([cap, b.i(8), "%small", bias, "%ab", "%ob", "%tasks"], ["index", "index"] + ["buffer"] * 5)))
    b.emit("} else {")
    klaunch(b, "inkling_router_select", [cap], list(zip([cap, "%small", bias, "%ab", "%ob"], ["index"] + ["buffer"] * 4)))
    b.emit("}")
    b.emit("scf.if %s {" % wide)
    chain(64, "inkling_experts_gate_up_wmma", "inkling_experts_down_wmma")
    b.emit("} else {")
    b.emit("scf.if %s {" % decode)
    chain(8, "inkling_experts_gate_up", ("inkling_experts_down_w64", "inkling_experts_down"), routes=False)
    b.emit("} else {")
    chain(8, "inkling_experts_gate_up", ("inkling_experts_down_w64", "inkling_experts_down"))
    b.emit("}")
    b.emit("}")
    klaunch(b, "inkling_experts_combine", [cap], list(zip([cap, "%dout", "%ob", "%y"], ["index"] + ["buffer"] * 3)))
    out.append(hook("experts", b))
    return "".join(out)

# ---------------------------------------------------------------- layer

HOOKS = "mix_attn mix_msg ctrl_qkv norm_in proj attn_prep attn oproj post_attn mix_ffn ctrl_ffn norm_ffn router experts post_ffn".split()
# capture points: (hook after which it runs, id, width, column in the capture-all row, half, source buffer, condition)
CAPTURES = [
    ("ctrl_qkv", 10, 2304, "x", 0), ("norm_in", 11, 2304, "xn", 0), ("proj", 12, 3072, "q", 0), ("proj", 13, 1024, "k", 0), ("proj", 14, 1024, "v", 0),
    ("proj", 15, 384, "r", 0), ("proj", 19, 3072, "gate", 0), ("proj", 21, 24, "small", 0), ("attn_prep", 22, 3072, "q", 0), ("attn_prep", 23, 1024, "xn", 1), ("attn_prep", 24, 1024, "x", 1), ("attn_prep", 25, 384, "r", 0),
    ("attn", 16, 3072, "ab", 0), ("oproj", 17, 2304, "ob", 0), ("post_attn", 27, 2304, "cur", 0), ("norm_ffn", 28, 2304, "xn", 0),
    ("router", 29, 128, "small", 0), ("experts", 18, 2304, "y", 0), ("post_ffn", 20, 2304, "cur", 0),
]


def layer_file():
    cols, at = {}, 0
    for _, cid, width, _, _ in CAPTURES:
        cols[cid] = at
        at += width
    total = at
    o = ["// Registry of kernel hooks and the visit command. Every hook takes the visit coordinates as command specializations and the same 20 buffers (docs/DISPATCH.md).\n// Buffers: parameters, span_table, token_map, metadata, origins, arena, cur, blocks, bank, x, xn, small, q, k, v, r, gate, ab, ob, y.\n"]
    for h in HOOKS:
        o.append("command.decl @inkling_hook_%s(%%layer: index, %%visit: index, %%pass: index) launch(%s)\n\n" % (h, SIG))
    o.append("""func.decl pure @inkling_layer_is_global(%layer: index) -> (i1)

func.decl pure @inkling_layer_in_loop(%layer: index) -> (i1)

config.decl @inkling.capture_id : %value: index where [range(%value, 0, 1024)]

config.decl @inkling.capture_visit : %value: index where [range(%value, 0, 47)]

config.decl @inkling.capture_width : %value: index where [range(%value, 1, 1048576)]

config.decl @inkling.stop_after : %value: index where [range(%value, 0, 1000)]

kernel.decl @inkling_capture_rows() launch(%src_width: index, %col: index, %half: index, %metadata: buffer, %src: buffer, %residual: buffer)

// Debug capture of a shared tensor at one visit. Capture id 99 copies every tensor of the table below into its column range (total width WIDTH); another id copies only the tensor with that id into column 0.
command.def @inkling_capture_point(%id: index, %visit: index, %src_width: index, %col: index, %half: index) launch(%metadata: buffer, %src: buffer, %residual: buffer) {
  %capture = config.get @inkling.capture_id : index
  %capture_visit = config.get @inkling.capture_visit : index
  %c0 = index.constant 0 : index
  %c99 = index.constant 99 : index
  %all = index.cmp eq, %capture, %c99 : index
  %id_match = index.cmp eq, %capture, %id : index
  %any_id = scalar.ori %all, %id_match : i1
  %visit_match = index.cmp eq, %capture_visit, %visit : index
  %selected = scalar.andi %any_id, %visit_match : i1
  scf.if %selected {
    %column = scf.select %all, %col, %c0 : index
    kernel.launch @inkling_capture_rows(%src_width, %column, %half, %metadata, %src, %residual) : (index, index, index, buffer, buffer, buffer)
  }
  command.return
}

""".replace("WIDTH", str(total)))
    o.append("// One layer visit. The visit coordinates are exact specializations, so global or local attention, loop controllers and expert weight types resolve while the command is planned.\n")
    o.append("command.def public @inkling_visit(%%layer: index, %%visit: index, %%pass: index) launch(%s, %%residual: buffer) {\n" % SIG)
    o.append("""  %global = func.call pure @inkling_layer_is_global(%layer) : (index) -> (i1)
  %in_loop = func.call pure @inkling_layer_in_loop(%layer) : (index) -> (i1)
  %c0 = index.constant 0 : index
  %later = index.cmp ugt, %pass, %c0 : index
  %controlled = scalar.andi %in_loop, %later : i1
""")
    consts = sorted({w for _, _, w, _, _ in CAPTURES} | {c for _, c, _, _, _ in CAPTURES} | {cols[c] for c in cols} | {0, 1} | set(range(len(HOOKS) + 1)))
    for v in consts:
        o.append("  %%k%d = index.constant %d : index\n" % (v, v))
    o.append("  %stop = config.get @inkling.stop_after : index\n")
    o.append("  command.serial {\n")
    for n, h in enumerate(HOOKS):
        cond = h in ("mix_msg", "ctrl_qkv", "ctrl_ffn")
        o.append("    %%go%d = index.cmp ugt, %%stop, %%k%d : index\n" % (n, n))
        o.append("    scf.if %%go%d {\n" % n)
        if cond:
            o.append("    scf.if %controlled {\n")
        o.append("    command.launch @inkling_hook_%s[%%layer, %%visit, %%pass](%s) : [index, index, index](%s)\n" % (h, ARGS, TYPES20))
        if cond:
            o.append("    }\n")
        for hk, cid, width, buf, half in CAPTURES:
            if hk == h:
                o.append("    command.launch @inkling_capture_point[%%k%d, %%visit, %%k%d, %%k%d, %%k%d](%%metadata, %%%s, %%residual) : [index, index, index, index, index](buffer, buffer, buffer)\n" % (cid, width, cols[cid], half, buf))
        o.append("    }\n")
    o.append("  }\n  command.return\n}\n")
    return "".join(o)


CONFIGS = {
    "inkling.tokens": "range(%value, 1, 2147483647)",
    "inkling.spans": "range(%value, 1, 2147483647)",
    "inkling.context": "range(%value, 1, 2147483647)",
    "inkling.pool": "range(%value, 0, 2147483647)",
    "inkling.dense_type": "range(%value, 0, 1)",
    "inkling.mix.ctrl_chunk": "range(%value, 64, 512), multiple_of(%value, 64)",
    "inkling.capture_id": "range(%value, 0, 1024)",
}
FUNCS = {
    "inkling_layer_is_global": "func.decl pure @inkling_layer_is_global(%layer: index) -> (i1)",
    "inkling_layer_in_loop": "func.decl pure @inkling_layer_in_loop(%layer: index) -> (i1)",
    "inkling_expert_wtype": "func.decl pure @inkling_expert_wtype(%layer: index) -> (index)",
    "inkling_visit_index": "func.decl pure @inkling_visit_index(%layer: index, %pass: index) -> (index)",
}


EXTRA_DECLS = {
    "@inkling_sample_stats": "kernel.decl @inkling_sample_stats(%rows_value: index) launch(%rows_value: index, %logit_stride: index, %epoch: buffer, %logits: buffer, %scratch: buffer)",
    "@inkling_sample_draw": "kernel.decl @inkling_sample_draw(%rows_value: index) launch(%rows_value: index, %logit_stride: index, %epoch: buffer, %logits: buffer, %scratch: buffer)",
    "@inkling_sample_rows": "kernel.decl @inkling_sample_rows(%rows_value: index) launch(%rows_value: index, %logit_stride: index, %epoch: buffer, %logits: buffer, %outputs: buffer, %scratch: buffer)",
    "@inkling_capture_rows": "kernel.decl @inkling_capture_rows() launch(%src_width: index, %col: index, %half: index, %metadata: buffer, %src: buffer, %residual: buffer)",
}


def with_decls(text):
    """Prepends the declarations that the file needs: kernels (from kernel_decls.loom), configs and pure functions."""
    import re
    kdecl = {}
    for line in open(os.path.join(OUT, "kernel_decls.loom")):
        m = re.match(r"kernel\.decl (@\w+)", line)
        if m:
            kdecl[m.group(1)] = line.rstrip()
    head = []
    for name in sorted(set(re.findall(r"kernel\.launch (@\w+)", text))):
        if name in kdecl and (name + "(") not in "".join(l for l in text.split(chr(10)) if l.startswith("kernel.decl")):
            head.append(kdecl[name])
    for name in sorted(set(re.findall(r"config\.get @([\w.]+)", text))):
        if name in CONFIGS and ("config.decl @" + name) not in text:
            head.append("config.decl @%s : %%value: index where [%s]" % (name, CONFIGS[name]))
    for name in sorted(set(re.findall(r"func\.call pure @(\w+)", text))):
        if name in FUNCS:
            head.append(FUNCS[name])
    for name, decl in EXTRA_DECLS.items():
        if "kernel.launch " + name in text and (name + "(") not in "".join(l for l in text.split(chr(10)) if l.startswith("kernel.decl")):
            head.append(decl)
    head.insert(0, "target.decl @inkling_execution")
    return (chr(10) + chr(10)).join(head) + chr(10) + chr(10) + text if head else text



NAMES = {10: "x", 11: "xn", 12: "q_raw", 13: "k_raw", 14: "v_raw", 15: "r_raw", 19: "gate", 21: "sink", 22: "q", 23: "k", 24: "v", 25: "r", 16: "ab", 17: "ob", 27: "ffn_inp", 28: "xn_ffn", 29: "logits", 18: "y", 20: "l_out"}


def layout_file():
    import json
    items, at = [], 0
    for _, cid, width, _, _ in CAPTURES:
        items.append({"id": cid, "name": NAMES[cid], "col": at, "width": width})
        at += width
    return json.dumps({"total": at, "items": items}, indent=1) + chr(10)


def main():
    with open(os.path.join(OUT, "capture_layout.json"), "w") as f:
        f.write(layout_file())
    for name, fn in [("dense", dense_file), ("attn", attn_file), ("mix", mix_file), ("experts", experts_file), ("LAYER", layer_file)]:
        with open(os.path.join(OUT, "layer.loom" if name == "LAYER" else "hooks_%s.loom" % name), "w") as f:
            f.write(fn() if name == "LAYER" else with_decls(fn()))


main()
