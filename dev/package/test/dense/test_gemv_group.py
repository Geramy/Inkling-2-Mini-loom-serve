"""Grouped attention projection GEMV (inkling_gemv_attn_group) against numpy on the exact weight values.
Usage: python test_gemv_group.py ARCH [filter]      (PKG env var: package directory)
"""
import os
import sys

import numpy as np

import dtest
from dtest import *

if os.environ.get("PKG"):
    dtest.PKG = os.environ["PKG"]

LIBS = ["lib/inkling/dense_common.loom", "lib/inkling/dense_float.loom", "lib/inkling/dense_q8_0.loom", "lib/inkling/dense_abi.loom",
        "lib/inkling/dense_vec.loom", "kernel/dense/gemv.loom"]
PER_UNIT = {"gfx1151": 4, "gfx1201": 7}.get(sys.argv[1], 2)
NAMES = [("attn_q", 3072), ("attn_rel", 384), ("attn_gate", 3072), ("attn_k", 1024), ("attn_v", 1024),
         ("attn_sink_proj", 24), ("attn_k_conv_proj", 32), ("attn_v_conv_proj", 32)]
DECL = ("kernel.decl @inkling_gemv_attn_group(%conv_on: index, %per_unit: index) launch(%conv_on: index, %control: buffer, %count_origin: offset, %x_stride: index, "
        "%xo_a: offset, %xo_k: offset, %xo_v: offset, %w_q: buffer, %w_rel: buffer, %w_gate: buffer, %w_k: buffer, %w_v: buffer, %w_sink: buffer, %w_kc: buffer, %w_vc: buffer, "
        "%input: buffer, %o_q: buffer, %o_rel: buffer, %o_gate: buffer, %o_k: buffer, %o_v: buffer, %o_sink: buffer, %o_kc: buffer, %o_vc: buffer)")


def group_case(layer, t, conv, rng):
    ws, refs = {}, {}
    x = rng.standard_normal((3, t, 2304)).astype(np.float32)
    for nm, n in NAMES:
        if not conv and nm.endswith("conv_proj"):
            nm_use = "attn_sink_proj"
        else:
            nm_use = nm
        raw, ttype, ne, wdeq = load_weight("q4", "blk.%d.%s.weight" % (layer, nm_use))
        ws[nm] = raw.view(np.int8)
        refs[nm] = (x[1 if nm in ("attn_k", "attn_k_conv_proj") else 2 if nm in ("attn_v", "attn_v_conv_proj") else 0].astype(np.float64) @ wdeq.astype(np.float64).T)
        assert wdeq.shape[0] == n or nm_use != nm, (nm, wdeq.shape)
    ctl = np.array([t, 0, 0, 0], dtype=np.int32)
    inputs = {"ctl": ctl, "x": x}
    for nm, n in NAMES:
        inputs["w_" + nm] = ws[nm]
    outs = {"o_" + nm: np.zeros((t, n), np.float32) for nm, n in NAMES}
    colb = t * 2304 * 4
    args = ", ".join("%w_" + nm for nm, _ in NAMES) + ", %x, " + ", ".join("%o_" + nm for nm, _ in NAMES)
    types = ", ".join(tensor_type(inputs["w_" + nm]) for nm, _ in NAMES) + ", " + tensor_type(x) + ", " + ", ".join(tensor_type(outs["o_" + nm]) for nm, _ in NAMES)
    body = """  %%pu = check.literal value(%d) : index
  %%conv = check.literal value(%d) : index
  %%k = check.literal value(2304) : index
  %%zero = check.literal value(0) : offset
  %%xk = check.literal value(%d) : offset
  %%xv = check.literal value(%d) : offset
  kernel.launch @inkling_gemv_attn_group[%%conv, %%pu](%%conv, %%ctl, %%zero, %%k, %%zero, %%xk, %%xv, %s) : [index, index](index, tensor<4xi32>, offset, index, offset, offset, offset, %s)""" % (
        PER_UNIT, 1 if conv else 0, colb, 2 * colb, args, types)
    c = Case("group_l%d_t%d_c%d" % (layer, t, conv), LIBS, inputs, outs, body, preamble=DECL, config={})
    return c, refs


def main():
    arch = sys.argv[1]
    flt = sys.argv[2] if len(sys.argv) > 2 else ""
    rep = Report()
    rng = np.random.default_rng(11)
    for layer, conv in ((5, True), (0, False)):
        for t in (1, 3, 8):
            c, refs = group_case(layer, t, conv, rng)
            if flt and flt not in c.name:
                continue
            out, err = run_case(c, arch, "/tmp/optD2_t/%s/%s" % (arch, c.name))
            if out is None:
                rep.add(c.name, arch, False, "", err.replace("\n", " ")[:800])
                continue
            worst, ok = 0.0, True
            for nm, n in NAMES:
                if not conv and nm.endswith("conv_proj"):
                    continue
                e = nmse(out["o_" + nm], refs[nm])
                worst = max(worst, e)
                ok = ok and e < 1e-9
            rep.add(c.name, arch, ok, "worst nmse=%.2e" % worst)
    rep.finish()


if __name__ == "__main__":
    main()
