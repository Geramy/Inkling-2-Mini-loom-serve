"""Tests of inkling_conv_res and inkling_conv_commit. Usage: test_conv.py [gfx1201|gfx1151]"""
import os
import sys

import numpy as np

from harness import nmse, run_launch
from inkg import *
from test_mix import SENT, check, spans_buf

ROW = 6656
VISIT_BYTES = 3 * ROW * 4
SITE = {"attn": 2048, "ffn": 4352}
rows = []


def conv_call(device, site, visit, tokens_a, w, cur, spans, arena, push=0, push_block=0, bank_on=0, bank_origin=0, cap=None):
    """One conv_res + conv_commit launch pair. cur [cap][E] and arena are copied; returns (cur, blocks, bank, arena)."""
    cap = cap or cur.shape[0]
    blocks = np.full((8, cap, E), SENT, np.float32)
    bank = np.full((2, cap, E), SENT, np.float32)
    a = np.zeros((cap, E), np.float32)
    a[:tokens_a.shape[0]] = tokens_a
    r = run_launch(["conv_res.loom"], "inkling_conv_res", cap,
                   [SITE[site], visit * VISIT_BYTES, push, push_block, cap * E, bank_on, bank_origin],
                   [spans, arena.copy(), a, w.astype(np.float32), cur.copy(), blocks, bank], device)
    n_spans = int(spans[0])
    if os.environ.get("INK_COMMIT", "mix") == "mix":
        r2 = run_launch(["mix_states.loom", "conv_res.loom"], "inkling_mix_states", [cap, n_spans, 1],
                        [0, cap * E, cap * E, 0, 0, 0, 0, 0, 0, SITE[site], visit * VISIT_BYTES],
                        [spans, np.zeros(4, np.int32), np.zeros((8, cap, E), np.float32), cur.copy(), np.zeros((4, E), np.float32), np.zeros((4, E), np.float32),
                         np.zeros((1, cap, E), np.float32), np.zeros((4, cap), np.float32), r[1], a], device)
        return r[4], r[5], r[6], r2[8]
    r2 = run_launch(["conv_res.loom"], "inkling_conv_commit", n_spans,
                    [SITE[site], visit * VISIT_BYTES],
                    [spans, r[1], a], device)
    return r[4], r[5], r[6], r2[1]


def ref_state(arena, site, visit, spans_desc, x_all, w):
    """Numpy model of one visit: returns y [T][E] and the updated arena."""
    new = arena.copy()
    y = np.zeros_like(x_all)
    first = 0
    for (n, rd, wr) in spans_desc:
        rbase = rd // 4 + visit * 3 * ROW
        wbase = wr // 4 + visit * 3 * ROW
        hist = arena[rbase:rbase + 3 * ROW].reshape(3, ROW)[:, SITE[site]:SITE[site] + E]
        yy, ns = conv_ref(x_all[first:first + n], w, hist)
        y[first:first + n] = yy
        new[wbase:wbase + 3 * ROW].reshape(3, ROW)[:, SITE[site]:SITE[site] + E] = ns
        first += n
    return y, new


def oracle_cases(device):
    g = Gguf("Q8")
    d = Dump("Q8_K_XL_A")
    t = d.n
    cap = t + 5
    for slot in range(48):
        ip = phys(slot)
        push = int(ip % 6 == 0)
        arena = np.zeros(3 * ROW * 4 * 2, np.float32)
        arena[:] = SENT
        arena[:3 * ROW * 2] = 0
        spans = spans_buf(t, [t], [0], [0])
        cur = np.zeros((cap, E), np.float32)
        cur[:t] = slot_input(d, slot)
        w = g.f32("blk.%d.attn_post_conv.weight" % ip)
        out, blocks, bank, na = conv_call(device, "attn", 0, d.get("conv_in-%d" % slot, 2), w, cur, spans, arena, push, 3, 1, cap * E, cap)
        check("A slot %d post-attn conv_res vs ffn_inp" % slot, out[:t], d.get("ffn_inp-%d" % slot, 0), 1e-9, rows)
        if push:
            check("A slot %d block store" % slot, blocks[3, :t], slot_input(d, slot), 1e-14, rows)
        check("A slot %d bank copy (%s)" % (slot, "residual" if push else "new cur"), bank[1, :t], slot_input(d, slot) if push else out[:t], 1e-14, rows)
        _, exp_arena = ref_state(arena, "attn", 0, [(t, 0, 0)], d.get("conv_in-%d" % slot, 2), w)
        check("A slot %d post-attn state commit" % slot, na, exp_arena, 1e-14, rows)
        w2 = g.f32("blk.%d.ffn_post_conv.weight" % ip)
        out2, _, _, na2 = conv_call(device, "ffn", 0, d.get("conv_in-%d" % slot, 3), w2, out, spans, na)
        check("A slot %d post-ffn conv_res vs l_out" % slot, out2[:t], d.get("l_out-%d" % slot), 1e-9, rows)
        check("A slot %d untouched guard" % slot, out2[t:], out[t:], 0.0, rows)


def span_cases(device):
    """Random data: chunking, multiple spans, fork (read != write) and in-place update."""
    rng = np.random.default_rng(7)
    g = Gguf("Q8")
    w = g.f32("blk.3.ffn_post_conv.weight")
    n_slots = 4
    slot_bytes = 2 * VISIT_BYTES
    arena0 = rng.standard_normal(n_slots * 2 * 3 * ROW).astype(np.float32)
    x_total = 40
    xs = rng.standard_normal((x_total, E)).astype(np.float32) * 3

    def run(lengths, reads, writes, tag, visit=1, x=None, arena=arena0):
        t = sum(lengths)
        x = xs[:t] if x is None else x
        cap = t + 3
        spans = spans_buf(t, lengths, reads, writes)
        cur = np.zeros((cap, E), np.float32)
        cur[:t] = rng.standard_normal((t, E))
        out, _, _, na = conv_call(device, "ffn", visit, x, w, cur, spans, arena, 0, 0, 0, 0, cap)
        y, exp_arena = ref_state(arena, "ffn", visit, list(zip(lengths, reads, writes)), x, w)
        check("%s y" % tag, out[:t] - cur[:t], y, 1e-12, rows)
        check("%s arena" % tag, na, exp_arena, 1e-14, rows)
        return na

    # one span of every small length, in place
    for n in (1, 2, 3, 4, 5, 6, 7, 17, 40):
        run([n], [slot_bytes], [slot_bytes], "span len %d in-place" % n)
    # several spans with distinct slots; slot 0 is read and slot 3 written (fork)
    run([1, 2, 5, 3], [0, slot_bytes, 2 * slot_bytes, 3 * slot_bytes], [0, slot_bytes, 2 * slot_bytes, 3 * slot_bytes], "4 spans in-place")
    run([3, 1, 9], [0, slot_bytes, 2 * slot_bytes], [3 * slot_bytes, slot_bytes, 2 * slot_bytes], "3 spans, span 0 forked")
    # chunking: a 40 token sequence in one launch equals any chunking
    arena_zero = np.zeros_like(arena0)
    one = run([40], [0], [0], "chunk 40", 1, xs, arena_zero)
    for sizes in ([13, 27], [1] * 6 + [34], [2, 1, 3, 4, 30], [5, 5, 5, 5, 5, 5, 5, 5], [39, 1]):
        a = arena_zero
        pos = 0
        for n in sizes:
            a = run([n], [0], [0], "chunk %s step %d" % (sizes, pos), 1, xs[pos:pos + n], a)
            pos += n
        check("chunking %s equals one launch" % (sizes,), a, one, 1e-14, rows)
    # decode one token at a time after a prefill
    a = run([30], [0], [0], "prefill 30", 1, xs[:30], arena_zero)
    for k in range(30, 36):
        a = run([1], [0], [0], "decode %d" % k, 1, xs[k:k + 1], a)
    a2 = run([36], [0], [0], "prefill 36", 1, xs[:36], arena_zero)
    check("prefill 30 + 6 decodes equals prefill 36", a, a2, 1e-14, rows)


if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else "gfx1201"
    what = sys.argv[2] if len(sys.argv) > 2 else "all"
    if what in ("all", "span"):
        span_cases(device)
    if what in ("all", "oracle"):
        oracle_cases(device)
    bad = [r for r in rows if not r[2]]
    print("TOTAL %d FAIL %d" % (len(rows), len(bad)))
    sys.exit(1 if bad else 0)
