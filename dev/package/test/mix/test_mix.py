"""Tests of inkling_mix_states against the oracle (prompts A, B) and synthetic data. Usage: test_mix.py [gfx1201|gfx1151]"""
import sys

import numpy as np

from harness import nmse, run_launch
from inkg import *

SENT = 12345.0


def spans_buf(live, lengths=None, reads=None, writes=None):
    """Shared span table: header {count, live, 0, 0} then 8 i32 words per span {first, length, read lo, hi, write lo, hi, position, row}."""
    lengths = lengths or [live]
    b = np.zeros(4 + 8 * len(lengths), np.int32)
    b[0] = len(lengths)
    b[1] = live
    first = 0
    for i, n in enumerate(lengths):
        rd = reads[i] if reads else 0
        wr = writes[i] if writes else 0
        b[4 + 8 * i: 12 + 8 * i] = [first, n, np.uint32(rd & 0xFFFFFFFF).astype(np.int32), rd >> 32, np.uint32(wr & 0xFFFFFFFF).astype(np.int32), wr >> 32, 0, i]
        first += n
    return b


def run_mix(states, query, mode, norm_w, w_stride, cap, device, nq=None, sel=None, nout=None):
    """states: list of [T][E] (blocks then cur). Returns out [nq][nout][E], rinv [nq][cap]."""
    t = states[0].shape[0]
    nb = len(states) - 1
    nq = nq or query.reshape(-1, E).shape[0]
    blocks = np.zeros((8, cap, E), np.float32)
    for i in range(nb):
        blocks[i, :t] = states[i]
    cur = np.zeros((cap, E), np.float32)
    cur[:t] = states[nb]
    q = np.zeros((4, E), np.float32)
    q[:nq] = query.reshape(-1, E)
    nw = np.zeros((4, E), np.float32)
    if norm_w is not None:
        nw[:] = norm_w if norm_w.ndim == 1 else 0
        if norm_w.ndim == 2:
            nw[:norm_w.shape[0]] = norm_w
    out = np.full((nq, cap, E), SENT, np.float32)
    rinv = np.full((4, cap), SENT, np.float32)
    selb = np.zeros(max(4, 1 + (len(sel) if sel is not None else 0)), np.int32)
    if sel is not None:
        selb[0] = len(sel)
        selb[1:1 + len(sel)] = sel
    res = run_launch(["mix_states.loom", "conv_res.loom"], "inkling_mix_states", [cap, 0, nq],
                     [nb, cap * E, cap * E, mode, w_stride, 1 if sel is not None else 0, 0, 0, 0, 0, 0],
                     [spans_buf(t), selb, blocks, cur, q, nw, out, rinv, np.zeros(4, np.float32), np.zeros(4, np.float32)], device)
    return res[6], res[7]


def check(name, got, exp, tol, rows):
    e = nmse(got, exp)
    ok = e <= tol
    rows.append((name, e, ok))
    print("%-44s nmse %.3e %s" % (name, e, "PASS" if ok else "FAIL"), flush=True)


def oracle_a(device, rows):
    g = Gguf("Q8")
    d = Dump("Q8_K_XL_A")
    t = d.n
    cap = t + 8
    for slot in (0, 1, 5, 6, 11, 12, 17, 18, 23, 24, 29, 30, 35, 36, 41, 42, 47):
        ip = phys(slot)
        if ip > 0:
            qn = g.f32("blk.%d.attn_res_query.weight" % ip)
            states = blocks_at(d, slot, False) + [slot_input(d, slot)]
            out, rinv = run_mix(states, qn, 0, None, 0, cap, device)
            exp = d.get("mix_out-%d" % slot, 0).reshape(t, -1, E).transpose(1, 0, 2)
            check("A slot %d attn mix nb=%d nq=%d vs oracle" % (slot, len(states) - 1, exp.shape[0]), out[:, :t], exp, 1e-9, rows)
            check("A slot %d attn mix guard" % slot, out[:, t:], np.full(out[:, t:].shape, SENT, np.float32), 0.0, rows)
            check("A slot %d attn rinv" % slot, rinv[:out.shape[0], :t], rms_inv(exp)[..., 0], 1e-9, rows)
        ffn_inp = d.get("ffn_inp-%d" % slot, 0)
        qn = g.f32("blk.%d.ffn_res_query.weight" % ip)
        states = blocks_at(d, slot, True) + [ffn_inp]
        nw = g.f32("blk.%d.ffn_norm.weight" % ip)
        out, _ = run_mix(states, qn, 1, nw, 0, cap, device)
        exp = d.get("mix_out-%d" % slot, 0 if ip == 0 else 1).reshape(t, E)
        check("A slot %d ffn mix+norm nb=%d vs oracle" % (slot, len(states) - 1), out[0, :t], rms_norm(exp, nw), 1e-9, rows)
    # all message mixes, S = 2, nq = 4
    for slot in range(24, 48):
        ip = phys(slot)
        prev = 17 if slot == 24 else (35 if slot == 36 else slot - 1)
        last = 17 if slot < 36 else 35
        bank = [slot_input(d, 12 if slot < 36 else 30), d.get("l_out-%d" % last)]
        qm = g.f32("blk.%d.loop_msg_query.weight" % ip)
        out, rinv = run_mix(bank, qm, 0, None, 0, cap, device)
        exp = d.get("mix_out-%d" % prev, 2).reshape(t, 4, E).transpose(1, 0, 2)
        check("A slot %d message mix S=2 nq=4 vs oracle" % slot, out[:, :t], exp, 1e-9, rows)
        check("A slot %d message rinv" % slot, rinv[:, :t], rms_inv(exp)[..., 0], 1e-9, rows)


def head_and_light(device, rows):
    g = Gguf("Q8")
    qo = g.f32("output_res_query.weight")
    ow = g.f32("output_norm.weight")
    for name in ("A", "B", "C"):
        d = Dump("Q8_K_XL_A") if name == "A" else Light("Q8_K_XL_" + name, g)
        t = d.n
        cap = t + 4
        states = blocks_at(d, 23, True) + [d.get("l_out-23")]
        out, _ = run_mix(states, qo, 1, ow, 0, cap, device, sel=[t - 1])
        exp = d.get("result_norm", 0).reshape(-1)
        check("%s head mix+output_norm, last token (sel) vs result_norm" % name, out[0, 0], exp, 1e-9, rows)
        check("%s head sel guard" % name, out[0, 1:], np.full(out[0, 1:].shape, SENT, np.float32), 0.0, rows)
        if name == "A":
            sel = [0, 5, t - 1, 3]
            out, _ = run_mix(states, qo, 1, ow, 0, cap, device, sel=sel)
            full = rms_norm(mix_ref(states, qo)[0][0], ow)
            check("A head mix with 4 selected tokens", out[0, :4], full[sel], 1e-12, rows)
        if name != "A":
            for slot in range(1, 48):
                ip = phys(slot)
                if ip == 0:
                    continue
                qn = g.f32("blk.%d.attn_res_query.weight" % ip)
                st = blocks_at(d, slot, False) + [slot_input(d, slot)]
                out, _ = run_mix(st, qn, 0, None, 0, cap, device)
                ref, _ = mix_ref(st, qn)
                check("%s (T=%d) slot %d attn mix vs numpy" % (name, t, slot), out[:, :t], ref, 1e-10, rows)
            for slot in range(24, 48):
                ip = phys(slot)
                last = 17 if slot < 36 else 35
                bank = [slot_input(d, 12 if slot < 36 else 30), d.get("l_out-%d" % last)]
                qm = g.f32("blk.%d.loop_msg_query.weight" % ip)
                out, _ = run_mix(bank, qm, 0, None, 0, cap, device)
                ref, _ = mix_ref(bank, qm)
                check("%s (T=%d) slot %d message mix vs numpy" % (name, t, slot), out[:, :t], ref, 1e-10, rows)


def synthetic(device, rows):
    rng = np.random.default_rng(3)
    for nb in (0, 1, 2, 3, 5, 8):
        for nq in (1, 3, 4):
            t = 5
            cap = 9
            st = [(rng.standard_normal((t, E)) * (1 + 3 * i)).astype(np.float32) for i in range(nb + 1)]
            qn = rng.standard_normal((nq, E)).astype(np.float32)
            nw = rng.standard_normal((nq, E)).astype(np.float32)
            out, rinv = run_mix(st, qn, 1, nw, E, cap, device)
            ref, _ = mix_ref(st, qn) if nb > 0 else (np.stack([st[0]] * nq), None)
            exp = np.stack([rms_norm(ref[q], nw[q]) for q in range(nq)])
            check("synthetic nb=%d nq=%d mode 1 per-column weights" % (nb, nq), out[:, :t], exp, 1e-11, rows)
            out, rinv = run_mix(st, qn, 0, None, 0, cap, device)
            check("synthetic nb=%d nq=%d mode 0" % (nb, nq), out[:, :t], ref, 1e-11, rows)
    # many tokens
    t = 700
    st = [(rng.standard_normal((t, E))).astype(np.float32) for _ in range(5)]
    qn = rng.standard_normal((3, E)).astype(np.float32)
    out, _ = run_mix(st, qn, 0, None, 0, t, device)
    check("synthetic T=700 nb=4 nq=3", out, mix_ref(st, qn)[0], 1e-11, rows)


if __name__ == "__main__":
    device = sys.argv[1] if len(sys.argv) > 1 else "gfx1201"
    rows = []
    for fn in (oracle_a, head_and_light, synthetic):
        fn(device, rows)
    bad = [r for r in rows if not r[2]]
    print("TOTAL %d FAIL %d" % (len(rows), len(bad)))
    sys.exit(1 if bad else 0)
