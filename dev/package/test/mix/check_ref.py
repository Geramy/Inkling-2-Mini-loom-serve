"""Check the numpy reference against the oracle (validates the spec reading)."""
import sys
import numpy as np
from inkg import *

d = Dump(sys.argv[1] if len(sys.argv) > 1 else "Q8_K_XL_A")
g = Gguf("Q8")


def nmse(a, b):
    return float(((a - b) ** 2).sum() / (b ** 2).sum())


bad = 0
for slot in range(48):
    ip = phys(slot)
    push = ip % 6 == 0
    # attention mix
    if ip > 0:
        q = g.f32("blk.%d.attn_res_query.weight" % ip)
        out, _ = mix_ref(blocks_at(d, slot, False) + [slot_input(d, slot)], q)
        occ = 0
        exp = d.get("mix_out-%d" % slot, 0)
        exp = exp.reshape(d.n, -1, E).transpose(1, 0, 2)
        e = nmse(out, exp)
        print("slot %2d attn mix nq=%d nmse %.2e" % (slot, out.shape[0], e))
        bad += e > 1e-9
    # ffn mix: cur is the post-attn residual
    ffn_inp = d.get("ffn_inp-%d" % slot, 0)
    q = g.f32("blk.%d.ffn_res_query.weight" % ip)
    out, _ = mix_ref(blocks_at(d, slot, True) + [ffn_inp], q)
    occ = 0 if ip == 0 else 1
    exp = d.get("mix_out-%d" % slot, occ).reshape(d.n, -1, E).transpose(1, 0, 2)
    e = nmse(out, exp)
    print("slot %2d ffn  mix nmse %.2e" % (slot, e))
    bad += e > 1e-9
# messages
for slot in range(24, 48):
    ip = phys(slot)
    prev = 17 if slot == 24 else (35 if slot == 36 else slot - 1)
    p = 0 if slot < 36 else 1
    bank = blocks_at(d, 17 if slot < 36 else 35, True)[2]  # block pushed at ip 12 of the previous pass
    bank_out = d.get("l_out-%d" % (17 if slot < 36 else 35))
    q = g.f32("blk.%d.loop_msg_query.weight" % ip)
    out, _ = mix_ref([bank, bank_out], q)
    exp = d.get("mix_out-%d" % prev, 2).reshape(d.n, 4, E).transpose(1, 0, 2)
    e = nmse(out, exp)
    print("slot %2d msg mix nmse %.2e" % (slot, e))
    bad += e > 1e-9
print("BAD", bad)

# controllers: x' normed (no weight) must equal the dumped norm-S
bad = 0
for slot in list(range(24, 48)):
    ip = phys(slot)
    prev = 17 if slot == 24 else (35 if slot == 36 else slot - 1)
    msgs = d.get("mix_out-%d" % prev, 2).reshape(d.n, 4, E)
    glob = ip % 6 == 5
    q = g.f32("blk.%d.attn_res_query.weight" % ip)
    xm, _ = mix_ref(blocks_at(d, slot, False) + [slot_input(d, slot)], q)
    for c in range(3):
        x = xm[c if glob else 0]
        wd, wf, wg = ctrl_weights(g, ip, c)
        xo = ctrl_ref(x, msgs[:, c], wd, wf, wg)
        errs = [nmse(rms_norm(xo), d.get("norm-%d" % slot, o)) for o in range(3)]
        e = min(errs)
        bad += e > 1e-9
        if e > 1e-9 or slot in (24, 29):
            print("slot %d ctrl %d best occ %d nmse %.2e" % (slot, c, errs.index(e), e))
    ffn_inp = d.get("ffn_inp-%d" % slot, 0)
    q = g.f32("blk.%d.ffn_res_query.weight" % ip)
    xm, _ = mix_ref(blocks_at(d, slot, True) + [ffn_inp], q)
    wd, wf, wg = ctrl_weights(g, ip, 3)
    xo = ctrl_ref(xm[0], msgs[:, 3], wd, wf, wg)
    e = nmse(rms_norm(xo), d.get("norm-%d" % slot, 3))
    bad += e > 1e-9
    if e > 1e-9 or slot in (24, 29):
        print("slot %d ctrl ffn nmse %.2e" % (slot, e))
print("CTRL BAD", bad)

# post convs, zero initial state (prompt A is one span from empty state)
bad = 0
for slot in range(48):
    ip = phys(slot)
    for name, site, occ_in in (("attn_post_conv", "attn", 2), ("ffn_post_conv", "ffn", 3)):
        w = g.f32("blk.%d.%s.weight" % (ip, name))
        x = d.get("conv_in-%d" % slot, occ_in)
        y, _ = conv_ref(x, w)
        if site == "attn":
            cur = y if ip % 6 == 0 else slot_input(d, slot) + y
            e = nmse(cur, d.get("ffn_inp-%d" % slot, 0))
        else:
            e = nmse(slot_input(d, slot) * 0 + d.get("ffn_inp-%d" % slot, 0) + y, d.get("l_out-%d" % slot))
        bad += e > 1e-9
        if e > 1e-9:
            print("slot %d %s conv nmse %.2e" % (slot, site, e))
print("CONV BAD", bad)
