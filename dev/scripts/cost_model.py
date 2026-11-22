import sys, re, collections
inv = sys.argv[1]
T = {}
for line in open(inv, encoding="utf-8"):
    p = [x.strip() for x in line.split("|")]
    if len(p) == 4 and not line.startswith("#"):
        T[p[0]] = (tuple(int(x) for x in p[1].split("x")), p[2], int(p[3]))
NL, LS, LE, NP = 24, 6, 18, 3
def phys(v):
    return v if v < LS else (LS + (v - LS) % (LE - LS) if v < LS + NP*(LE-LS) else v - (NP-1)*(LE-LS))
visits = [phys(v) for v in range(NL + (NP-1)*(LE-LS))]
assert len(visits) == 48
glob = {5, 11, 17, 23}
def b(name): return T[name][2]
def expert_slice(name): return T[name][2] // 128
fam = collections.defaultdict(int)
def add(f, n): fam[f] += n
for vi, l in enumerate(visits):
    p = f"blk.{l}."
    ploop = (LS <= vi < LS + NP*(LE-LS)) and ((vi - LS)//(LE-LS) >= 1)
    for n in ["attn_q", "attn_k", "attn_v", "attn_output", "attn_gate", "attn_rel", "attn_sink_proj"]:
        add("attn_proj(" + T[p+n+".weight"][1] + ")", b(p+n+".weight"))
    if l in glob:
        for n in ["attn_k_conv_proj", "attn_v_conv_proj"]:
            add("dyn_conv_proj", b(p+n+".weight"))
    add("conv_f32+norm+clip+misc", sum(b(n) for n in T if n.startswith(p) and any(s in n for s in ["_conv.weight","_clip","_norm","res_query","rel_bias"])))
    add("router_f32", b(p+"ffn_gate_inp.weight"))
    et = T[p+"ffn_gate_up_exps.weight"][1]
    add("experts8(" + et + ")", 8*(expert_slice(p+"ffn_gate_up_exps.weight") + expert_slice(p+"ffn_down_exps.weight")))
    if ploop:
        add("loop_ctrl+msg", sum(b(n) for n in T if n.startswith(p+"loop_")))
add("lm_head(" + T["output.weight"][1] + ")", b("output.weight"))
rowb = b("token_embd.weight") // T["token_embd.weight"][0][1]
add("embd_row", rowb)
tot = sum(fam.values())
print(inv)
for k, v in sorted(fam.items(), key=lambda x: -x[1]):
    print(f"  {k:40s} {v/1e6:10.2f} MB  {100*v/tot:5.1f}%")
print(f"  TOTAL per decoded token: {tot/1e6:.2f} MB ({tot/2**30:.3f} GiB)")
for name, bw in (("R9700 620", 620e9), ("8060S 230", 230e9)):
    print(f"  floor @ {name} GB/s: {tot/bw*1e3:.2f} ms/token = {bw/tot:.1f} tok/s")
# flops per token (MAC*2), excluding attention over context
nh, nk, hd, E, FF, NR, K = 24, 8, 128, 2304, 1152, 16, 4
macs_visit = E*nh*hd + 2*E*nk*hd + nh*hd*E + E*nh*hd + E*nh*NR + E*nh + E*128 + 8*(E*2*FF + FF*E)
macs_glob_extra = 2*E*K*nk
macs_ctrl = 4*(2*E*64 + 2*64*E)
macs_conv = (2*nk*hd + 2*E)*K
tot_macs = 48*macs_visit + 8*macs_glob_extra + 24*macs_ctrl + 48*macs_conv
print(f"  prefill FLOPs/token (no attn-ctx, no lm_head): {2*tot_macs/1e9:.2f} GFLOP; lm_head per output token {2*E*201024/1e9:.2f} GFLOP")
print(f"  attn-ctx FLOPs per token per global visit: {4*nh*hd:d} x n_kv ; local visit: {4*nh*hd} x min(n_kv,1024)")
print("  visit plan:", visits)
