#!/usr/bin/env python3
"""Loop-aware roofline for Inkling 2 Mini (inkling2_loop). Standard library only.

usage: roofline_inkling.py [--root DIR] [--out bench] [--skills-commit SHA]
       roofline_inkling.py --crosscheck --skills ~/inkling/loom-skills/skills --gguf-dir ~/models/inkling2-mini/Mainline_Support

Writes bench/<gpu>/roofline.json (all checkpoints, with inputs, extras and measured Loom ratios) and
bench/<gpu>/ev_roofline_<variant>.json (a loom-card-evidence/2 roofline block: only what info_schema.json lists).
Formulas: docs/ROOFLINE.md. skills/loom-roofline/roofline.py ignores the loop and is not edited or imported here,
except by --crosscheck.
"""
import argparse, collections, hashlib, json, math, os, sys

NL, LS, LE, NP = 24, 6, 18, 3
GLOBAL = {5, 11, 17, 23}
WINDOW, KV_TOKEN_BYTES, CONV_STATE_BYTES = 1024, 4096, 3 * 6656 * 4
E, NH, NK, HD, FF, NR, KC, NEXP, NUSED, GATE_RANK = 2304, 24, 8, 128, 1152, 16, 4, 128, 8, 64
VOCAB_ROWS = 201024
CHUNKS = {"ud-q4-k-xl": 2048, "ud-q8-k-xl": 512}
C4_CTX = 230
CTXS = {"decode_2k": 2000, "decode_8k": 8000, "decode_32k": 32000}
PREFILL = {"prefill_1k": 1000, "prefill_2k": 2000, "prefill_8k": 8000, "prefill_16k": 16000, "prefill_32k": 32000}
# Copied from skills/loom-roofline/roofline.py (EFF, STRETCH, LAYER_US, KV_EFF) at the commit named below.
EFF = {"decode_moe": 0.78, "prefill": 0.33, "c4_moe": 0.85}
STRETCH = {"decode_moe": 0.88, "prefill": 0.45, "c4_moe": 0.95}
LAYER_US, KV_EFF = 35, 0.97
SKILLS_COMMIT = "618f89e134ea7bdc579f01f2d1825622df12912d"

# bandwidth: measured read bandwidth of this box (docs/DEVICES.md, scripts/floor_bench.py).
# peak: skills device presets (radeon-r9700 spec peak; strix-halo sustained WMMA at the 85 W mode, spec 59 TFLOP/s).
GPUS = {
    "radeon-r9700": dict(name="AMD Radeon AI PRO R9700 (gfx1201, 32 GB), measured read bandwidth", bw=620e9, flops=191.4e12,
                         spec=191.4e12, power=None, label="R9700"),
    "strix-halo": dict(name="AMD Ryzen AI Max+ 395 \"Strix Halo\" 8060S (gfx1151), measured read bandwidth", bw=230e9,
                       flops=43.8e12, spec=59e12, power="85 W", label="8060S"),
}
VARIANTS = {"ud-q4-k-xl": ("Inkling-2-Mini-UD-Q4_K_XL.gguf", "gguf_inventory_q4_k_xl.txt"),
            "ud-q8-k-xl": ("Inkling-2-Mini-UD-Q8_K_XL.gguf", "gguf_inventory_q8_k_xl.txt")}
# Loom measurements, docs/INTEGRATION_LOG.md (chat route, first measurements, before the wave 1 optimizations): decode ctx 16 and ctx 8005, prefill 520 tokens.
MEASURED = {
    ("ud-q4-k-xl", "radeon-r9700"): dict(decode=86.6, decode_ctx8005=74.9, prefill_520=2202),
    ("ud-q8-k-xl", "radeon-r9700"): dict(decode=69.7, decode_ctx8005=61.9, prefill_520=2186),
    ("ud-q4-k-xl", "strix-halo"): dict(decode=38.9, decode_ctx8005=33.3, prefill_520=900),
    ("ud-q8-k-xl", "strix-halo"): dict(decode=29.3, decode_ctx8005=26.1, prefill_520=833),
}


# Loom, Q4_K_XL, wave 1 build (prefill_capacity 2048), median of 3 sessions, docs/PERF_BASELINE.md (bead llama-cpp-tg3.30).
MEASURED_BASELINE = {
    ("ud-q4-k-xl", "radeon-r9700"): dict(decode=90.2, decode_2k=83.2, decode_8k=80.3, decode_32k=71.3, prefill_1k=2576.3, prefill_2k=2996.0, prefill_8k=3091.5, prefill_16k=2800.9, prefill_32k=2242.0, streams4=218.9),
    ("ud-q4-k-xl", "strix-halo"): dict(decode=39.7, decode_2k=36.7, decode_8k=34.9, decode_32k=29.7, prefill_1k=893.7, prefill_2k=1033.1, prefill_8k=1050.7, prefill_16k=973.1, prefill_32k=811.8, streams4=89.0),
}


def phys(v):
    return v if v < LS else (LS + (v - LS) % (LE - LS) if v < LS + NP * (LE - LS) else v - (NP - 1) * (LE - LS))


VISITS = [phys(v) for v in range(NL + (NP - 1) * (LE - LS))]
assert len(VISITS) == 48


def in_loop_pass2plus(vi):
    return LS <= vi < LS + NP * (LE - LS) and (vi - LS) // (LE - LS) >= 1


class Model:
    def __init__(self, inv_path):
        T = {}
        for line in open(inv_path, encoding="utf-8"):
            p = [x.strip() for x in line.split("|")]
            if len(p) == 4 and not line.startswith("#"):
                T[p[0]] = (tuple(int(x) for x in p[1].split("x")), p[2], int(p[3]))
        self.T = T
        self.file_bytes = sum(t[2] for t in T.values())

    def b(self, n):
        return self.T[n][2]

    def visit(self, vi, experts):
        """Weight bytes one layer visit reads, by family. experts: 'active' (8 of 128, decode) or 'all' (prefill chunk)."""
        T, l = self.T, VISITS[vi]
        p, fam = f"blk.{l}.", collections.defaultdict(int)
        for n in ["attn_q", "attn_k", "attn_v", "attn_output", "attn_gate", "attn_rel", "attn_sink_proj"]:
            fam["attn_proj(" + T[p + n + ".weight"][1] + ")"] += self.b(p + n + ".weight")
        if l in GLOBAL:
            for n in ["attn_k_conv_proj", "attn_v_conv_proj"]:
                fam["dyn_conv_proj"] += self.b(p + n + ".weight")
        fam["conv_f32+norm+clip+misc"] += sum(self.b(n) for n in T if n.startswith(p) and any(
            s in n for s in ["_conv.weight", "_clip", "_norm", "res_query", "rel_bias"]))
        fam["router_f32"] += self.b(p + "ffn_gate_inp.weight")
        et = T[p + "ffn_gate_up_exps.weight"][1]
        full = self.b(p + "ffn_gate_up_exps.weight") + self.b(p + "ffn_down_exps.weight")
        fam["experts(" + et + ")"] += full * NUSED // NEXP if experts == "active" else full
        if in_loop_pass2plus(vi):
            fam["loop_ctrl+msg"] += sum(self.b(n) for n in T if n.startswith(p + "loop_"))
        return fam

    def decode_families(self):
        fam = collections.defaultdict(int)
        for vi in range(48):
            for k, v in self.visit(vi, "active").items():
                fam[k] += v
        fam["lm_head(" + self.T["output.weight"][1] + ")"] += self.b("output.weight")
        fam["embd_row"] += self.b("token_embd.weight") // self.T["token_embd.weight"][0][1]
        return dict(fam)

    def head_bytes(self):
        return self.b("output.weight")

    def chunk_weight_bytes(self, n):
        """Weight bytes one prefill chunk of n tokens reads, without the lm_head. Expert tensors: the expected
        distinct experts of n tokens (each token picks 8 of 128), all of them for n >= 64 for practical purposes."""
        cover = 1 - (1 - NUSED / NEXP) ** n
        tot = 0.0
        for vi in range(48):
            for k, v in self.visit(vi, "all").items():
                tot += v * cover if k.startswith("experts") else v
        return tot


def kv_read_bytes(ctx):
    return KV_TOKEN_BYTES * (8 * ctx + 40 * min(ctx, WINDOW))


def decode_bytes(m, ctx, streams=1, shared=True):
    """Bytes of one decode pass for `streams` rows. Weights once; per row: KV read at ctx, KV write, conv state read+write."""
    fam = m.decode_families()
    w = sum(fam.values())
    if streams > 1 and not shared:
        ex = sum(v for k, v in fam.items() if k.startswith("experts"))
        cover = NEXP * (1 - (1 - NUSED / NEXP) ** streams) / NUSED
        w += ex * (cover - 1)
    per_row = kv_read_bytes(ctx) + KV_TOKEN_BYTES * 48 + 2 * CONV_STATE_BYTES * 48
    return w + streams * per_row


MACS_VISIT = E * NH * HD + 2 * E * NK * HD + NH * HD * E + E * NH * HD + E * NH * NR + E * NH + E * NEXP + NUSED * (E * 2 * FF + FF * E)
MACS_GLOBAL_EXTRA = 2 * E * KC * NK
MACS_CTRL = 4 * (2 * E * GATE_RANK + 2 * GATE_RANK * E)
MACS_CONV = (2 * NK * HD + 2 * E) * KC
DENSE_FLOPS_PER_TOKEN = 2 * (48 * MACS_VISIT + 8 * MACS_GLOBAL_EXTRA + 24 * MACS_CTRL + 48 * MACS_CONV)
HEAD_FLOPS = 2 * E * VOCAB_ROWS
ATTN_FLOPS_PER_PAIR = 4 * NH * HD


def attn_pairs(c0, n):
    """(global, local) key counts summed over the n tokens of a chunk that starts at context c0 (causal, window 1024)."""
    g = sum(c0 + i + 1 for i in range(n))
    loc = sum(min(c0 + i + 1, WINDOW) for i in range(n))
    return g, loc


def prefill_time(m, P, bw, flops, chunk):
    """Seconds to prefill P tokens in chunks of `chunk` tokens (prefill_capacity): per chunk max(compute, bytes / bw).
    Returns (seconds, compute-only seconds, memory-only seconds)."""
    t_total = t_c = t_m = 0.0
    c0 = 0
    while c0 < P:
        n = min(chunk, P - c0)
        g, loc = attn_pairs(c0, n)
        fl = n * DENSE_FLOPS_PER_TOKEN + ATTN_FLOPS_PER_PAIR * (8 * g + 40 * loc)
        by = m.chunk_weight_bytes(n) + KV_TOKEN_BYTES * (8 * (c0 + n) + 40 * min(c0 + n, WINDOW - 1 + n)) + KV_TOKEN_BYTES * 48 * n
        if c0 + n >= P:
            fl += HEAD_FLOPS
            by += m.head_bytes()
        t_total += max(fl / flops, by / bw)
        t_c += fl / flops
        t_m += by / bw
        c0 += n
    return t_total, t_c, t_m


def eff_decode(ceiling, ctx, m, streams_weight_bytes):
    floor_ms = 48 * LAYER_US / 1000
    e = min(EFF["decode_moe"], (1000 / (1000 / ceiling + floor_ms)) / ceiling)
    if ctx:
        share = kv_read_bytes(ctx) / decode_bytes(m, ctx)
        e = 1 / ((1 - share) / e + share / max(KV_EFF, e))
    return e


def build(m, gpu, chunk):
    g = GPUS[gpu]
    bw, flops = g["bw"], g["flops"]
    metrics, extras = {}, {}
    fam = m.decode_families()
    wb = sum(fam.values())

    def put(key, ceiling, eff, stretch, **kw):
        metrics[key] = dict(ceiling=round(ceiling, 1), efficiency=round(eff, 3), target=round(ceiling * eff, 1),
                            stretch=round(ceiling * stretch, 1), **kw)

    def decode_row(key, ctx):
        by = decode_bytes(m, ctx)
        c = bw / by
        stretch = STRETCH["decode_moe"]
        if ctx:
            share = kv_read_bytes(ctx) / by
            stretch = 1 / ((1 - share) / stretch + share / max(KV_EFF, stretch))
        put(key, c, eff_decode(c, ctx, m, wb), stretch, bytes_per_token=round(by), context=ctx)

    decode_row("decode", 0)
    for k, ctx in CTXS.items():
        decode_row(k, ctx)
    sc = g["spec"] / flops
    for k, P in PREFILL.items():
        t, tc, tm = prefill_time(m, P, bw, flops, chunk)
        put(k, P / t, min(1.0, EFF["prefill"] * sc), min(1.0, STRETCH["prefill"] * sc), prompt_tokens=P,
            compute_only=round(P / tc, 1), memory_only=round(P / tm, 1), bound="memory" if tm > tc else "compute")
    by4 = decode_bytes(m, C4_CTX, streams=4)
    put("streams4", 4 * bw / by4, EFF["c4_moe"], STRETCH["c4_moe"], basis="shared-experts", bytes_per_pass=round(by4), context=C4_CTX,
        independent_routing=round(4 * bw / decode_bytes(m, C4_CTX, 4, shared=False), 1))
    t520, tc520, tm520 = prefill_time(m, 520, bw, flops, chunk)
    extras["prefill_520"] = round(520 / t520, 1)
    extras["decode_ctx8005"] = round(bw / decode_bytes(m, 8005), 1)
    extras["decode_ctx16"] = round(bw / decode_bytes(m, 16), 1)
    return dict(weight_bytes_per_decode_token=wb, bytes_by_family={k: v for k, v in sorted(fam.items(), key=lambda x: -x[1])},
                dense_flops_per_token=DENSE_FLOPS_PER_TOKEN, head_flops_per_output_token=HEAD_FLOPS,
                conv_state_bytes_per_token=2 * CONV_STATE_BYTES * 48, kv_bytes_per_context_token_global_visits=8 * KV_TOKEN_BYTES,
                metrics=metrics, checks_at_measured_points=extras)


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def run(args):
    root = os.path.abspath(args.root)
    me = os.path.abspath(__file__)
    tool = {"script": "scripts/roofline_inkling.py", "sha256": sha(me), "skills_commit": args.skills_commit}
    problems = []
    for gpu, g in GPUS.items():
        out = {"schema": "inkling-roofline/1", "gpu": gpu,
               "device": {"name": g["name"], "bandwidth_bytes_per_s": g["bw"], "peak_flops": g["flops"],
                          **({"power_mode": g["power"], "spec_peak_flops": g["spec"]} if g["power"] else {})},
               "tool": tool, "chunk_tokens": CHUNKS, "checkpoints": {}}
        d = os.path.join(root, args.out, gpu)
        os.makedirs(d, exist_ok=True)
        for var, (gguf, inv) in VARIANTS.items():
            m = Model(os.path.join(root, "docs", inv))
            r = build(m, gpu, CHUNKS[var])
            meas = MEASURED[(var, gpu)]
            ratios = {}
            met = r["metrics"]
            ratios["decode"] = meas["decode"] / met["decode"]["ceiling"]
            ratios["decode_ctx8005"] = meas["decode_ctx8005"] / r["checks_at_measured_points"]["decode_ctx8005"]
            ratios["prefill_520"] = meas["prefill_520"] / r["checks_at_measured_points"]["prefill_520"]
            r["measured_loom"] = dict(meas, ratio_to_ceiling={k: round(v, 3) for k, v in ratios.items()})
            mb = MEASURED_BASELINE.get((var, gpu))
            if mb:
                r["measured_loom_wave1"] = dict(values=mb, ratio_to_ceiling={k: round(v / met[k]["ceiling"], 3) for k, v in mb.items()})
                problems += [f"{var} {gpu} {k}" for k, v in mb.items() if v / met[k]["ceiling"] >= 0.97]
            for k, v in ratios.items():
                if v >= 0.97:
                    problems.append(f"{var} {gpu} {k} {v:.3f}")
            out["checkpoints"][var] = dict(checkpoint=gguf, **r)
            block = {"device": out["device"], "tool": tool, "checkpoint": gguf, "drafting": None,
                     "metrics": {k: {kk: x[kk] for kk in ("ceiling", "efficiency", "basis") if kk in x}
                                 for k, x in met.items() if k != "decode_8k"}}
            json.dump({"evidence": "loom-card-evidence/2", "gpu": gpu, "roofline": block},
                      open(os.path.join(d, f"ev_roofline_{var}.json"), "w"), indent=1)
        json.dump(out, open(os.path.join(d, "roofline.json"), "w"), indent=1)
        for var, c in out["checkpoints"].items():
            print(f"{g['label']:6} {var}: {c['weight_bytes_per_decode_token']/1e6:.2f} MB/token; " + ", ".join(
                f"{k} {x['ceiling']}" for k, x in c["metrics"].items()) + "; measured/ceiling " + str(c["measured_loom"]["ratio_to_ceiling"]))
    if problems:
        print("EFFICIENCY >= 0.97:", problems)
        return 1
    return 0


def crosscheck(args):
    """Substitute the loop into roofline.py's model of the same GGUF and compare bytes, KV bytes and flops with this script."""
    sys.path.insert(0, os.path.join(args.skills, "loom-roofline"))
    import roofline as R
    root = os.path.abspath(args.root)
    ok = True
    for var, (gguf, inv) in VARIANTS.items():
        m = Model(os.path.join(root, "docs", inv))
        meta, tens = R.load(os.path.join(args.gguf_dir, gguf))
        base = R.shape_of(meta, tens, {})
        geo = base["geo"]

        def layer_bytes(l, ctrl):
            sub = {n: v for n, v in tens.items() if n.startswith(f"blk.{l}.") and (ctrl or ".loop_" not in n)}
            return R.weight_bytes(sub, geo)

        wb = params = 0.0
        for vi, l in enumerate(VISITS):
            b, p, _, _ = layer_bytes(l, in_loop_pass2plus(vi))
            wb, params = wb + b, params + p
        rest = {n: v for n, v in tens.items() if not n.startswith("blk.")}
        b, p, head_b, _ = R.weight_bytes(rest, geo)
        wb, params = wb + b, params + p
        ours = sum(m.decode_families().values())
        rows = [("weights bytes/token", wb, ours)]
        geo48 = dict(geo, L=48, attn=set(range(48)), sliding={vi for vi, l in enumerate(VISITS) if l not in GLOBAL},
                     kvh_l=[8] * 48)
        for ctx in (2000, 8000, 32000):
            rows.append((f"KV bytes ctx {ctx} (roofline.py kv_read_bytes with 48 visits, window {geo['window']})",
                         R.kv_read_bytes(geo48, ctx), kv_read_bytes(ctx)))
        shape = dict(geo=geo48, wb=wb, params=params, head=head_b, detail={}, head_params=base["head_params"])
        for P in (1000, 8000, 32000):
            rows.append((f"prefill flops/token P={P} (roofline.py 2*(params-head)+attn, window clamp)",
                         2 * (params - base["head_params"]) + R.attn_flops_per_token(geo48, P),
                         DENSE_FLOPS_PER_TOKEN + ATTN_FLOPS_PER_PAIR * (8 * attn_pairs(0, P)[0] + 40 * attn_pairs(0, P)[1]) / P))
        for gpu, g in GPUS.items():
            c_theirs = R.ceilings(shape, g["bw"], g["flops"], (1000, 2000, 32000))
            c_ours = build(m, gpu)["metrics"]
            rows.append((f"{g['label']} decode_tps (roofline.py ceilings() with loop-substituted bytes vs ours)",
                         c_theirs["decode_tps"], c_ours["decode"]["ceiling"]))
            rows.append((f"{g['label']} decode_tps_2k", c_theirs["decode_tps_2k"], c_ours["decode_2k"]["ceiling"]))
            rows.append((f"{g['label']} decode_tps_32k", c_theirs["decode_tps_32k"], c_ours["decode_32k"]["ceiling"]))
            rows.append((f"{g['label']} prefill_1000 compute-only", c_theirs["prefill_1000"], c_ours["prefill_1k"]["compute_only"]))
        print(f"== {var}: roofline.py as shipped (no loop): {base['wb']/1e6:.1f} MB/token")
        print(f"   {'quantity':92} {'roofline.py+loop':>16} {'ours':>16} {'diff %':>7}")
        for name, a, b2 in rows:
            d = 100 * (b2 / a - 1)
            print(f"   {name:92} {a:16.6g} {b2:16.6g} {d:7.2f}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
    ap.add_argument("--out", default="bench")
    ap.add_argument("--skills-commit", default=SKILLS_COMMIT)
    ap.add_argument("--crosscheck", action="store_true")
    ap.add_argument("--skills")
    ap.add_argument("--gguf-dir")
    a = ap.parse_args()
    sys.exit(crosscheck(a) if a.crosscheck else run(a))


if __name__ == "__main__":
    main()
