"""Condensed bucket tables from profile_report.py JSON of both GPUs. usage: profile_condense.py R9700_REP.json STRIX_REP.json [OUT.md]
Buckets: experts (gate_up, down, routes, combine), proj (proj, oproj), attention (attention, attn_prep), small (mix, ctrl, norm, post_conv, router), head (head, sampler, embed, prep)."""
import json
import sys

BUCKETS = [("experts", ["experts_gate_up", "experts_down", "experts_prep", "combine"]), ("proj+oproj", ["proj", "oproj"]),
           ("attention+prep", ["attention", "attn_prep"]), ("small: mix, ctrl, norm, conv, router", ["mix", "ctrl", "norm", "post_conv", "router"]),
           ("head+sampler", ["head", "sampler", "embed", "prep"])]
SHARED_EXPERTS_MB = 1847.3


def floor_of(sc, fam, d, bw):
    if "floor_ms" in d:
        if d["floor_ms"] is None and fam.startswith("experts_"):
            return SHARED_EXPERTS_MB * 1e6 * (2 / 3 if fam == "experts_gate_up" else 1 / 3) / bw * 1e3
        return d["floor_ms"] or 0.0
    return max(d["mem_ms"], d["flop_ms"])


def row(sc, name, fams, bw):
    if not fams:
        return None
    ms = fl = la = 0.0
    for f in dict.fromkeys(fams):
        d = sc["families"].get(f)
        if not d:
            continue
        ms += d["ms"]
        fl += floor_of(sc, f, d, bw)
        la += d["launches"]
    return la, ms, fl


def main():
    r, s = (json.load(open(p)) for p in sys.argv[1:3])
    bw = {"r": 620e9, "s": 230e9}
    out = []
    for scn in r:
        a, b = r[scn], s.get(scn)
        dec = "steps" in a
        unit = "step" if dec else "512-token chunk"
        hdr = f"**{scn}** (ms per {unit}; R9700 wall {a['wall_ms_step']:.2f} ms/step, {1000 / a['wall_ms_step']:.1f} tok/s profiled)" if dec else \
            f"**{scn}** (ms per {unit}; R9700 {a['tok_s_wall']:.0f} tok/s, 8060S {b['tok_s_wall']:.0f} tok/s profiled)"
        if dec and b:
            hdr = f"**{scn}** (ms per {unit}; profiled wall R9700 {a['wall_ms_step']:.2f} ms ({1000 * (4 if '4 rows' in scn else 1) / a['wall_ms_step']:.0f} tok/s), 8060S {b['wall_ms_step']:.2f} ms ({1000 * (4 if '4 rows' in scn else 1) / b['wall_ms_step']:.0f} tok/s))"
        out += [hdr, "", "| bucket | launches | R9700 ms | floor | gap | share % | 8060S ms | floor | gap | share % |", "|---|---|---|---|---|---|---|---|---|---|"]
        ta = sum(d["ms"] for d in a["families"].values())
        tb = sum(d["ms"] for d in b["families"].values()) if b else 0
        sa = sb = fa = fb = 0.0
        for name, fams in BUCKETS:
            ra = row(a, name, fams, bw["r"])
            rb = row(b, name, fams, bw["s"]) if b else None
            if not ra:
                continue
            la, ma, fla = ra
            mb, flb = (rb[1], rb[2]) if rb else (0, 0)
            sa += ma; fa += fla; sb += mb; fb += flb
            out.append(f"| {name} | {la:.0f} | {ma:.2f} | {fla:.2f} | {ma - fla:+.2f} | {100 * ma / ta:.0f} | {mb:.2f} | {flb:.2f} | {mb - flb:+.2f} | {100 * mb / tb if tb else 0:.0f} |")
        out.append(f"| total busy | | {sa:.2f} | {fa:.2f} | {sa - fa:+.2f} | 100 | {sb:.2f} | {fb:.2f} | {sb - fb:+.2f} | 100 |")
        out.append("")
    text = "\n".join(out)
    if len(sys.argv) > 3:
        open(sys.argv[3], "w").write(text)
    print(text)


main()
