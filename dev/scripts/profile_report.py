"""Bucket tables from the profiled sessions of profile_run.sh. usage: profile_report.py PROF_DIR ROOFLINE_JSON OUT_PREFIX [--gemm-tflops X]
Per scenario the difference of two sessions (scenario session minus its baseline: weight load, warm-up request and prompt prefill cancel).
Name-level merge: the HAL table has one row per kernel function, so one row holds every hook that launches it. The rows are split by launch
count and weight bytes (proj, oproj and the lm_head share gemv_q8_0 and gemm_q8_0); the lm_head launch time is taken from its own row in the
prefill sessions (one launch per request, 1.19 ms on the R9700). Floors: bytes / measured bandwidth, flops / the best GEMM rate."""
import json
import math
import re
import sys
from pathlib import Path

UNITS = {"ns": 1e-6, "us": 1e-3, "ms": 1.0, "s": 1e3}
ROW = re.compile(r"session=\d+ dispatch (\S+)\s+count=(\d+) total=([\d.]+) (ns|us|ms|s) avg=([\d.]+) (ns|us|ms|s)")
HERE = Path(__file__).resolve().parent
FAMS = [(f, re.compile(r)) for f, r in json.load(open(HERE / "profile_families.json"))]
# weight elements of one visit (Q8_0): q, k, v, gate, rel, oproj; proj has five of the six launches
ELEM = {"q": 3072 * 2304, "k": 1024 * 2304, "v": 1024 * 2304, "gate": 3072 * 2304, "r": 384 * 2304, "o": 2304 * 3072}
OPROJ_SHARE = ELEM["o"] / sum(ELEM.values())
MIN_LAUNCH_US = 0.7


def parse(path):
    text = Path(path).read_text(errors="replace")
    rows, lm = {}, []
    dropped = sum(int(x) for x in re.findall(r"dropped_records=(\d+)", text))
    for m in ROW.finditer(text):
        n, c, t, u = m.group(1), int(m.group(2)), float(m.group(3)), m.group(4)
        ms = t * UNITS[u]
        if n == "inkling_gemv_q8_0" and ms / c > 0.5:
            lm.append((c, ms))
            continue
        r = rows.setdefault(n, [0, 0.0])
        r[0] += c
        r[1] += ms
    return rows, lm, dropped


def family(n):
    for f, rx in FAMS:
        if rx.search(n):
            return f
    return "other"


def load(d, s):
    rows, lm, dropped = parse(Path(d) / f"{s}.serve.log")
    req = json.loads(Path(d, f"{s}.req.json").read_text().strip().splitlines()[-1])["requests"]
    return rows, lm, dropped, [r for r in req if r["tag"] not in ("warm", "c4_window")], req


def diff(a, b):
    out = {}
    for n in set(a) | set(b):
        ca, ta = a.get(n, [0, 0.0])
        cb, tb = b.get(n, [0, 0.0])
        out[n] = [ca - cb, ta - tb]
    return out


def lm_avg(*sessions):
    v = [ms / c for _, lm, _, _, _ in sessions for c, ms in lm if c]
    return sum(v) / len(v) if v else 0.0


def split_names(rows, units, lm_ms, decode):
    """name rows -> family dict {family: [launches, ms]}; apportion the shared kernels."""
    fam = {}

    def add(f, c, ms):
        e = fam.setdefault(f, [0.0, 0.0])
        e[0] += c
        e[1] += ms
    for n, (c, ms) in rows.items():
        if n in ("inkling_gemv_q8_0", "inkling_gemm_q8_0"):
            if decode and n == "inkling_gemv_q8_0":
                add("head", units, units * lm_ms)
                c -= units
                ms -= units * lm_ms
            add("oproj", c / 6, ms * OPROJ_SHARE)
            add("proj", c * 5 / 6, ms * (1 - OPROJ_SHARE))
        else:
            add(family(n), c, ms)
    return fam


def attn_kv_bytes(ctx):
    return 4096 * (8 * ctx + 40 * min(ctx, 1024)) + 48 * 4096


def run(d, prefix, roof_path, gemm_tflops):
    roof = json.load(open(roof_path))
    bw = roof["device"]["bandwidth_bytes_per_s"]
    ck = roof["checkpoints"]["ud-q4-k-xl"]["bytes_by_family"]
    MB = 1e6
    oproj_b = 48 * ELEM["o"] * 1.0625
    ex_b = ck["experts(NVFP4)"] + ck["experts(Q8_0)"]
    dec_bytes = {  # bytes per decoded token (batch of one row)
        "proj": ck["attn_proj(Q8_0)"] - oproj_b + ck["attn_proj(BF16)"] + ck["dyn_conv_proj"], "oproj": oproj_b,
        "ctrl": ck["loop_ctrl+msg"], "router": ck["router_f32"], "experts_gate_up": ex_b * 2 / 3, "experts_down": ex_b / 3,
        "head": ck["lm_head(Q8_0)"], "norm": ck["conv_f32+norm+clip+misc"], "post_conv": roof["checkpoints"]["ud-q4-k-xl"]["conv_state_bytes_per_token"],
        "embed": ck["embd_row"]}
    out_md = []
    res = {}

    def decode(name, ptag, dtag, ctx, rows_n=1):
        A, B = load(d, dtag), load(d, ptag)
        steps = sum(r["completion_tokens"] for r in A[3]) / max(1, len(A[3])) - sum(r["completion_tokens"] for r in B[3]) / max(1, len(B[3]))
        wall = sum(r["decode_wall_s"] for r in A[3]) * 1e3 / len(A[3]) / (sum(r["completion_tokens"] for r in A[3]) / len(A[3]) - 1) * steps
        if dtag.startswith("c4"):
            win = [r for r in A[4] if r["tag"] == "c4_window"][0]
            steps = sum(win["tokens_in_window"]) / 4
            wall = win["window_s"] * 1e3
            # the window excludes ramp and tail: use the steps of the difference for the busy side
            steps_busy = (sum(r["completion_tokens"] for r in A[3]) - sum(r["completion_tokens"] for r in B[3])) / 4
        else:
            steps_busy = steps
        rows = diff(A[0], B[0])
        fam = split_names(rows, steps_busy, lm_avg(A, B, load(d, "base")), True)
        busy = sum(v[1] for v in fam.values()) / steps_busy
        wall_step = wall / steps
        kv = attn_kv_bytes(ctx) * rows_n if rows_n == 1 else None
        t = [f"### {name}: {steps_busy:.0f} steps, wall {wall_step:.3f} ms/step ({1000 / wall_step * rows_n:.1f} tok/s), device busy {busy:.3f} ms/step, "
             f"launch gap + host {wall_step - busy:.3f} ms ({100 * (wall_step - busy) / wall_step:.1f} %), dropped_records {A[2]}",
             "", "| family | launches/step | busy ms | share % | floor ms | gap ms | us/launch |", "|---|---|---|---|---|---|---|"]
        tot_floor = 0.0
        resf = {}
        for f, (c, ms) in sorted(fam.items(), key=lambda kv: -kv[1][1]):
            cs, mss = c / steps_busy, ms / steps_busy
            b = dec_bytes.get(f, 0.0)
            if f == "attention":
                b = attn_kv_bytes(ctx) if rows_n == 1 else 0
            floor = b / bw * 1e3
            if rows_n > 1 and f in ("experts_gate_up", "experts_down"):
                floor = None
            tot_floor += floor or 0
            resf[f] = {"launches": round(cs, 2), "ms": round(mss, 4), "floor_ms": None if floor is None else round(floor, 4)}
            t.append(f"| {f} | {cs:.1f} | {mss:.3f} | {100 * mss / busy:.1f} | {'-' if floor is None else f'{floor:.3f}'} | {'-' if floor is None else f'{mss - floor:+.3f}'} | {mss / cs * 1e3 if cs else 0:.2f} |")
        t.append(f"| total | {sum(v[0] for v in fam.values()) / steps_busy:.0f} | {busy:.3f} | 100 | {tot_floor:.3f} | {busy - tot_floor:+.3f} | |")
        out_md.append("\n".join(t) + "\n")
        res[name] = {"steps": steps_busy, "wall_ms_step": wall_step, "busy_ms_step": busy, "floor_ms_step": tot_floor, "families": resf}

    def prefill(name, tag, ptoks_label):
        A, B = load(d, tag), load(d, "base")
        reqs = A[3]
        toks = sum(r["prompt_tokens"] for r in reqs)
        chunks = sum(math.ceil(r["prompt_tokens"] / 512) for r in reqs)
        wall = sum(r["total_s"] for r in reqs) * 1e3
        rows = diff(A[0], B[0])
        fam = split_names(rows, len(reqs), lm_avg(A, B), False)
        # the lm_head and the sampler run once per request; the separate lm rows are taken from the lm list difference
        lc = sum(c for c, _ in A[1]) - sum(c for c, _ in B[1])
        lms = sum(ms for _, ms in A[1]) - sum(ms for _, ms in B[1])
        e = fam.setdefault("head", [0.0, 0.0])
        e[0] += lc
        e[1] += lms
        busy = sum(v[1] for v in fam.values())
        ctx_mean = toks / len(reqs) / 2
        gemm_rate = gemm_tflops * 1e12
        macs_proj = 48 * (ELEM["q"] + ELEM["k"] + ELEM["v"] + ELEM["gate"] + ELEM["r"])
        macs_o = 48 * ELEM["o"]
        macs_experts = 48 * 8 * 3 * 2304 * 1152
        macs_ctrl = 24 * 4 * (2 * 2304 * 64 * 2)
        macs_router = 48 * 2304 * 128
        fl = {"proj": 2 * macs_proj, "oproj": 2 * macs_o, "experts_gate_up": 2 * macs_experts * 2 / 3, "experts_down": 2 * macs_experts / 3,
              "ctrl": 2 * macs_ctrl, "router": 2 * macs_router}
        byt = {"proj": dec_bytes["proj"], "oproj": dec_bytes["oproj"], "ctrl": dec_bytes["ctrl"], "router": dec_bytes["router"],
               "experts_gate_up": ex_b * 2 / 3 * 16, "experts_down": ex_b / 3 * 16, "norm": dec_bytes["norm"]}
        # attention flops of a prompt of n tokens (per token at position c: 12288 x (8 c + 40 min(c, 1024)))
        n_mean = toks / len(reqs)
        att_flops = 12288 * (8 * n_mean * n_mean / 2 + 40 * (n_mean * 1024 - 1024 * 1024 / 2 if n_mean > 1024 else n_mean * n_mean / 2))
        att_bytes_chunk = sum(4096 * (8 * (c0 + 512) + 40 * min(c0 + 512, 1023 + 512)) for c0 in range(0, int(n_mean), 512)) / math.ceil(n_mean / 512)
        t = [f"### {name}: {len(reqs)} request(s), {toks} prompt tokens, {chunks} chunks of up to 512, wall {wall:.1f} ms ({toks / wall * 1e3:.0f} tok/s), device busy {busy:.1f} ms, "
             f"launch gap + host {wall - busy:.1f} ms ({100 * (wall - busy) / wall:.1f} %), dropped_records {A[2]}",
             "", f"per chunk (GEMM rate for floors {gemm_tflops} TFLOP/s, bandwidth {bw / 1e9:.0f} GB/s)", "",
             "| family | launches/chunk | busy ms/chunk | share % | mem floor | flop floor | gap ms | us/launch |", "|---|---|---|---|---|---|---|---|"]
        tot = 0.0
        resf = {}
        for f, (c, ms) in sorted(fam.items(), key=lambda kv: -kv[1][1]):
            denom = chunks if f not in ("head", "sampler") else len(reqs)
            cs, mss = c / chunks, ms / chunks
            mem = byt.get(f, 0) / bw * 1e3
            fls = fl.get(f, 0) * 512 / gemm_rate * 1e3
            if f == "attention":
                mem = att_bytes_chunk / bw * 1e3
                fls = att_flops / math.ceil(n_mean / 512) / gemm_rate * 1e3
            if f in ("head", "sampler"):
                mem = byt.get(f, dec_bytes.get(f, 0)) / bw * 1e3 / chunks * len(reqs) if f == "head" else 0
            floor = max(mem, fls)
            tot += floor
            resf[f] = {"launches": round(cs, 2), "ms": round(mss, 3), "mem_ms": round(mem, 3), "flop_ms": round(fls, 3)}
            t.append(f"| {f} | {cs:.1f} | {mss:.3f} | {100 * ms / busy:.1f} | {mem:.3f} | {fls:.3f} | {mss - floor:+.3f} | {ms / c * 1e3 if c else 0:.2f} |")
        t.append(f"| total | {sum(v[0] for v in fam.values()) / chunks:.0f} | {busy / chunks:.3f} | 100 | | | {busy / chunks - tot:+.3f} (floor sum {tot:.3f}) | |")
        out_md.append("\n".join(t) + "\n")
        res[name] = {"tokens": toks, "chunks": chunks, "wall_ms": wall, "busy_ms": busy, "tok_s_wall": toks / wall * 1e3, "families": resf}

    have = {p.name[:-len(".req.json")] for p in Path(d).glob("*.req.json")}
    for nm, p, s, ctx in (("decode short", "dshort_p", "dshort_d", 40), ("decode 8k", "d8k_p", "d8k_d", 9100), ("decode 32k", "d32k_p", "d32k_d", 37000)):
        if p in have and s in have:
            decode(nm, p, s, ctx)
    if "c4_p" in have and "c4_d" in have:
        decode("4 rows decode", "c4_p", "c4_d", 230, rows_n=4)
    for nm, s in (("prefill 2k", "pf2k"), ("prefill 8k", "pf8k"), ("prefill 32k", "pf32k")):
        if s in have:
            prefill(nm, s, nm)
    Path(prefix + ".md").write_text("\n".join(out_md))
    Path(prefix + ".json").write_text(json.dumps(res, indent=1))
    print("\n".join(out_md))


if __name__ == "__main__":
    g = 120.0
    a = sys.argv[1:]
    if "--gemm-tflops" in a:
        i = a.index("--gemm-tflops")
        g = float(a[i + 1])
        del a[i:i + 2]
    run(a[0], a[2], a[1], g)
