#!/usr/bin/env python3
"""Attention kernel tests against oracle tensors. Runs on the box.

usage: attn_test.py run [--dev gfx1151|gfx1201] [--only REGEX] [--long DIR]
The raw projection outputs, the post-prep tensors and the attention output come from the oracle dumps (inkB
tree, or --long for the 1332 token dump) and the Q8_K_XL GGUF. A test is a chain of the real kernels (prep,
attention, merge, conv commit, KV append) over one or more chunks; outputs are written by
check.file.write.npy and compared here (NMSE, max abs).
"""
import argparse, json, os, re, subprocess, sys
import numpy as np

HOME = os.path.expanduser("~")
ROOT = os.environ.get("ATTN_ROOT", f"{HOME}/inkling/inkE")
ORACLE = os.environ.get("ATTN_ORACLE", f"{HOME}/inkling/inkB/oracle/dumps")
LONG = os.environ.get("ATTN_LONG", f"{HOME}/inkling/inkE/oracle_long/Q8_K_XL_L1")
GGUF = f"{HOME}/models/inkling2-mini/Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf"
FIX = f"{ROOT}/fixtures"
OUT = f"{ROOT}/out"
DEVICES = {"gfx1151": "amdgpu://id/000000e2000000808e270f2b0bbb4e00", "gfx1201": "amdgpu://GPU-e0880c05003c324d"}
SRC = ["package/kernel/attn/attn_prep.loom", "package/kernel/attn/attn_conv_commit.loom", "package/kernel/attn/attn_tile.loom",
       "package/kernel/attn/attn_merge.loom", "package/kernel/attn/attn_kv_append.loom", "package/lib/inkling/attn_common.loom",
       "package/lib/inkling/attn_matrix.loom", "package/lib/inkling/attn_targets.loom"]
CONV_ORIGIN = 6 << 20
SLOT0 = 16 << 20
SLOT_STRIDE = 8 << 20
STATE_HALVES = (SLOT0 + 2 * SLOT_STRIDE) // 2

_reader = None


def weight(name):
    global _reader
    import gguf
    if _reader is None:
        _reader = gguf.GGUFReader(GGUF)
    for t in _reader.tensors:
        if t.name == name:
            d = np.array(t.data)
            shape = [int(n) for n in t.shape][::-1]
            if t.tensor_type == gguf.GGMLQuantizationType.BF16:
                d = (np.ascontiguousarray(d).view(np.uint16).astype(np.uint32) << 16).view(np.float32)
            return d.astype(np.float32).reshape(shape)
    raise KeyError(name)


def bf16_round(x):
    u = np.ascontiguousarray(x.astype(np.float32)).view(np.uint32)
    r = (u + 0x7FFF + ((u >> 16) & 1)) & 0xFFFF0000
    return r.astype(np.uint32).view(np.float32)


class Oracle:
    def __init__(self, path):
        self.dir = path
        self.m = json.load(open(f"{self.dir}/manifest.json"))
        self.idx = {(t["name"], t["occurrence"]): t for t in self.m["tensors"]}

    def has(self, name, occ=0):
        return (name, occ) in self.idx

    def get(self, name, occ=0):
        t = self.idx[(name, occ)]
        dt = np.float32 if t["dtype"] == "f32" else np.int32
        a = np.fromfile(f"{self.dir}/{t['file']}", dtype=dt).reshape(t["ne"][::-1])
        while a.ndim > 1 and a.shape[0] == 1:
            a = a[0]
        return a

    @property
    def n_tokens(self):
        return self.m["n_tokens"]


def phys(slot):
    return slot if slot < 24 else 6 + (slot - 24) % 12


def is_global(slot):
    return phys(slot) in (5, 11, 17, 23)


def npy(path, a):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.save(path, np.ascontiguousarray(a))


def spans_table(spans, cap=4):
    """spans: list of (first, length, slot_byte_offset, position, row)."""
    w = np.zeros(4 + 8 * cap, np.int32)
    w[0] = len(spans)
    w[1] = sum(s[1] for s in spans)
    for i, (first, length, slot, pos, row) in enumerate(spans):
        b = 4 + 8 * i
        w[b:b + 8] = [first, length, slot & 0xFFFFFFFF, slot >> 32, slot & 0xFFFFFFFF, slot >> 32, pos, row]
    return w


def rows_table(rows=2, pages=32):
    w = np.zeros(32 + rows * pages, np.int32)
    for r in range(rows):
        w[10 * r + 2] = (SLOT0 + r * SLOT_STRIDE) & 0xFFFFFFFF
        w[10 * r + 3] = (SLOT0 + r * SLOT_STRIDE) >> 32
        w[10 * r + 8] = 128 + r * pages * 4
        w[32 + r * pages:32 + (r + 1) * pages] = r * pages + np.arange(pages)
    return w


class Seq:
    """All tensors of one visit (cache slot) for a prompt."""

    def __init__(self, path, slot, gate_seed=1):
        o = Oracle(path)
        p = phys(slot)
        T = self.T = o.n_tokens
        self.slot, self.glob = slot, is_global(slot)
        self.q_raw = o.get(f"proj_q-{slot}").reshape(T, 3072)
        self.k_raw = o.get(f"conv_in-{slot}", 0).reshape(T, 1024)
        self.v_raw = o.get(f"conv_in-{slot}", 1).reshape(T, 1024)
        self.r_raw = o.get(f"proj_r-{slot}").reshape(T, 384)
        qocc = 2 if self.glob else 0
        xq = o.get(f"norm-{slot}", qocc).reshape(T, 2304) * weight(f"blk.{p}.attn_{'q_inp_norm' if self.glob else 'norm'}.weight")
        if self.glob:
            xk = o.get(f"norm-{slot}", 0).reshape(T, 2304) * weight(f"blk.{p}.attn_k_inp_norm.weight")
            xv = o.get(f"norm-{slot}", 1).reshape(T, 2304) * weight(f"blk.{p}.attn_v_inp_norm.weight")
            self.dyn_k = bf16_round(xk) @ bf16_round(weight(f"blk.{p}.attn_k_conv_proj.weight")).T
            self.dyn_v = bf16_round(xv) @ bf16_round(weight(f"blk.{p}.attn_v_conv_proj.weight")).T
        else:
            self.dyn_k = self.dyn_v = np.zeros((T, 32), np.float32)
        self.sink = bf16_round(xq) @ bf16_round(weight(f"blk.{p}.attn_sink_proj.weight")).T
        self.sink_check = None
        if o.has(f"kq_b-{slot}"):
            kqb = o.get(f"kq_b-{slot}")
            self.sink_check = float(np.abs(-kqb[:, :, -1].T - self.sink).max())
        rng = np.random.default_rng(gate_seed)
        self.gate = rng.normal(0, 1.5, (T, 3072)).astype(np.float32)
        self.w = dict(
            w_q_clip=weight(f"blk.{p}.attn_q_clip.weight").reshape(24, 128), w_rel_clip=weight(f"blk.{p}.attn_rel_clip.weight").reshape(24, 16),
            w_k_conv=weight(f"blk.{p}.attn_k_conv.weight").reshape(1024, 4), w_v_conv=weight(f"blk.{p}.attn_v_conv.weight").reshape(1024, 4),
            rel_bias=weight(f"blk.{p}.attn_rel_bias.weight").reshape(1024, 16))
        self.q_exp = o.get(f"Qcur-{slot}").reshape(T, 3072)
        self.k_exp = o.get(f"Kcur-{slot}").reshape(T, 1024)
        self.v_exp = o.get(f"Vcur-{slot}").reshape(T, 1024)
        self.r_exp = o.get(f"Rcur-{slot}").reshape(T, 384)
        kqv = o.get(f"kqv_out-{slot}").reshape(T, 3072)
        self.out_exp = kqv * (1.0 / (1.0 + np.exp(-self.gate)))


def ty(a, dt="f32"):
    return "tensor<" + "x".join(str(s) for s in a.shape) + "x" + dt + ">"


class Case:
    """Builds the .loom source of one check.case."""

    def __init__(self, name):
        self.name = name
        self.lines, self.vals, self.decls = [], {}, {}

    def read(self, var, path, a, dt="f32"):
        self.lines.append(f'  %{var} = check.file.read.npy path("{path}") : {ty(a, dt)}')
        self.vals[var] = ty(a, dt)

    def zeros(self, var, shape, dt="f32"):
        t = "tensor<" + "x".join(map(str, shape)) + "x" + dt + ">"
        self.lines.append(f"  %{var} = check.generate.fill value(0.0) : {t}")
        self.vals[var] = t

    def lit(self, var, value, t="index"):
        self.lines.append(f"  %{var} = check.literal value({value}) : {t}")
        self.vals[var] = t

    def launch(self, kernel, decl_launch, workload, args):
        if kernel not in self.decls:
            wl = ", ".join(f"%w{i}: {self.vals[v]}" for i, v in enumerate(workload))
            self.decls[kernel] = f"kernel.decl @{kernel}({wl}) launch({decl_launch})"
        wlt = ", ".join(self.vals[v] for v in workload)
        at = ", ".join(self.vals[v] for v in args)
        self.lines.append(f"  kernel.launch @{kernel}[{', '.join('%' + v for v in workload)}]({', '.join('%' + v for v in args)}) : [{wlt}]({at})")

    def write(self, var, path):
        self.lines.append(f'  check.file.write.npy value(%{var}) path("{path}") mode(always) : {self.vals[var]}')

    def text(self):
        return "\n".join(self.decls.values()) + f"\ncheck.case public @{self.name} {{\n" + "\n".join(self.lines) + "\n  check.return\n}\n"


DECL = {
    "prep": ("inkling_attn_prep", "%cap: index, %span_cap: index, %global: index, %conv_origin: offset, %spans: buffer, %state: buffer, "
             "%q_raw: buffer, %k_raw: buffer, %v_raw: buffer, %r_raw: buffer, %dyn_k: buffer, %dyn_v: buffer, "
             "%w_q_clip: buffer, %w_rel_clip: buffer, %w_k_conv: buffer, %w_v_conv: buffer, %q_out: buffer, "
             "%r_out: buffer, %k_act: buffer, %v_act: buffer"),
    "tile": ("inkling_attn_tile", "%cap: index, %span_cap: index, %mode: index, %global: index, %split_blocks: index, %max_splits: index, "
             "%max_blocks: index, %kv_origin: offset, %spans: buffer, %rows: buffer, %state: buffer, %q: buffer, %r: buffer, "
             "%k_act: buffer, %v_act: buffer, %rel_bias: buffer, %sink: buffer, %gate: buffer, %out: buffer, %partial: buffer"),
    "merge": ("inkling_attn_merge", "%cap: index, %span_cap: index, %global: index, %split_blocks: index, %max_splits: index, "
              "%spans: buffer, %sink: buffer, %gate: buffer, %out: buffer, %partial: buffer"),
    "conv": ("inkling_attn_conv_commit", "%span_cap: index, %cap: index, %conv_origin: offset, %spans: buffer, %state: buffer, "
             "%k_raw: buffer, %v_raw: buffer"),
    "append": ("inkling_attn_kv_append", "%cap: index, %span_cap: index, %global: index, %max_blocks: index, %kv_origin: offset, "
               "%spans: buffer, %rows: buffer, %state: buffer, %k_act: buffer, %v_act: buffer"),
    "post": ("inkling_attn_post", "%cap: index, %span_cap: index, %global: index, %split_blocks: index, %max_splits: index, %max_blocks: index, "
             "%kv_origin: offset, %conv_origin: offset, %spans: buffer, %rows: buffer, %state: buffer, %sink: buffer, %gate: buffer, %out: buffer, "
             "%partial: buffer, %k_act: buffer, %v_act: buffer, %k_raw: buffer, %v_raw: buffer"),
}
PER_CHUNK = ("spans", "q_raw", "k_raw", "v_raw", "r_raw", "dyn_k", "dyn_v", "sink", "gate", "q", "r", "k_act", "v_act", "out", "partial", "cap", "post")


def nmse(a, b):
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    return float(((a - b) ** 2).sum() / max((b ** 2).sum(), 1e-300))


def run_loom(case, dev, extra_srcs=SRC):
    os.makedirs(f"{ROOT}/build", exist_ok=True)
    test = f"{ROOT}/build/{case.name}_cases.loom"
    open(test, "w").write(case.text())
    merged = f"{ROOT}/build/{case.name}_{dev}.loom"
    venv = f"{HOME}/loom-serve/.venv-inkE/bin"
    r = subprocess.run([f"{venv}/loom-link", test] + extra_srcs + ["--mode=merge", "--to=text", f"--output={merged}"],
                       capture_output=True, text=True, cwd=ROOT)
    if r.returncode:
        print(r.stderr[-3000:])
        return False
    r = subprocess.run(["flock", "-s", f"{HOME}/hrx-pr1/gpu.lock", f"{venv}/loom-test", merged, f"--device={DEVICES[dev]}",
                        f"--file-output-root={OUT}/{dev}"], capture_output=True, text=True, cwd=ROOT)
    ok = r.returncode == 0
    if not ok:
        print(r.stdout[-1500:], r.stderr[-2500:])
    return ok


def run_epochs(dev, name, seqs, epochs, bps=8):
    """Run prep, attention, merge, conv commit and KV append once per epoch.

    seqs: key -> Seq (same visit). epochs: list of span lists [(key, a, b, row)]; the spans of one epoch are packed in order.
    Returns, per epoch, a list of per-span output dicts.
    """
    first = next(iter(seqs.values()))
    glob = first.glob
    span_cap = 4
    T = max(s.T for s in seqs.values())
    nb_total = T // 64 + 1
    max_splits = max(1, -(-(nb_total if glob else min(17, nb_total)) // bps))
    c = Case(name)
    d = f"{FIX}/{name}"
    c.zeros("state", [STATE_HALVES], "f16")
    rows = rows_table()
    npy(f"{d}/rows.npy", rows)
    c.read("rows", f"{d}/rows.npy", rows, "i32")
    for k, v in first.w.items():
        npy(f"{d}/{k}.npy", v)
        c.read(k, f"{d}/{k}.npy", v)
    for k, v, t in (("span_cap", span_cap, "index"), ("global", int(glob), "index"), ("conv_origin", CONV_ORIGIN, "offset"),
                    ("kv_origin", 0, "offset"), ("split_blocks", bps, "index"), ("max_splits", max_splits, "index"),
                    ("max_blocks", 32, "index"), ("mode0", 0, "index"), ("mode1", 1, "index"), ("kv8", 8, "index"),
                    ("heads", 24, "index"), ("one", 1, "index")):
        c.lit(k, v, t)
    for i, spans in enumerate(epochs):
        p = f"c{i}_"
        S = lambda names: [p + x if x in PER_CHUNK else x for x in names.split()]
        n = sum(b - a for _, a, b, _ in spans)
        offs, o = [], 0
        for _, a, b, _ in spans:
            offs.append(o)
            o += b - a
        sp = spans_table([(offs[j], b - a, SLOT0 + row * SLOT_STRIDE, a, row) for j, (_, a, b, row) in enumerate(spans)], span_cap)
        cat = lambda f: np.concatenate([getattr(seqs[k], f)[a:b] for k, a, b, _ in spans])
        for k, v in dict(spans=sp, q_raw=cat("q_raw"), k_raw=cat("k_raw"), v_raw=cat("v_raw"), r_raw=cat("r_raw"), dyn_k=cat("dyn_k"),
                         dyn_v=cat("dyn_v"), sink=cat("sink"), gate=cat("gate")).items():
            npy(f"{d}/{p}{k}.npy", v)
            c.read(p + k, f"{d}/{p}{k}.npy", v, "i32" if k == "spans" else "f32")
        c.zeros(p + "q", [n, 3072])
        c.zeros(p + "r", [n, 384])
        c.zeros(p + "k_act", [n, 1024], "f16")
        c.zeros(p + "v_act", [n, 1024], "f16")
        c.zeros(p + "out", [n, 3072])
        c.zeros(p + "partial", [span_cap * 24 * max_splits * 136])
        c.lit(p + "cap", n)
        c.lit(p + "post", span_cap * 40 + n)
        c.lit(p + "tiles", sum(-(-(b - a) // 16) for _, a, b, _ in spans) + span_cap)
        c.launch(*DECL["prep"], [p + "cap"], S("cap span_cap global conv_origin spans state q_raw k_raw v_raw r_raw dyn_k dyn_v "
                                               "w_q_clip w_rel_clip w_k_conv w_v_conv q r k_act v_act"))
        tile = "cap span_cap {m} global split_blocks max_splits max_blocks kv_origin spans rows state q r k_act v_act rel_bias sink gate out partial"
        c.launch(*DECL["tile"], [p + "tiles", "heads", "one"], S(tile.format(m="mode0")))
        c.launch(*DECL["tile"], ["span_cap", "kv8", "max_splits"], S(tile.format(m="mode1")))
        if os.environ.get("ATTN_SEPARATE"):
            c.launch(*DECL["merge"], ["span_cap"], S("cap span_cap global split_blocks max_splits spans sink gate out partial"))
            c.launch(*DECL["conv"], ["span_cap"], S("span_cap cap conv_origin spans state k_raw v_raw"))
            c.launch(*DECL["append"], [p + "cap"], S("cap span_cap global max_blocks kv_origin spans rows state k_act v_act"))
        else:
            c.launch(*DECL["post"], [p + "post"], S("cap span_cap global split_blocks max_splits max_blocks kv_origin conv_origin spans rows state sink gate out partial k_act v_act k_raw v_raw"))
        for x in ("q", "r", "k_act", "v_act", "out"):
            c.write(p + x, f"{name}/{p}{x}.npy")
    if not run_loom(c, dev):
        return None
    res = []
    for i, spans in enumerate(epochs):
        arr = {x: np.load(f"{OUT}/{dev}/{name}/c{i}_{x}.npy").astype(np.float32) for x in ("q", "r", "k_act", "v_act", "out")}
        o, per = 0, []
        for _, a, b, _ in spans:
            per.append({x: v[o:o + b - a] for x, v in arr.items()})
            o += b - a
        res.append(per)
    return res


def expected(seq, a, b):
    return dict(q=seq.q_exp[a:b], r=seq.r_exp[a:b], k_act=seq.k_exp[a:b], v_act=seq.v_exp[a:b], out=seq.out_exp[a:b])


TOL = dict(q=1e-6, r=1e-6, k_act=2e-7, v_act=2e-7, out=1e-6)


def check_epochs(results, dev, label, seqs, epochs, bps=8):
    name = re.sub(r"[^A-Za-z0-9_]", "_", label)
    res = run_epochs(dev, name, seqs, epochs, bps)
    got = {}
    for x in TOL:
        if res is None:
            results.append((f"{label} {x}", dev, None, TOL[x], False))
            continue
        errs = []
        for e, spans in enumerate(epochs):
            for j, (k, a, b, _) in enumerate(spans):
                errs.append((res[e][j][x], expected(seqs[k], a, b)[x]))
        v = nmse(np.concatenate([g.ravel() for g, _ in errs]), np.concatenate([w.ravel() for _, w in errs]))
        results.append((f"{label} {x}", dev, v, TOL[x], v < TOL[x]))
    if res is not None:
        got = {k: np.concatenate([res[e][j]["out"] for e, spans in enumerate(epochs) for j, (kk, a, b, _) in enumerate(spans) if kk == k])
               for k in seqs}
    return got


def main():
    global LONG
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run"])
    ap.add_argument("--dev", default="gfx1201")
    ap.add_argument("--only", default="")
    ap.add_argument("--long", default=LONG)
    a = ap.parse_args()
    LONG = a.long
    results, diffs = [], []
    sel = lambda s: re.search(a.only, s)
    for slot in (0, 5):
        if not any(sel(f"A slot {slot} {x}") for x in ("1chunk", "2chunks", "decode")):
            continue
        seq = Seq(f"{ORACLE}/Q8_K_XL_A", slot)
        print(f"slot {slot}: sink check max abs {seq.sink_check:.2e}")
        for label, chunks in (("1chunk", [(0, 16)]), ("2chunks", [(0, 6), (6, 16)]), ("decode", [(0, 10)] + [(i, i + 1) for i in range(10, 16)])):
            nm = f"A slot {slot} {label}"
            if sel(nm):
                check_epochs(results, a.dev, nm, {"A": seq}, [[("A", x, y, 0)] for x, y in chunks])
    for slot in (0, 5, 6, 11):
        labels = (f"L slot {slot} 1chunk", f"L slot {slot} 3chunks", f"L slot {slot} 512+rest", f"L slot {slot} decode", f"L slot {slot} mixed")
        if not os.path.isdir(LONG) or not any(sel(x) for x in labels):
            continue
        seq = Seq(LONG, slot)
        T = seq.T
        seqA = Seq(f"{ORACLE}/Q8_K_XL_A", slot)
        runs = {}
        for label, chunks, bps in ((labels[0], [(0, T)], 8), (labels[1], [(0, 444), (444, 888), (888, T)], 8),
                                   (labels[2], [(0, 512), (512, 1024), (1024, T)], 8),
                                   (labels[3], [(0, 1300)] + [(i, i + 1) for i in range(1300, T)], 2)):
            if sel(label):
                runs[label] = check_epochs(results, a.dev, label, {"L": seq}, [[("L", x, y, 0)] for x, y in chunks], bps)
        if sel(labels[4]):
            ep = [[("L", 0, 500, 1)], [("A", 0, 10, 0), ("L", 500, 501, 1)], [("L", 501, 502, 1), ("A", 10, 11, 0)],
                  [("A", 11, 16, 0), ("L", 502, 700, 1)], [("L", 700, 701, 1)]]
            check_epochs(results, a.dev, labels[4], {"L": seq, "A": seqA}, ep)
        base = runs.get(labels[0])
        for lb, got in runs.items():
            if base and got and lb != labels[0]:
                diffs.append((f"{lb} vs 1chunk", float(np.abs(got["L"] - base["L"]).max()), bool(np.array_equal(got["L"], base["L"]))))
    print(f"{'test':44s} {'dev':8s} {'NMSE':>10s} {'tol':>8s} result")
    bad = 0
    for name, dev, v, tol, ok in results:
        bad += not ok
        print(f"{name:44s} {dev:8s} {'n/a' if v is None else format(v, '10.3e'):>10s} {tol:8.0e} {'PASS' if ok else 'FAIL'}")
    for name, m, same in diffs:
        print(f"chunk invariance {name:34s} max abs diff {m:.3e} bitwise {same}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
