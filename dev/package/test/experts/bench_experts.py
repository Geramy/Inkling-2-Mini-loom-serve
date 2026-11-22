#!/usr/bin/env python3
"""Per-kernel timing of the expert kernels on real layer-0 weights. Usage: bench_experts.py gfx1201|gfx1151 [n ...]
Fixtures: x from the oracle (prompt A, tiled with noise for n > 16), full 128-expert NVFP4 weights, ids from the router kernel."""
import sys, os, json, subprocess
import numpy as np
from runner import Case, URI, LOCK, lib_files, cached_npy
import refs, gguf_io as G

arch = sys.argv[1]
WMMA = os.environ.get("WMMA") == "1"
SUF = "_wmma" if WMMA else ""
TILE = 64 if WMMA else 8
ns = [int(a) for a in sys.argv[2:]] or [1, 8, 16, 128, 512]
cache = os.path.expanduser("~/inkling/optP1/work/cache")
os.makedirs(cache, exist_ok=True)
L = int(os.environ.get("LAYER", "0"))
KIND = os.environ.get("KIND", "q4")
dump = G.Dump("Q8_K_XL_A")
wt, gu, dn, sc = G.expert_weights(KIND, L)
if sc is None:
    sc = (np.zeros(128, np.float32), np.zeros(128, np.float32))
shapes = {
    "wgu": cached_npy(f"{cache}/wgu_{KIND}_L{L}.npy", lambda: np.ascontiguousarray(gu).reshape(-1)),
    "wdn": cached_npy(f"{cache}/wdn_{KIND}_L{L}.npy", lambda: np.ascontiguousarray(dn).reshape(-1)),
}
rw, bias = G.router_weights(KIND, L)
xn = dump.get(f"norm-{L}", dump.last_occ(f"norm-{L}")).reshape(16, 2304) * np.asarray(G.reader(KIND)[f"blk.{L}.ffn_norm.weight"].data).astype(np.float32)
GB = {"gfx1201": 620e9, "gfx1151": 230e9}[arch]


def make_x(n):
    rng = np.random.default_rng(n)
    base = np.tile(xn, (n // 16 + 1, 1))[:n]
    return (base * (1 + 0.05 * rng.standard_normal(base.shape))).astype(np.float32)


def prelude(n):
    x = make_x(n)
    c = Case("pre", f"{cache}/pre_{arch}_{KIND}{L}_{n}")
    c.inp("x", x)
    c.inp("rw", rw)
    c.inp("bias", bias)
    c.inp_file("wgu", f"{cache}/wgu_{KIND}_L{L}.npy", shapes["wgu"], np.int8)
    c.inp_file("wdn", f"{cache}/wdn_{KIND}_L{L}.npy", shapes["wdn"], np.int8)
    c.inp("sgu", sc[0])
    c.inp("sdn", sc[1])
    recs = min(n * 8, n + 128)
    c.zeros("logits", (n, 128), np.float32)
    c.zeros("x16", (n, 2304), np.float16)
    c.zeros("ids", (n, 8), np.int32)
    c.zeros("wts", (n, 8), np.float32)
    c.zeros("tasks", (4 + recs * 66,), np.int32)
    c.zeros("act", (n * 8, 1152), np.float16)
    c.zeros("dout", (n * 8, 2304), np.float32)
    c.zeros("y", (n, 2304), np.float32)
    c.lit("n", n)
    c.lit("wtype", wt)
    c.lit("tile", TILE)
    c.launch("inkling_router_logits", ["n"], [("n", "index"), ("x", "x"), ("rw", "rw"), ("logits", "logits"), ("x16", "x16")], ("%n: index", "%n: index, %x: buffer, %rw: buffer, %logits: buffer, %x16: buffer"))
    c.launch("inkling_router_select", ["n"], [("n", "index"), ("logits", "logits"), ("bias", "bias"), ("ids", "ids"), ("wts", "wts")], ("%n: index", "%n: index, %logits: buffer, %bias: buffer, %ids: buffer, %wts: buffer"))
    c.launch("inkling_experts_routes", [], [("n", "index"), ("tile", "index"), ("ids", "ids"), ("tasks", "tasks")], ("", "%n: index, %tile: index, %ids: buffer, %tasks: buffer"))
    c.launch("inkling_experts_gate_up" + SUF, ["n"], [("n", "index"), ("wtype", "index"), ("x16", "x16"), ("tasks", "tasks"), ("wgu", "wgu"), ("sgu", "sgu"), ("act", "act")], ("%n: index", "%n: index, %wtype: index, %x: buffer, %tasks: buffer, %w: buffer, %scales: buffer, %act: buffer"))
    c.launch("inkling_experts_down" + SUF, ["n"], [("n", "index"), ("wtype", "index"), ("act", "act"), ("tasks", "tasks"), ("wdn", "wdn"), ("sdn", "sdn"), ("dout", "dout")], ("%n: index", "%n: index, %wtype: index, %act: buffer, %tasks: buffer, %w: buffer, %scales: buffer, %dout: buffer"))
    c.launch("inkling_experts_combine", ["n"], [("n", "index"), ("dout", "dout"), ("wts", "wts"), ("y", "y")], ("%n: index", "%n: index, %dout: buffer, %wts: buffer, %y: buffer"))
    for o in ("logits", "x16", "ids", "wts", "tasks", "act", "dout", "y"):
        c.out(o)
    out = c.run(arch, ["router.loom", "routes.loom", "gate_up.loom", "down.loom", "combine.loom"] if not WMMA else ["router.loom", "routes.loom", "gate_up_wmma.loom", "down_wmma.loom", "combine.loom"])
    return x, out


def bench_module(n, out, d):
    """One case per kernel with the intermediate fixtures; returns the module path."""
    for k, v in out.items():
        np.save(os.path.join(d, k + ".npy"), v)
    x = make_x(n)
    np.save(os.path.join(d, "x.npy"), x)
    np.save(os.path.join(d, "rw.npy"), rw)
    np.save(os.path.join(d, "bias.npy"), bias)
    np.save(os.path.join(d, "sgu.npy"), sc[0])
    np.save(os.path.join(d, "sdn.npy"), sc[1])
    specs = {
        "router_logits": (["n"], ["n", "x", "rw", "logits", "x16"], "inkling_router_logits", ["x", "rw", "logits", "x16"]),
        "router_select": (["n"], ["n", "logits", "bias", "ids", "wts"], "inkling_router_select", ["logits", "bias", "ids", "wts"]),
        "routes": ([], ["n", "tile", "ids", "tasks"], "inkling_experts_routes", ["ids", "tasks"]),
        "gate_up": (["n"], ["n", "wtype", "x16", "tasks", "wgu", "sgu", "act"], "inkling_experts_gate_up" + SUF, ["x16", "tasks", "wgu", "sgu", "act"]),
        "down": (["n"], ["n", "wtype", "act", "tasks", "wdn", "sdn", "dout"], "inkling_experts_down" + SUF, ["act", "tasks", "wdn", "sdn", "dout"]),
        "combine": (["n"], ["n", "dout", "wts", "y"], "inkling_experts_combine", ["dout", "wts", "y"]),
    }
    return specs


def run_bench(n, tag):
    x, out = prelude(n)
    d = f"{cache}/bench_{arch}_{KIND}{L}_{n}"
    os.makedirs(d, exist_ok=True)
    specs = bench_module(n, out, d)
    results = {}
    decl_sigs = {
        "inkling_router_logits": ("%n: index", "%n: index, %x: buffer, %rw: buffer, %logits: buffer, %x16: buffer"),
        "inkling_router_select": ("%n: index", "%n: index, %logits: buffer, %bias: buffer, %ids: buffer, %wts: buffer"),
        "inkling_experts_routes": ("", "%n: index, %tile: index, %ids: buffer, %tasks: buffer"),
        "inkling_experts_gate_up" + SUF: ("%n: index", "%n: index, %wtype: index, %x: buffer, %tasks: buffer, %w: buffer, %scales: buffer, %act: buffer"),
        "inkling_experts_down" + SUF: ("%n: index", "%n: index, %wtype: index, %act: buffer, %tasks: buffer, %w: buffer, %scales: buffer, %dout: buffer"),
        "inkling_experts_combine": ("%n: index", "%n: index, %dout: buffer, %wts: buffer, %y: buffer"),
    }
    src = []
    kfiles = ["router.loom", "routes.loom", "gate_up.loom", "down.loom", "combine.loom"] if not WMMA else ["router.loom", "routes.loom", "gate_up_wmma.loom", "down_wmma.loom", "combine.loom"]
    decls = []
    for name, (wl, args, kern, ins) in specs.items():
        c = Case("b_" + name, d)
        for a in set(args) | set(ins):
            if a in ("n", "wtype", "tile"):
                continue
            if a in ("wgu", "wdn"):
                c.inp_file(a, f"{cache}/{a}_{KIND}_L{L}.npy", shapes[a], np.int8)
            else:
                arr = np.load(os.path.join(d, a + ".npy"))
                c.types[a] = __import__("runner").ttype(arr)
                c.lines.append(f'  %{a} = check.file.read.npy path("{d}/{a}.npy") : {c.types[a]}')
        c.lit("n", n)
        c.lit("wtype", wt)
        c.lit("tile", TILE)
        c.launch(kern, wl, [(a, "index" if a in ("n", "wtype", "tile") else a) for a in args], decl_sigs[kern])
        decls.extend(c.decls)
        src.append(c.source().split("\n\n", 1)[1] + f"check.benchmark<@b_{name}> @bench_{name}\n")
    text = "\n".join(sorted(set(decls))) + "\n\n" + "\n".join(src)
    open(f"{d}/bench.loom", "w").write(text)
    subprocess.run(["loom-link", *lib_files(*kfiles), f"{d}/bench.loom", "--output=" + f"{d}/bench_merged.loom"], check=True)
    for name in specs:
        res = f"{d}/res_{name}.json"
        cmd = ["flock", "-x", LOCK, "loom-benchmark", f"{d}/bench_merged.loom", f"--benchmark=@bench_{name}", "--device=" + URI[arch], "--output=" + res,
               "--measure=dispatch_complete", "--batch-size=16", "--iterations=20", "--warmup-iterations=3", "--input-ring-count=4", "--profile-final-batch=true"]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(name, "benchmark failed", r.stderr[-1500:], r.stdout[-500:])
            continue
        results[name] = json.load(open(res))
    return results


def find_time(r):
    w = r["work_items"][0]
    op = w["measurement"]["operation_timing_ns"]["p50"]
    dev = w.get("profile_replay", {}).get("summary", {}).get("dispatch_distribution", {}).get("duration_ns", {}).get("p50")
    return op, dev


if __name__ == "__main__":
    for n in ns:
        res = run_bench(n, "")
        for name, r in res.items():
            op, dev = find_time(r)
            print(f"{arch} L{L}/{KIND} wtype={wt} {'wmma' if WMMA else 'portable'} n={n} {name}: host_p50={op / 1e3:.1f} us device_p50={(dev or 0) / 1e3:.1f} us")
