#!/usr/bin/env python3
"""Runs the kernel and numerics tests of package/test/ on one GPU and writes the skills' kernel_tests evidence.

usage (GPU box): ktest_adapter.py <r9700|strix> [--suites experts,dense,mix,attn,chat] [--src DIR] [--out DIR]
  --src  directory with package/ and scripts/ (default $KT_BASE/src; KT_BASE defaults to ~/inkling/inkR); the tests run against that package
  --out  result directory (default $KT_BASE/kt/results_<gpu>)
Each test command is one target. A target passes when its exit status is 0. Output: kernel_tests.log (ends with "N/M passed", "FAIL name" lines
before it, which is what make_info.py gates reads), kernel_tests.json (per target status, seconds, last line, log file).
The tests keep their work files in <kt>/ (a copy of package/test with the box paths of the earlier agents rewritten to <kt>/).
Targets that need the capture route (scripts/capture_all.py, scripts/head_check.py) are listed as skipped: the published route has no capture service.
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time

HOME = os.path.expanduser("~")
ARCH = {"r9700": "gfx1201", "strix": "gfx1151"}
VENV = os.environ.get("KT_VENV", f"{HOME}/loom-serve/.venv-inkD")
LOCK = f"{HOME}/hrx-pr1/gpu.lock"
BASE = os.environ.get("KT_BASE", f"{HOME}/inkling/inkR")

DENSE = ["test_norm_embed.py", "test_gemv.py - xs", "test_gemv.py - split", "test_gemv.py - rows", "test_gemm.py - gemm_q8_0", "test_gemm.py - gemm_wide_q8_0",
         "test_gemm.py - gemm_wmma_bf16", "test_gemm.py - gemm_bf16", "test_gemm.py - gemm_f32", "test_head.py", "test_oracle_layers.py"]


def suites(arch):
    s = {"experts": [("test_gate_up", ["test_gate_up.py", arch], {}),
                     ("test_oracle", ["test_oracle.py", arch], {}),
                     ("test_oracle_rows5_portable", ["test_oracle.py", arch, "0", "9", "19", "27", "39", "20", "21", "22", "23"], {"ROWS": "5", "PATHS": "portable"}),
                     ("test_oracle_rows3_9_portable", ["test_oracle.py", arch, "0", "19"], {"ROWS": "3,9", "PATHS": "portable"}),
                     ("test_prefill", ["test_prefill.py", arch, "37", "308", "512", "1024", "2048"], {})],
         "dense": [], "attn": [("attn_test", ["attn_test.py", "run", "--dev", arch], {})]}
    for t in DENSE:
        f, *rest = t.split()
        a = [f, arch, *rest]
        s["dense"].append((t.replace(" - ", ":").replace(" ", ":").replace(".py", ""), a, {}))
    s["mix"] = [("test_mix", ["test_mix.py", arch], {}), ("test_conv", ["test_conv.py", arch], {}),
                ("test_ctrl_c128_all_t1", ["test_ctrl.py", arch, "128", "all", "1"], {}),
                ("test_ctrl_c256_all_t16", ["test_ctrl.py", arch, "256", "all", "16"], {})]
    for c in (64, 128, 256):
        s["mix"].append((f"test_ctrl_q4_c{c}", ["test_ctrl.py", arch, str(c), "q4", "8"], {}))
    return s


def prepare(src, kt):
    os.makedirs(kt, exist_ok=True)
    dst = os.path.join(kt, "test")
    if os.path.exists(dst):
        shutil.rmtree(dst)
    shutil.copytree(os.path.join(src, "package", "test"), dst, ignore=shutil.ignore_patterns("__pycache__", "vendor"))
    for root, _, files in os.walk(dst):
        for f in files:
            if f.endswith((".py", ".sh")):
                p = os.path.join(root, f)
                t = open(p, encoding="utf-8").read()
                n = re.sub(r"inkling/(?:ink[DEFG]|opt[A-Z]+[0-9]*)\b", "inkling/" + os.path.basename(BASE) + "/kt", t)
                n = re.sub(r"/tmp/ink([DEFG])_", lambda m: f"{kt}/work/ink{m.group(1)}_", n)
                if n != t:
                    open(p, "w", encoding="utf-8").write(n)
    pkg = os.path.join(src, "package")
    for d in ["package"] + [x for x in os.listdir(pkg) if x != "test"]:
        link = os.path.join(kt, d)
        if os.path.islink(link):
            os.remove(link)
        os.symlink(pkg if d == "package" else os.path.join(pkg, d), link)
    for d in ("work", "build", "logs", "results"):
        os.makedirs(os.path.join(kt, d), exist_ok=True)


def run_target(name, cwd, argv, env, log, timeout):
    t0 = time.time()
    cmd = [f"{VENV}/bin/python", *argv] if os.environ.get("KT_LOCKED") else ["/usr/bin/flock", "-s", LOCK, f"{VENV}/bin/python", *argv]  # KT_LOCKED: the caller holds one outer flock; the shim flock on PATH makes the inner locks of the tests no-ops
    with open(log, "w") as f:
        try:
            rc = subprocess.run(cmd, cwd=cwd, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except subprocess.TimeoutExpired:
            rc = 124
    last = ""
    with open(log, errors="replace") as f:
        lines = [x.strip() for x in f.read().splitlines() if x.strip()]
    if lines:
        last = lines[-1][:200]
    return rc, round(time.time() - t0, 1), last


def run_chat(src, arch, out):
    env = dict(os.environ, INKLING_CHAT_ROOT=src, INKLING_CHAT_PY=f"{VENV}/bin/python")
    if os.environ.get("KT_LOCKED"):
        env["PATH"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ktest_shim") + ":" + env["PATH"]
    log = os.path.join(out, "chat_run_all.log")
    t0 = time.time()
    with open(log, "w") as f:
        subprocess.run(["bash", os.path.join(src, "scripts", "chat_run_all.sh"), arch], cwd=src, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=3600)
    rows = []
    for variant in ("plain", "echo", "script", "policy"):
        p = os.path.join(src, f"result_{arch}_{variant}.txt")
        txt = open(p, errors="replace").read() if os.path.exists(p) else ""
        npass, nfail = len(re.findall(r"^PASS", txt, re.M)), len(re.findall(r"^FAIL", txt, re.M))
        ok = txt != "" and nfail == 0 and npass > 0
        shutil.copy(p, os.path.join(out, f"chat_{variant}.txt")) if os.path.exists(p) else None
        rows.append({"name": f"chat:{variant}", "status": "pass" if ok else "fail", "seconds": round((time.time() - t0) / 4, 1),
                     "last": f"{npass} PASS lines, {nfail} FAIL lines", "log": f"chat_{variant}.txt", "cases": npass})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gpu", choices=sorted(ARCH))
    ap.add_argument("--suites", default="experts,dense,mix,attn,chat")
    ap.add_argument("--src", default=f"{BASE}/src")
    ap.add_argument("--out", default=None)
    ap.add_argument("--timeout", type=int, default=3600)
    a = ap.parse_args()
    arch = ARCH[a.gpu]
    kt = f"{BASE}/kt"
    out = a.out or os.path.join(kt, f"results_{a.gpu}")
    os.makedirs(out, exist_ok=True)
    prepare(a.src, kt)
    env = dict(os.environ, PATH=f"{os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ktest_shim')}:{VENV}/bin:{os.environ['PATH']}", VIRTUAL_ENV=VENV, PYTHONDONTWRITEBYTECODE="1")
    rows = []
    table = suites(arch)
    for suite, name, argv, extra in [("dense", "test_gemv_group", ["test_gemv_group.py", arch], {}), ("experts", "test_router_decode", ["test_router_decode.py", arch], {}), ("mix", "test_ctrl_msg", ["test_ctrl_msg.py", arch], {})]:
        if os.path.exists(os.path.join(kt, "test", suite, argv[0])):
            table[suite].append((name, argv, extra))
    if os.path.exists(os.path.join(kt, "test", "mix", "test_ctrl_msg.py")):
        for c, t in (("128", "1"), ("256", "16")):
            table["mix"].append((f"test_ctrl_fold_c{c}_all_t{t}", ["test_ctrl.py", arch, c, "all", t], {"CTRL_FOLD": "1"}))
    for suite in a.suites.split(","):
        if suite == "chat":
            rows += run_chat(a.src, arch, out)
            continue
        for name, argv, extra in table[suite]:
            log = f"{suite}_{name.replace(':', '_')}.log"
            rc, sec, last = run_target(f"{suite}/{name}", os.path.join(kt, "test", suite), argv, {**env, **extra}, os.path.join(out, log), a.timeout)
            rows.append({"name": f"{suite}/{name}", "status": "pass" if rc == 0 else "fail", "seconds": sec, "last": last, "log": log, "rc": rc})
            print(rows[-1]["status"].upper(), rows[-1]["name"], f"{sec}s", last, flush=True)
            for w in os.listdir(os.path.join(kt, "work")):
                shutil.rmtree(os.path.join(kt, "work", w), ignore_errors=True)
    for name in ("scripts/capture_all.py", "scripts/head_check.py"):
        rows.append({"name": name, "status": "skipped", "last": "needs a capture-route server (old package/ card, INKLING_CAPTURE) and the oracle dumps; not part of the published route"})
    ran = [r for r in rows if r["status"] != "skipped"]
    ok = sum(r["status"] == "pass" for r in ran)
    with open(os.path.join(out, "kernel_tests.log"), "w") as f:
        for r in rows:
            if r["status"] == "skipped":
                f.write(f"SKIP {r['name']}\n")
            elif r["status"] == "fail":
                f.write(f"FAIL {r['name']}\n")
        f.write(f"{ok}/{len(ran)} passed\n")
    json.dump({"gpu": a.gpu, "arch": arch, "passed": ok, "total": len(ran), "skipped": len(rows) - len(ran), "targets": rows},
              open(os.path.join(out, "kernel_tests.json"), "w"), indent=1)
    print(f"{ok}/{len(ran)} passed ({len(rows) - len(ran)} skipped)")
    sys.exit(0 if ok == len(ran) else 1)


if __name__ == "__main__":
    main()
