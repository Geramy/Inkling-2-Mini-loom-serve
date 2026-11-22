#!/usr/bin/env python3
"""Largest served context per GPU: for each (rows, context_capacity) start loom-serve on a modified copy of the materialized preset and run ctx_admit.py --complete.

usage (GPU box, under flock -s): ctx_table.py <r9700|strix> ROWSxCTX [ROWSxCTX ...] [--memory-bytes N] [--tokens-margin 100]
The directory <W>/mat-<gpu> is the loom-fetch copy of the preset. Each case gets <W>/ctx-<gpu>/r<rows>_c<ctx>: symlinks to every file of the copy except model.json,
which is a copy with rows and context_capacity changed (prefill_capacity stays as in the preset). Results go to <W>/ctx-<gpu>/results.jsonl, one JSON line per case.
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time

HOME = os.path.expanduser("~")
W = os.environ.get("CTX_W", f"{HOME}/inkling/inkR/W")
GPU = {"r9700": ("chat-r9700", "amdgpu://GPU-e0880c05003c324d", 18192), "strix": ("chat", "amdgpu://id/000000e2000000808e270f2b0bbb4e00", 18193)}
VENV = f"{HOME}/loom/.venv"
SK = f"{W}/skills/loom-benchmark/scripts"
T = f"{HOME}/loom/.tools/bin/python"


def vram_used():
    out = {}
    for card in sorted(glob.glob("/sys/class/drm/card[0-9]/device")):
        try:
            out[os.path.basename(os.path.dirname(card))] = {k: int(open(f"{card}/mem_info_{k}_used").read()) for k in ("vram", "gtt")}
        except OSError:
            pass
    return out or None


def case(gpu, rows, ctx, extra, margin, complete_s):
    preset, uri, port = GPU[gpu]
    mat = f"{W}/mat-{gpu}"
    d = f"{W}/ctx-{gpu}/r{rows}_c{ctx}"
    os.makedirs(d, exist_ok=True)
    for f in os.listdir(mat):
        if f != "model.json":
            link = os.path.join(d, f)
            if not os.path.lexists(link):
                os.symlink(os.path.join(mat, f), link)
    m = json.load(open(f"{mat}/model.json"))
    ex = m["presets"][preset]["execution"]
    ex["rows"], ex["context_capacity"] = rows, ctx
    json.dump(m, open(f"{d}/model.json", "w"), indent=2)
    pkg = f"{d}/pkg"
    os.makedirs(pkg, exist_ok=True)
    for n, tgt in (("model.json", f"{d}/model.json"), ("program", f"{mat}/program"), ("tokenizer.json", f"{mat}/tokenizer.json"), ("llm_config.json", f"{W}/pkg-{gpu}/llm_config.json")):
        if not os.path.lexists(f"{pkg}/{n}"):
            os.symlink(tgt, f"{pkg}/{n}")
    res = {"gpu": gpu, "rows": rows, "context": ctx, "extra": extra}
    log = f"{d}/serve.log"
    cmd = [f"{VENV}/bin/loom-serve", f"--device={uri}", f"--model={d}", f"--preset={preset}", "--pool_backing=fixed", f"--listen=127.0.0.1:{port}", "--workers=4", *extra]
    t0 = time.time()
    srv = subprocess.Popen(cmd, stdout=open(log, "w"), stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
    url = f"http://127.0.0.1:{port}"
    try:
        state = None
        for _ in range(300):
            if srv.poll() is not None:
                break
            r = subprocess.run(["curl", "-s", "-m", "2", f"{url}/v1/status"], capture_output=True, text=True).stdout
            if '"registration":"ready"' in r.replace(" ", ""):
                state = "ready"
                break
            if '"registration":"failed' in r.replace(" ", "") or "registration_error" in r:
                state = "failed"
                break
            time.sleep(1)
        res["register_s"] = round(time.time() - t0, 1)
        if state != "ready":
            res["registered"] = False
            res["detail"] = open(log, errors="replace").read()[-600:]
            return res
        res["registered"] = True
        env = dict(os.environ, K=SK, LOOM_KIT=f"{HOME}/inkling/inkR/kit", LOOM_PRESET=preset, PYTHONDONTWRITEBYTECODE="1")
        w = subprocess.run([T, f"{SK}/wait_model.py", url, "--wait", "900"], capture_output=True, text=True, env=env)
        res["first_request"] = (w.stdout + w.stderr).strip()[-200:]
        if w.returncode != 0:
            res["detail"] = open(log, errors="replace").read()[-600:]
            return res
        res["vram_loaded"] = vram_used()
        out = f"{d}/ctx_admit.json"
        cflag = ["--complete", str(complete_s)] if complete_s else []
        a = subprocess.run([T, f"{SK}/ctx_admit.py", pkg, url, "--tokens", str(ctx - margin), "--wait", str(max(120, complete_s)), *cflag, "--out", out],
                           capture_output=True, text=True, env=env, timeout=complete_s + 600)
        res["admit"] = (a.stdout + a.stderr).strip().splitlines()[-1][:300] if (a.stdout + a.stderr).strip() else ""
        res["admit_rc"] = a.returncode
        res["vram_after"] = vram_used()
        return res
    finally:
        srv.terminate()
        try:
            srv.wait(60)
        except subprocess.TimeoutExpired:
            srv.kill()
        time.sleep(5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("gpu", choices=sorted(GPU))
    ap.add_argument("cases", nargs="+", help="ROWSxCONTEXT")
    ap.add_argument("--extra", action="append", default=[], help="extra loom-serve flag (repeat), e.g. --extra=--memory_bytes=30gib")
    ap.add_argument("--tokens-margin", type=int, default=100)
    ap.add_argument("--complete", type=int, default=1800, help="seconds to wait for the full prefill; 0: admission only (cancel)")
    a = ap.parse_args()
    os.makedirs(f"{W}/ctx-{a.gpu}", exist_ok=True)
    for c in a.cases:
        rows, ctx = (int(x) for x in c.lower().split("x"))
        r = case(a.gpu, rows, ctx, a.extra, a.tokens_margin, a.complete)
        print(json.dumps(r), flush=True)
        with open(f"{W}/ctx-{a.gpu}/results.jsonl", "a") as f:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
