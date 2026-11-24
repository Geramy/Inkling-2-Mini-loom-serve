"""usage: chat_session.py --mat DIR [--preset P] [--set key=value ...] [--arch gfx1201] [--port N] [--log F] -- COMMAND...
Starts loom-serve on a materialized model directory (the preset execution values can be overridden with --set execution key=value),
waits for registration, runs COMMAND with the URL as the last argument, then stops the server by its own PID.
Call it under: flock -x (timed) or flock -s (correctness) ~/hrx-pr1/gpu.lock. Prints the peak VRAM of the R9700 (sysfs)."""
import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--mat", required=True)
ap.add_argument("--preset", default="chat-r9700")
ap.add_argument("--set", action="append", default=[])
ap.add_argument("--patch", action="append", default=[], help="REL=FILE: replace REL (e.g. program/control/chat.loom) in the run copy of the package by FILE")
ap.add_argument("--mem", action="append", default=[], help="memory key=value override of the preset (for example transient_bytes=10800000000)")
ap.add_argument("--arch", default="gfx1201")
ap.add_argument("--port", type=int, default=18500)
ap.add_argument("--log", default="serve.log")
ap.add_argument("--work", default=os.path.expanduser("~/inkling/optCHAT/run"))
ap.add_argument("--venv", default=os.path.expanduser("~/inkling/tools/M30-venv"))
ap.add_argument("cmd", nargs=argparse.REMAINDER)
a = ap.parse_args()
cmd = a.cmd[1:] if a.cmd and a.cmd[0] == "--" else a.cmd
URI = {"gfx1201": "amdgpu://GPU-e0880c05003c324d", "gfx1151": "amdgpu://id/000000e2000000808e270f2b0bbb4e00"}[a.arch]
VRAM = "/sys/class/drm/card0/device/mem_info_vram_used"

os.makedirs(a.work, exist_ok=True)
run = os.path.join(a.work, f"v{os.getpid()}")
os.makedirs(run)
for name in os.listdir(a.mat):
    if name not in ("model.json", ".loom-fetch.lock"):
        os.symlink(os.path.join(os.path.abspath(a.mat), name), os.path.join(run, name))
for spec in a.patch:
    rel, src = spec.split("=", 1)
    link = os.path.join(run, rel.split("/")[0])
    if os.path.islink(link):
        real = os.path.realpath(link)
        os.unlink(link)
        shutil.copytree(real, link, copy_function=lambda s, d, **k: os.symlink(os.path.realpath(s), d))
    target = os.path.join(run, rel)
    os.unlink(target)
    shutil.copyfile(os.path.abspath(os.path.expanduser(src)), target)
doc = json.load(open(os.path.join(a.mat, "model.json")))
ex = doc["presets"][a.preset]["execution"]
for kv in a.set:
    k, v = kv.split("=", 1)
    ex[k] = json.loads(v)
for kv in a.mem:
    k, v = kv.split("=", 1)
    doc["presets"][a.preset]["memory"][k] = json.loads(v)
json.dump(doc, open(os.path.join(run, "model.json"), "w"), indent=1)
print("execution", json.dumps(ex), flush=True)

url = f"http://127.0.0.1:{a.port}"
log = open(os.path.join(a.work, a.log), "w")
env = dict(os.environ, PATH=os.path.join(a.venv, "bin") + ":" + os.environ["PATH"])
proc = subprocess.Popen([os.path.join(a.venv, "bin", "loom-serve"), f"--device={URI}", f"--model={run}", f"--preset={a.preset}", "--pool_backing=fixed",
                         f"--listen=127.0.0.1:{a.port}", "--workers=4"], stdout=log, stderr=subprocess.STDOUT, env=env, stdin=subprocess.DEVNULL)
print("server pid", proc.pid, flush=True)
peak = [0, 0]
stop = threading.Event()


def sample():
    while not stop.is_set():
        try:
            v = int(open(VRAM).read())
            peak[0] = max(peak[0], v)
            peak[1] = v
        except Exception:
            pass
        time.sleep(0.5)


threading.Thread(target=sample, daemon=True).start()
rc = 1
try:
    t0 = time.time()
    while True:
        if proc.poll() is not None:
            print("server died", proc.returncode); sys.exit(2)
        try:
            st = json.load(urllib.request.urlopen(url + "/v1/status", timeout=2))
            if '"registration": "ready"' in json.dumps(st) or '"registration":"ready"' in json.dumps(st):
                break
            if "failed" in json.dumps(st):
                print("registration failed", json.dumps(st)[:600]); sys.exit(2)
        except Exception:
            pass
        if time.time() - t0 > 900:
            print("registration timeout"); sys.exit(2)
        time.sleep(1)
    print(f"ready after {time.time() - t0:.0f} s, vram used {peak[1] / 1e9:.2f} GB", flush=True)
    cmd = [c.replace("@URL@", url) for c in cmd] if any("@URL@" in c for c in cmd) else cmd + [url]
    rc = subprocess.call(cmd, env=dict(os.environ, SERVE_PID=str(proc.pid), VRAM_FILE=VRAM))
finally:
    stop.set()
    print(f"peak vram {peak[0] / 1e9:.2f} GB, last {peak[1] / 1e9:.2f} GB", flush=True)
    if proc.poll() is None:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(60)
        except subprocess.TimeoutExpired:
            proc.kill()
    shutil.rmtree(run, ignore_errors=True)
sys.exit(rc)
