"""Run one Loom kernel launch under loom-test with numpy fixtures and compare the outputs (inkG)."""
import json
import os
import re
import subprocess
import tempfile

import numpy as np

DEVICES = {
    "gfx1201": "amdgpu://GPU-e0880c05003c324d",
    "gfx1151": "amdgpu://id/000000e2000000808e270f2b0bbb4e00",
}
KDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "kernel", "mix")
TYPES = {np.dtype("float32"): "f32", np.dtype("int32"): "i32", np.dtype("int64"): "i64", np.dtype("int16"): "i16"}


def drop_defined_decls(src):
    """Sources are concatenated into one file: a func.decl of a function that another source defines is removed."""
    defined = set(re.findall(r"^func\.def [^\n]*?(@\w+)\(", src, re.M))
    keep = [l for l in src.split("\n") if not (l.startswith("func.decl ") and re.match(r"func\.decl (@\w+)", l).group(1) in defined)]
    return "\n".join(keep)


def nmse(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    den = (b ** 2).sum()
    return float(((a - b) ** 2).sum() / den) if den > 0 else float(((a - b) ** 2).sum())


def run_steps(sources, buffers, steps, device="gfx1201", config=None, lock=True):
    """buffers: dict name -> numpy array (initial contents). steps: list of (kernel, workload, scalars, [buffer names]).
    All steps run in order in one loom-test case. Returns dict name -> array after the last step."""
    with tempfile.TemporaryDirectory(prefix="inkg_") as tmp:
        lines = []
        tt = {}
        for i, (name, a) in enumerate(buffers.items()):
            np.save(os.path.join(tmp, "in_%s.npy" % name), np.ascontiguousarray(a).reshape(-1))
            tt[name] = "tensor<%dx%s>" % (a.size, TYPES[a.dtype])
            lines.append('  %%%s = check.file.read.npy path("in_%s.npy") : %s' % (name, name, tt[name]))
        for k, (kernel, workload, scalars, names) in enumerate(steps):
            sc = []
            for j, v in enumerate(scalars):
                lines.append("  %%s%d_%d = check.literal value(%d) : index" % (k, j, v))
                sc.append("%%s%d_%d" % (k, j))
            wl = list(workload) if isinstance(workload, (tuple, list)) else [workload]
            ws = []
            for j, v in enumerate(wl):
                lines.append("  %%w%d_%d = check.literal value(%d) : index" % (k, j, v))
                ws.append("%%w%d_%d" % (k, j))
            args = ws + sc + ["%" + n for n in names]
            types = ["index"] * (len(wl) + len(scalars)) + [tt[n] for n in names]
            lines.append("  kernel.launch @%s[%s](%s) : [%s](%s)" % (kernel, ", ".join(ws), ", ".join(args), ", ".join(["index"] * len(wl)), ", ".join(types)))
        for name in buffers:
            lines.append('  check.file.write.npy value(%%%s) path("out_%s.npy") mode(always) : %s' % (name, name, tt[name]))
        src = ""
        for s in sources:
            src += open(os.path.join(KDIR, s)).read() + "\n"
        src = drop_defined_decls(src)
        src += "check.case public @t {\n" + "\n".join(lines) + "\n  check.return\n}\n"
        path = os.path.join(tmp, "t.loom")
        open(path, "w").write(src)
        cmd = ["loom-test", path, "--device=" + DEVICES[device], "--file-output-root=" + os.path.join(tmp, "out")]
        for k, v in (config or {}).items():
            cmd.append("--config=%s=%s" % (k, v))
        if lock:
            cmd = ["flock", "-s", os.path.expanduser("~/hrx-pr1/gpu.lock")] + cmd
        r = subprocess.run(cmd, capture_output=True, text=True)
        try:
            rep = json.loads(r.stdout)
        except Exception:
            raise RuntimeError("loom-test failed: %s\n%s" % (r.stdout[:2000], r.stderr[:2000]))
        if rep.get("planning_issue_count") or any(s.get("issues") for s in rep.get("samples", [])):
            raise RuntimeError(json.dumps(rep)[:3000])
        return {n: np.load(os.path.join(tmp, "out", "out_%s.npy" % n)).reshape(a.shape) for n, a in buffers.items()}


def run_launch(sources, kernel, workload, scalars, buffers, device="gfx1201", config=None, lock=True):
    """Single launch; buffers is a list in launch order. Returns the list of buffers after the launch."""
    names = ["b%d" % i for i in range(len(buffers))]
    res = run_steps(sources, dict(zip(names, buffers)), [(kernel, workload, scalars, names)], device, config, lock)
    return [res[n] for n in names]


def bench_steps(sources, buffers, steps, device="gfx1201", config=None, extra=None):
    """Time the launches of one case with loom-benchmark (dispatch_complete) under an exclusive GPU lock.
    Returns the parsed JSON snapshot."""
    with tempfile.TemporaryDirectory(prefix="inkgb_") as tmp:
        lines = []
        tt = {}
        for name, a in buffers.items():
            np.save(os.path.join(tmp, "in_%s.npy" % name), np.ascontiguousarray(a).reshape(-1))
            tt[name] = "tensor<%dx%s>" % (a.size, TYPES[a.dtype])
            lines.append('  %%%s = check.file.read.npy path("in_%s.npy") : %s' % (name, name, tt[name]))
        for k, (kernel, workload, scalars, names) in enumerate(steps):
            sc = []
            for j, v in enumerate(scalars):
                lines.append("  %%s%d_%d = check.literal value(%d) : index" % (k, j, v))
                sc.append("%%s%d_%d" % (k, j))
            wl = list(workload) if isinstance(workload, (tuple, list)) else [workload]
            ws = []
            for j, v in enumerate(wl):
                lines.append("  %%w%d_%d = check.literal value(%d) : index" % (k, j, v))
                ws.append("%%w%d_%d" % (k, j))
            args = ws + sc + ["%" + n for n in names]
            types = ["index"] * (len(wl) + len(scalars)) + [tt[n] for n in names]
            lines.append("  kernel.launch @%s[%s](%s) : [%s](%s)" % (kernel, ", ".join(ws), ", ".join(args), ", ".join(["index"] * len(wl)), ", ".join(types)))
        src = ""
        for s in sources:
            src += open(os.path.join(KDIR, s)).read() + "\n"
        src = drop_defined_decls(src)
        src += "check.case public @t {\n" + "\n".join(lines) + "\n  check.return\n}\ncheck.benchmark<@t> @b\n"
        path = os.path.join(tmp, "t.loom")
        open(path, "w").write(src)
        out = os.path.join(tmp, "res.json")
        cmd = ["loom-benchmark", path, "--device=" + DEVICES[device], "--benchmark=@b", "--measure=dispatch_complete", "--output=" + out,
               "--repetitions=3", "--min-time-ms=200"] + (extra or [])
        for k, v in (config or {}).items():
            cmd.append("--config=%s=%s" % (k, v))
        cmd = ["flock", "-x", os.path.expanduser("~/hrx-pr1/gpu.lock")] + cmd
        r = subprocess.run(cmd, capture_output=True, text=True)
        if not os.path.exists(out):
            raise RuntimeError("loom-benchmark failed: %s\n%s" % (r.stdout[:2000], r.stderr[:2000]))
        return json.load(open(out))
