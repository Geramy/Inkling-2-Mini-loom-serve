"""Test harness for the dense kernels. Runs on the GPU box inside ~/loom-serve/.venv-inkF.

A case is a Loom check.case that reads .npy fixtures, launches kernels and writes the outputs as .npy.
Python compares the outputs with a numpy reference and prints one PASS/FAIL row per case.
"""
import json
import os
import subprocess
import sys
import tempfile

import numpy as np

HOME = os.path.expanduser("~")
PKG = os.path.join(HOME, "inkling/inkF/package")
ORACLE = os.path.join(HOME, "inkling/inkB/oracle/dumps")
GGUF = {
    "q4": os.path.join(HOME, "models/inkling2-mini/Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf"),
    "q8": os.path.join(HOME, "models/inkling2-mini/Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf"),
}
DEVICES = {
    "gfx1151": "amdgpu://id/000000e2000000808e270f2b0bbb4e00",
    "gfx1201": "amdgpu://GPU-e0880c05003c324d",
}
LOCK = os.path.join(HOME, "hrx-pr1/gpu.lock")
LIBS = ["dense_common", "dense_float", "dense_q8_0", "dense_mm_core", "dense_mm_vec", "dense_mm_wmma", "dense_wmma"]

NPY_TYPES = {np.dtype("float32"): "f32", np.dtype("int32"): "i32", np.dtype("int8"): "i8", np.dtype("uint8"): "i8", np.dtype("int16"): "i16"}


def tensor_type(a):
    return "tensor<" + "x".join(str(d) for d in a.shape) + "x" + NPY_TYPES[a.dtype] + ">"


class Case:
    """One Loom test case. `body` is Loom text that uses %<name> for every fixture and output."""

    def __init__(self, name, files, inputs, outputs, body, preamble="", config=None):
        self.name, self.files, self.inputs, self.outputs, self.body = name, files, inputs, outputs, body
        self.preamble, self.config = preamble, config or {}

    def source(self, benchmark=False):
        lines = [self.preamble, "", "check.case public @%s {" % self.name]
        for k, a in self.inputs.items():
            lines.append('  %%%s = check.file.read.npy path("%s.npy") : %s' % (k, k, tensor_type(a)))
        for k, a in self.outputs.items():
            lines.append("  %%%s = check.generate.fill value(0) : %s" % (k, tensor_type(a)) if a.dtype != np.float32 else
                         "  %%%s = check.generate.fill value(0.0) : %s" % (k, tensor_type(a)))
        lines.append(self.body)
        for k, a in self.outputs.items():
            lines.append('  check.file.write.npy value(%%%s) path("%s.npy") mode(always) : %s' % (k, k, tensor_type(a)))
        lines.append("  check.return")
        lines.append("}")
        if benchmark:
            lines.append("check.benchmark<@%s> @bench_%s" % (self.name, self.name))
        return "\n".join(lines)


def run_case(case, arch, workdir, lock="-s", timeout=600):
    os.makedirs(workdir, exist_ok=True)
    for k, a in case.inputs.items():
        save_fixture(workdir, k, a)
    test_path = os.path.join(workdir, case.name + "_test.loom")
    with open(test_path, "w") as f:
        f.write(case.source())
    merged = os.path.join(workdir, case.name + ".loom")
    files = [os.path.join(PKG, p) for p in case.files] + [test_path]
    r = subprocess.run(["loom-link"] + files + ["--output=" + merged], capture_output=True, text=True)
    if r.returncode != 0:
        return None, "link: " + r.stderr[-3000:]
    cmd = ["flock", lock, LOCK, "loom-test", merged, "--device=" + DEVICES[arch], "--file-output-root=" + os.path.join(workdir, "out")]
    for k, v in {**tune_config(arch), **case.config}.items():
        cmd.append("--config=%s=%s" % (k, v))
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        return None, "run: " + (r.stdout[-1500:] + r.stderr[-3000:])
    out = {}
    for k in case.outputs:
        out[k] = np.load(os.path.join(workdir, "out", k + ".npy"))
    return out, ""


def rel_l2(actual, expected):
    a = actual.astype(np.float64).ravel()
    e = expected.astype(np.float64).ravel()
    den = np.sqrt(np.sum(e * e))
    return float(np.sqrt(np.sum((a - e) ** 2)) / den) if den > 0 else float(np.sqrt(np.sum(a * a)))


def nmse(actual, expected):
    a = actual.astype(np.float64).ravel()
    e = expected.astype(np.float64).ravel()
    den = np.sum(e * e)
    return float(np.sum((a - e) ** 2) / den) if den > 0 else float(np.sum(a * a))


# GGUF access ---------------------------------------------------------------------------------------------------------

_readers = {}


def reader(which):
    if which not in _readers:
        from gguf import GGUFReader
        _readers[which] = GGUFReader(GGUF[which], "r")
    return _readers[which]


def tensor_raw(which, name):
    """Raw bytes (uint8 memmap) and (ggml type, ne tuple) of a tensor."""
    for t in reader(which).tensors:
        if t.name == name:
            return np.asarray(t.data).view(np.uint8).reshape(-1), t.tensor_type.name, tuple(int(x) for x in t.shape)
    raise KeyError(name)


def dequant(raw, ttype, ne):
    """Dequantize a tensor to float32 [rows, K] (ne[0] = K contiguous)."""
    k = ne[0]
    rows = int(np.prod(ne[1:])) if len(ne) > 1 else 1
    if ttype == "F32":
        return raw.view(np.float32).reshape(rows, k)
    if ttype == "BF16":
        w = raw.view(np.uint16).astype(np.uint32) << 16
        return w.view(np.float32).reshape(rows, k)
    if ttype == "Q8_0":
        b = raw.reshape(rows, k // 32, 34)
        d = b[:, :, :2].copy().view(np.float16).astype(np.float32)
        q = b[:, :, 2:].view(np.int8).astype(np.float32)
        return (q * d).reshape(rows, k)
    raise ValueError(ttype)


def load_weight(which, name):
    raw, ttype, ne = tensor_raw(which, name)
    return raw, ttype, ne, dequant(raw, ttype, ne)


# Oracle access -----------------------------------------------------------------------------------------------------

_manifests = {}


def oracle(prompt, name, occurrence=0):
    d = os.path.join(ORACLE, "Q8_K_XL_" + prompt)
    if d not in _manifests:
        _manifests[d] = json.load(open(os.path.join(d, "manifest.json")))
    for e in _manifests[d]["tensors"]:
        if e["name"] == name and e["occurrence"] == occurrence:
            ne = e["ne"]
            while ne and ne[-1] == 1:
                ne = ne[:-1]
            dt = np.int32 if e["dtype"] == "i32" else np.float32
            return np.fromfile(os.path.join(d, e["file"]), dtype=dt).reshape(ne[::-1])
    raise KeyError((name, occurrence))


def oracle_tokens(prompt):
    d = os.path.join(ORACLE, "Q8_K_XL_" + prompt)
    return np.array(json.load(open(os.path.join(d, "manifest.json")))["tokens"], dtype=np.int32)


class Report:
    def __init__(self):
        self.rows = []

    def add(self, name, arch, passed, metrics, note=""):
        self.rows.append((name, arch, passed, metrics, note))
        print("%-4s %-8s %-44s %s %s" % ("PASS" if passed else "FAIL", arch, name, metrics, note), flush=True)

    def finish(self):
        bad = [r for r in self.rows if not r[2]]
        print("%d cases, %d failed" % (len(self.rows), len(bad)))
        sys.exit(1 if bad else 0)


def bench_case(case, arch, workdir, batch=64, lock="-x", ring_mb=0, timeout=1800):
    """Runs loom-benchmark (dispatch_complete) on the case. Returns the parsed result JSON and the median time in microseconds."""
    os.makedirs(workdir, exist_ok=True)
    for k, a in case.inputs.items():
        save_fixture(workdir, k, a)
    test_path = os.path.join(workdir, case.name + "_test.loom")
    with open(test_path, "w") as f:
        f.write(case.source(benchmark=True))
    merged = os.path.join(workdir, case.name + ".loom")
    files = [os.path.join(PKG, p) for p in case.files] + [test_path]
    r = subprocess.run(["loom-link"] + files + ["--output=" + merged], capture_output=True, text=True)
    if r.returncode != 0:
        return None, "link: " + r.stderr[-3000:]
    res = os.path.join(workdir, "bench.json")
    cmd = ["flock", lock, LOCK, "loom-benchmark", merged, "--benchmark=@bench_" + case.name, "--device=" + DEVICES[arch], "--measure=dispatch_complete",
           "--batch-size=%d" % batch, "--output=" + res]
    if ring_mb:
        cmd.append("--input-ring-min-bytes=%d" % (ring_mb << 20))
    for k, v in {**tune_config(arch), **case.config}.items():
        cmd.append("--config=%s=%s" % (k, v))
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        return None, "bench: " + (r.stdout[-1500:] + r.stderr[-3000:])
    return json.load(open(res)), ""


def save_fixture(workdir, name, a):
    """Writes a fixture. Large int8 weight fixtures are kept when the file already has the same shape."""
    path = os.path.join(workdir, name + ".npy")
    if a.dtype == np.int8 and a.nbytes > (1 << 20) and os.path.exists(path):
        try:
            if np.load(path, mmap_mode="r").shape == a.shape:
                return
        except Exception:
            pass
    np.save(path, a)


def tune_config(arch):
    """Flat config of the arch row in kernel/dense/tune.json (the 'dispatch' object is not a config)."""
    rows = json.load(open(os.path.join(PKG, "kernel/dense/tune.json")))
    return {k: v for k, v in rows[arch].items() if k.startswith("inkling.")}
