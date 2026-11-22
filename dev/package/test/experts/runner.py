"""Builds a linked test module from kernel sources, runs it with loom-test and reads the written arrays."""
import os, subprocess, json, shutil, tempfile
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, "..", ".."))
URI = {"gfx1151": "amdgpu://id/000000e2000000808e270f2b0bbb4e00", "gfx1201": "amdgpu://GPU-e0880c05003c324d"}
LOCK = os.path.expanduser("~/hrx-pr1/gpu.lock")

DT = {"f32": "f32", "f16": "f16", "i32": "i32", "i8": "i8", "u8": "i8"}


def ttype(a):
    kind = {np.dtype(np.float32): "f32", np.dtype(np.float16): "f16", np.dtype(np.int32): "i32",
            np.dtype(np.int8): "i8", np.dtype(np.uint8): "i8"}[a.dtype]
    return "tensor<" + "x".join(str(d) for d in a.shape) + "x" + kind + ">"


def lib_files(*kernels):
    libs = sorted(os.path.join(PKG, "lib", "inkling", f) for f in os.listdir(os.path.join(PKG, "lib", "inkling")) if f.endswith(".loom"))
    return libs + [os.path.join(PKG, "kernel", "experts", k) for k in kernels]


class Case:
    """One check.case: inputs (npy fixtures), outputs (written npy), launches (text)."""

    def __init__(self, name, workdir):
        self.name, self.dir = name, workdir
        os.makedirs(workdir, exist_ok=True)
        self.lines, self.decls, self.outputs = [], [], {}
        self.types = {}

    def inp(self, name, arr):
        if arr.dtype == np.uint8:
            arr = arr.view(np.int8)
        np.save(os.path.join(self.dir, name + ".npy"), np.ascontiguousarray(arr))
        self.types[name] = ttype(arr)
        self.lines.append(f'  %{name} = check.file.read.npy path("{name}.npy") : {self.types[name]}')

    def zeros(self, name, shape, dtype):
        a = np.zeros(shape, dtype)
        self.types[name] = ttype(a)
        z = "0.0" if a.dtype.kind == "f" else "0"
        self.lines.append(f"  %{name} = check.generate.fill value({z}) : {self.types[name]}")

    def lit(self, name, value, ty="index"):
        self.lines.append(f"  %{name} = check.literal value({value}) : {ty}")

    def launch(self, kernel, workload, args, decl_sig):
        """args: list of (name, kind) with kind 'index' or tensor name; decl_sig: (workload params, launch params) text."""
        wl, lp = decl_sig
        self.decls.append(f"kernel.decl @{kernel}({wl}) launch({lp})")
        tys = ", ".join("index" if k == "index" else self.types[k] for _, k in args)
        names = ", ".join("%" + n for n, _ in args)
        wnames = ", ".join("%" + w for w in workload)
        wtys = ", ".join("index" for _ in workload)
        br = f"[{wnames}]" if workload else ""
        sig = f"[{wtys}]({tys})" if workload else f"({tys})"
        self.lines.append(f"  kernel.launch @{kernel}{br}({names}) : {sig}")

    def out(self, name):
        self.outputs[name] = self.types[name]
        self.lines.append(f'  check.file.write.npy value(%{name}) path("{name}.npy") mode(always) : {self.types[name]}')

    def source(self):
        body = "\n".join(self.lines)
        decls = "\n".join(sorted(set(self.decls)))
        return f"{decls}\n\ncheck.case public @{self.name} {{\n{body}\n  check.return\n}}\n"

    def run(self, arch, kernels, lock="-s", extra=()):
        src = os.path.join(self.dir, "case.loom")
        open(src, "w").write(self.source())
        merged = os.path.join(self.dir, "merged.loom")
        subprocess.run(["loom-link", *lib_files(*kernels), src, "--output=" + merged], check=True)
        cmd = ["loom-test", merged, "--device=" + URI[arch], "--file-output-root=" + os.path.join(self.dir, "out"), *extra]
        if lock:
            cmd = ["flock", lock, LOCK] + cmd
        r = subprocess.run(cmd, capture_output=True, text=True)
        rep = None
        try:
            rep = json.loads(r.stdout)
        except Exception:
            pass
        if r.returncode != 0:
            raise RuntimeError(f"loom-test failed ({self.name}, {arch}):\n{r.stdout[-3000:]}\n{r.stderr[-3000:]}")
        return {n: np.load(os.path.join(self.dir, "out", n + ".npy")) for n in self.outputs}


def _inp_file(self, name, path, shape, dtype):
    """Reads a fixture that already exists as .npy at an absolute path."""
    a = np.zeros((0,), dtype)
    kind = {np.dtype(np.float32): "f32", np.dtype(np.float16): "f16", np.dtype(np.int32): "i32", np.dtype(np.int8): "i8", np.dtype(np.uint8): "i8"}[a.dtype]
    self.types[name] = "tensor<" + "x".join(str(d) for d in shape) + "x" + kind + ">"
    self.lines.append(f'  %{name} = check.file.read.npy path("{path}") : {self.types[name]}')


Case.inp_file = _inp_file


def cached_npy(path, make):
    """Writes make() to path (as int8 for uint8 data) unless it exists. Returns the shape."""
    if not os.path.exists(path):
        a = make()
        if a.dtype == np.uint8:
            a = a.view(np.int8)
        np.save(path, np.ascontiguousarray(a))
        return a.shape
    return tuple(np.load(path, mmap_mode="r").shape)
