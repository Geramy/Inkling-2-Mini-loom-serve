"""Reads expert tensors from the GGUF files and oracle dumps."""
import json, os
import numpy as np
import gguf

MODELS = os.path.expanduser("~/models/inkling2-mini/Mainline_Support")
FILES = {"q4": "Inkling-2-Mini-UD-Q4_K_XL.gguf", "q8": "Inkling-2-Mini-UD-Q8_K_XL.gguf"}
DUMPS = os.path.expanduser("~/inkling/inkB/oracle/dumps")

_readers = {}


def reader(kind):
    if kind not in _readers:
        r = gguf.GGUFReader(os.path.join(MODELS, FILES[kind]))
        _readers[kind] = (r, {t.name: t for t in r.tensors})
    return _readers[kind][1]


def tensor_bytes(kind, name):
    return np.asarray(reader(kind)[name].data)


def tensor_type(kind, name):
    return int(reader(kind)[name].tensor_type)


def expert_type(kind, layer):
    """0 NVFP4, 1 Q8_0, 2 BF16."""
    t = tensor_type(kind, f"blk.{layer}.ffn_gate_up_exps.weight")
    return {int(gguf.GGMLQuantizationType.Q8_0): 1, int(gguf.GGMLQuantizationType.BF16): 2}.get(t, 0)


def expert_weights(kind, layer):
    """Returns wtype, gate_up bytes [128][2304][rb], down bytes [128][2304][rb2], scales (gu, down) or None."""
    wt = expert_type(kind, layer)
    gu = tensor_bytes(kind, f"blk.{layer}.ffn_gate_up_exps.weight")
    dn = tensor_bytes(kind, f"blk.{layer}.ffn_down_exps.weight")
    gu = gu.reshape(128, 2304, -1)
    dn = dn.reshape(128, 2304, -1)
    sc = None
    if wt == 0:
        sgu = np.asarray(reader(kind)[f"blk.{layer}.ffn_gate_up_exps.scale"].data).astype(np.float32).reshape(-1)
        sdn = np.asarray(reader(kind)[f"blk.{layer}.ffn_down_exps.scale"].data).astype(np.float32).reshape(-1)
        sc = (sgu, sdn)
    return wt, gu, dn, sc


def router_weights(kind, layer):
    w = np.asarray(reader(kind)[f"blk.{layer}.ffn_gate_inp.weight"].data).astype(np.float32).reshape(128, 2304)
    b = np.asarray(reader(kind)[f"blk.{layer}.exp_probs_b.bias"].data).astype(np.float32).reshape(128)
    return w, b


class Dump:
    def __init__(self, name):
        d = os.path.join(DUMPS, name)
        self.dir = d
        self.m = json.load(open(os.path.join(d, "manifest.json")))
        self.idx = {}
        for t in self.m["tensors"]:
            self.idx[(t["name"], t["occurrence"])] = t

    def get(self, name, occ=0):
        t = self.idx[(name, occ)]
        dt = np.int32 if t["dtype"] == "i32" else np.float32
        a = np.fromfile(os.path.join(self.dir, t["file"]), dtype=dt)
        return a

    def last_occ(self, name):
        return max(o for (n, o) in self.idx if n == name)


def phys(slot):
    return slot if slot < 24 else 6 + (slot - 24) % 12
