"""Numpy reference and oracle accessors for the mix, controller and conv kernels (inkG)."""
import json
import os

import numpy as np

E = 2304
EPS = 1e-6
ORACLE = os.path.expanduser("~/inkling/inkB/oracle/dumps")
GGUF = {
    "Q8": os.path.expanduser("~/models/inkling2-mini/Mainline_Support/Inkling-2-Mini-UD-Q8_K_XL.gguf"),
    "Q4": os.path.expanduser("~/models/inkling2-mini/Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf"),
}


class Dump:
    def __init__(self, name="Q8_K_XL_A"):
        self.dir = os.path.join(ORACLE, name)
        m = json.load(open(os.path.join(self.dir, "manifest.json")))
        self.n = m["n_tokens"]
        self.ent = {(t["name"], t["occurrence"]): t for t in m["tensors"]}

    def get(self, name, occ=0):
        t = self.ent[(name, occ)]
        dt = np.float32 if t["dtype"] == "f32" else np.int32
        a = np.fromfile(os.path.join(self.dir, t["file"]), dtype=dt).reshape(t["ne"][::-1])
        while a.ndim > 2 and a.shape[0] == 1:
            a = a[0]
        return a


def phys(slot):
    return slot if slot < 24 else 6 + (slot - 24) % 12


_TW = {}


def x0(d):
    if "w" not in _TW:
        _TW["w"] = Gguf("Q8").f32("token_embd_norm.weight")
    return d.get("norm", 0) * _TW["w"]


def slot_input(d, slot):
    if slot == 0:
        return d.x0() if isinstance(d, Light) else x0(d)
    if slot == 18:
        return d.get("l_out-47")
    if slot in (24, 36):
        return d.get("l_out-5")
    return d.get("l_out-%d" % (slot - 1))


def blocks_at(d, slot, after_push):
    """Block list seen by the attention mix (after_push False) or by the ffn mix (True)."""
    if slot < 6:
        cand = [0]
    elif slot < 18:
        cand = [0, 6, 12]
    elif slot < 24:
        cand = [0, 36, 42, 18]
    elif slot < 36:
        cand = [0, 24, 30]
    else:
        cand = [0, 36, 42]
    if slot >= 18 and slot < 24:
        before = [s for s in cand if s != 18 and s != slot] + ([18] if slot > 18 else [])
        incl = before + ([slot] if after_push and slot == 18 else [])
    else:
        incl = [s for s in cand if s < slot or (s == slot and after_push)]
    return [slot_input(d, s) for s in incl]


def rms_inv(x):
    return 1.0 / np.sqrt((x.astype(np.float64) ** 2).mean(-1, keepdims=True) + EPS)


def mix_ref(states, query):
    """states [S][T][E], query [nq][E] -> out [nq][T][E], rinv_out [nq][T]."""
    st = np.stack(states).astype(np.float64)
    n = st * rms_inv(st)
    q = query.reshape(-1, E).astype(np.float64)
    sc = np.einsum("ste,qe->qst", n, q) / np.sqrt(E)
    sc -= sc.max(1, keepdims=True)
    w = np.exp(sc)
    w /= w.sum(1, keepdims=True)
    out = np.einsum("qst,ste->qte", w, n)
    return out, rms_inv(out)[..., 0]


def rms_norm(x, w=None):
    y = x * rms_inv(x)
    return y if w is None else y * w


def conv_ref(x, w, state=None):
    """x [T][C]; w [C][4] (tap 0 oldest); state [3][C] or None. Returns y [T][C], new state [3][C]."""
    t, c = x.shape
    h = np.zeros((3, c), np.float64) if state is None else state.astype(np.float64)
    xc = np.concatenate([h, x.astype(np.float64)], 0)
    y = np.zeros((t, c))
    for i in range(4):
        y += xc[i:i + t] * w[:, i].astype(np.float64)
    y = (x + y) / np.sqrt(5.0)
    return y, xc[-3:]


class Gguf:
    def __init__(self, kind="Q8"):
        import gguf
        self.r = gguf.GGUFReader(GGUF[kind])
        self.t = {t.name: t for t in self.r.tensors}

    def f32(self, name):
        t = self.t[name]
        a = np.asarray(t.data)
        if t.tensor_type.name == "BF16":
            a = np.ascontiguousarray(a).view(np.uint16)
            a = (a.astype(np.uint32) << 16).view(np.float32)
        return np.array(a, dtype=np.float32).reshape(t.shape[::-1])

    def raw(self, name):
        return np.asarray(self.t[name].data)


def ctrl_ref(x, m, wd, wf, wg):
    """x, m [T][E]; wd [64][4608]; wf, wg [2304][64]. Returns x' [T][E]."""
    x = x.astype(np.float64)
    m = m.astype(np.float64)
    z = np.concatenate([x * rms_inv(x), m * rms_inv(m)], -1) @ wd.astype(np.float64).T
    gain = np.tanh(z @ wg.astype(np.float64).T)
    forget = np.tanh(z @ wf.astype(np.float64).T)
    return x - forget * x + gain * m


CTRL = ["q", "k", "v", "ffn"]


def ctrl_weights(g, ip, c):
    n = "blk.%d.loop_%s_ctrl_" % (ip, CTRL[c])
    return g.f32(n + "down.weight"), g.f32(n + "forget.weight"), g.f32(n + "gain.weight")


def embed_x0(g, ids):
    """Normalized embedding rows of the given token ids (token_embd rows in BF16 for the Q8 file)."""
    t = g.t["token_embd.weight"]
    raw = np.asarray(t.data).reshape(t.shape[1], -1)
    rows = raw[ids]
    rows = np.ascontiguousarray(rows).view(np.uint16).astype(np.uint32) << 16
    e = rows.view(np.float32).reshape(len(ids), E)
    return (rms_norm(e) * g.f32("token_embd_norm.weight")).astype(np.float32)


class Light:
    """Prompt B or C: only l_out-*, result_norm and result_output were dumped."""

    def __init__(self, name, g):
        self.g = g
        self.dir = os.path.join(ORACLE, name)
        m = json.load(open(os.path.join(self.dir, "manifest.json")))
        self.n = m["n_tokens"]
        self.tokens = m["tokens"]
        self.ent = {(t["name"], t["occurrence"]): t for t in m["tensors"]}
        self._x0 = None

    def get(self, name, occ=0):
        t = self.ent[(name, occ)]
        a = np.fromfile(os.path.join(self.dir, t["file"]), dtype=np.float32).reshape(t["ne"][::-1])
        while a.ndim > 2 and a.shape[0] == 1:
            a = a[0]
        return a

    def x0(self):
        if self._x0 is None:
            self._x0 = embed_x0(self.g, self.tokens)
        return self._x0
