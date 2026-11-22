"""Numpy references for the expert kernels. Weights are raw GGUF bytes; math is f32/f64."""
import numpy as np

SLOT = [0, 4, 1, 5, 8, 12, 9, 13, 2, 6, 3, 7, 10, 14, 11, 15]
ELEM_OF_SLOT = np.argsort(SLOT)
KV = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12], dtype=np.float32)
ROW_BYTES = {0: 36 * 36, 1: 72 * 34, 2: 2304 * 2}


def ue4m3(b):
    b = b.astype(np.int32) & 0x7F
    exp = (b >> 3) & 15
    man = b & 7
    raw = np.where(exp == 0, man * 2.0 ** -9, (1 + man / 8.0) * 2.0 ** (exp.astype(np.float64) - 7))
    raw = np.where(b == 0x7F, 0.0, raw)
    return raw.astype(np.float32) * 0.5


def dec_nvfp4(raw):
    """raw: (..., nblocks, 36) uint8 -> (..., nblocks*64) float32 (without the per-expert scale)."""
    d = ue4m3(raw[..., :4])
    qs = raw[..., 4:].reshape(raw.shape[:-1] + (4, 8))
    lo = KV[qs & 15]
    hi = KV[qs >> 4]
    out = np.concatenate([lo, hi], axis=-1) * d[..., None]
    return out.reshape(raw.shape[:-2] + (-1,))


def dec_q8_0(raw):
    d = raw[..., :2].copy().view(np.float16).astype(np.float32)
    q = raw[..., 2:].view(np.int8).astype(np.float32)
    return (q * d).reshape(raw.shape[:-2] + (-1,))


def dec_bf16(raw):
    u = raw.reshape(raw.shape[:-1] + (-1, 2)).astype(np.uint32)
    bits = (u[..., 0] | (u[..., 1] << 8)) << 16
    return bits.astype(np.uint32).view(np.float32)


def decode_rows(wtype, raw, k):
    """raw: (rows, row_bytes) uint8 -> (rows, k) float32."""
    if wtype == 0:
        return dec_nvfp4(raw.reshape(raw.shape[0], k // 64, 36))
    if wtype == 1:
        return dec_q8_0(raw.reshape(raw.shape[0], k // 32, 34))
    return dec_bf16(raw)


def to_slots(x):
    """x: (..., k) -> slot order per unit of 16."""
    s = x.reshape(x.shape[:-1] + (-1, 16))
    return s[..., ELEM_OF_SLOT].reshape(x.shape)


def from_slots(x):
    s = x.reshape(x.shape[:-1] + (-1, 16))
    return s[..., np.array(SLOT)].reshape(x.shape)


def situ(g, u):
    sig = 1.0 / (1.0 + np.exp(-g))
    return 4 * np.tanh(g / 4) * sig * 25 * np.tanh(u / 25)


def rel_l2(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return float(np.linalg.norm(a - b) / max(np.linalg.norm(b), 1e-30))


def random_nvfp4_rows(rng, rows, k, lo=0x30, hi=0x78):
    nb = k // 64
    raw = rng.integers(0, 256, size=(rows, nb, 36), dtype=np.uint8)
    raw[..., :4] = rng.integers(lo, hi, size=(rows, nb, 4), dtype=np.uint8)
    return raw.reshape(rows, nb * 36)
