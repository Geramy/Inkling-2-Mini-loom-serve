#!/usr/bin/env python3
"""Compare two tensor dump directories (manifest.json + raw .bin files) against the oracle.

Usage: compare_to_oracle.py REF_DIR TEST_DIR [--profile f32|q8|q4] [--only REGEX] [--skip REGEX] [--allow-missing] [--tol-nmse X] [--tol-cos X]

Tensors are matched by (name, occurrence). Layout: ne[0] is the fastest dimension (ggml order), the file is raw
little-endian f32 or i32, so the numpy shape is ne reversed. Exit code 0 when every compared tensor passes.
"""
import argparse
import json
import os
import re
import sys

import numpy as np

# profile: (max NMSE, min cosine, max fraction of mismatching i32 entries)
PROFILES = {
    "f32": (1e-6, 0.999999, 0.0),
    "q8": (1e-3, 0.9995, 0.02),
    "q4": (1e-2, 0.995, 0.25),
}


def load_dump(path):
    with open(os.path.join(path, "manifest.json")) as f:
        man = json.load(f)
    entries = {}
    for t in man["tensors"]:
        entries[(t["name"], t["occurrence"])] = t
    return man, entries


def load_tensor(path, t):
    dtype = np.float32 if t["dtype"] == "f32" else np.int32
    shape = tuple(t["ne"][::-1])
    a = np.fromfile(os.path.join(path, t["file"]), dtype=dtype)
    return a.reshape(shape)


def compare(a, b):
    """a = reference, b = test. Returns dict of metrics."""
    if a.dtype == np.int32:
        if a.shape != b.shape:
            return {"shape_mismatch": True}
        # expert ids: compare as sets per row (order may differ for equal weights)
        flat_a = a.reshape(-1, a.shape[-1]) if a.ndim > 1 else a.reshape(1, -1)
        flat_b = b.reshape(-1, b.shape[-1]) if b.ndim > 1 else b.reshape(1, -1)
        bad = sum(1 for ra, rb in zip(flat_a, flat_b) if set(ra.tolist()) != set(rb.tolist()))
        return {"mismatch": bad / max(1, len(flat_a)), "rows": len(flat_a)}
    a64 = a.astype(np.float64).ravel()
    b64 = b.astype(np.float64).ravel()
    if a.shape != b.shape:
        return {"shape_mismatch": True}
    d = a64 - b64
    ref_energy = float(np.sum(a64 * a64))
    err_energy = float(np.sum(d * d))
    na = float(np.sqrt(ref_energy))
    nb = float(np.sqrt(np.sum(b64 * b64)))
    return {
        "max_abs": float(np.max(np.abs(d))) if d.size else 0.0,
        "mean_abs": float(np.mean(np.abs(d))) if d.size else 0.0,
        "nmse": err_energy / ref_energy if ref_energy > 0 else (0.0 if err_energy == 0 else float("inf")),
        "cos": float(np.dot(a64, b64) / (na * nb)) if na > 0 and nb > 0 else (1.0 if na == nb else 0.0),
        "finite": bool(np.all(np.isfinite(b64))),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ref")
    ap.add_argument("test")
    ap.add_argument("--profile", choices=sorted(PROFILES), default="f32")
    ap.add_argument("--only", help="regex on tensor name")
    ap.add_argument("--skip", help="regex on tensor name")
    ap.add_argument("--allow-missing", action="store_true", help="tensors absent from TEST are not failures")
    ap.add_argument("--tol-nmse", type=float)
    ap.add_argument("--tol-cos", type=float)
    ap.add_argument("--quiet", action="store_true", help="print only failures and the summary")
    args = ap.parse_args()

    tol_nmse, tol_cos, tol_ids = PROFILES[args.profile]
    if args.tol_nmse is not None:
        tol_nmse = args.tol_nmse
    if args.tol_cos is not None:
        tol_cos = args.tol_cos

    _, ref = load_dump(args.ref)
    _, tst = load_dump(args.test)
    only = re.compile(args.only) if args.only else None
    skip = re.compile(args.skip) if args.skip else None

    n_pass = n_fail = n_miss = 0
    print(f"{'tensor':<34}{'occ':>4}{'max_abs':>12}{'mean_abs':>12}{'nmse':>12}{'cos':>11}  result")
    for key in sorted(ref, key=lambda k: ref[k]["seq"]):
        name, occ = key
        if only and not only.search(name):
            continue
        if skip and skip.search(name):
            continue
        if key not in tst:
            n_miss += 1
            if not args.allow_missing:
                n_fail += 1
            print(f"{name:<34}{occ:>4}{'-':>12}{'-':>12}{'-':>12}{'-':>11}  MISSING")
            continue
        m = compare(load_tensor(args.ref, ref[key]), load_tensor(args.test, tst[key]))
        if m.get("shape_mismatch"):
            n_fail += 1
            print(f"{name:<34}{occ:>4}{'':>12}{'':>12}{'':>12}{'':>11}  FAIL (shape)")
            continue
        if "mismatch" in m:
            ok = m["mismatch"] <= tol_ids
            line = f"{name:<34}{occ:>4}{'':>12}{'':>12}{m['mismatch']:>12.4f}{'':>11}  {'PASS' if ok else 'FAIL'} (id rows differing, {m['rows']} rows)"
        else:
            ok = m["finite"] and m["nmse"] < tol_nmse and m["cos"] >= tol_cos
            line = f"{name:<34}{occ:>4}{m['max_abs']:>12.3e}{m['mean_abs']:>12.3e}{m['nmse']:>12.3e}{m['cos']:>11.6f}  {'PASS' if ok else 'FAIL'}"
        n_pass += ok
        n_fail += not ok
        if not args.quiet or not ok:
            print(line)

    print(f"\nprofile={args.profile} tol: nmse<{tol_nmse:g} cos>={tol_cos:g} id_rows<={tol_ids:g}")
    print(f"pass={n_pass} fail={n_fail} missing={n_miss}")
    print("RESULT:", "PASS" if n_fail == 0 else "FAIL")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
