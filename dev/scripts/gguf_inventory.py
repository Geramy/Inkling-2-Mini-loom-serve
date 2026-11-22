import sys, collections
import numpy as np
from gguf import GGUFReader, GGMLQuantizationType, GGUFValueType

def fmt_val(f):
    t = f.types[0]
    if t == GGUFValueType.ARRAY:
        n = len(f.data)
        et = f.types[1]
        if et == GGUFValueType.STRING:
            vals = [bytes(f.parts[i]).decode("utf-8", "replace") for i in f.data]
        else:
            vals = [f.parts[i].tolist()[0] for i in f.data]
        if n > 64:
            return f"array[{et.name}] n={n} head={vals[:8]} tail={vals[-4:]}"
        return f"array[{et.name}] n={n} {vals}"
    if t == GGUFValueType.STRING:
        s = bytes(f.parts[f.data[0]]).decode("utf-8", "replace")
        return s if len(s) < 200 else f"<string {len(s)} chars> {s[:80]!r}..."
    return f.parts[f.data[0]].tolist()[0] if f.parts[f.data[0]].size == 1 else f.parts[f.data[0]].tolist()

def main(path, out):
    r = GGUFReader(path, "r")
    with open(out, "w", encoding="utf-8") as o:
        o.write(f"# {path}\n## METADATA\n")
        for k, f in r.fields.items():
            if k.startswith("GGUF."):
                continue
            o.write(f"{k} [{f.types[0].name}] = {fmt_val(f)}\n")
        o.write(f"\n## TENSORS ({len(r.tensors)})\n# name | shape (ggml ne order) | type | bytes\n")
        fam = collections.defaultdict(lambda: collections.defaultdict(lambda: [0, 0]))
        tot = 0
        import re
        for t in r.tensors:
            shape = "x".join(str(int(x)) for x in t.shape)
            o.write(f"{t.name} | {shape} | {t.tensor_type.name} | {int(t.n_bytes)}\n")
            key = re.sub(r"blk\.\d+\.", "blk.N.", t.name)
            e = fam[key][t.tensor_type.name]
            e[0] += 1; e[1] += int(t.n_bytes)
            tot += int(t.n_bytes)
        o.write(f"\n## FAMILY SUMMARY (name pattern: type -> count, bytes)\n")
        for key, d in fam.items():
            o.write(key + ": " + "; ".join(f"{ty} x{c} {b}B" for ty, (c, b) in d.items()) + "\n")
        o.write(f"\nTOTAL tensor bytes: {tot}\n")

main(sys.argv[1], sys.argv[2])
