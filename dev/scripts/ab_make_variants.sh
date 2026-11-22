#!/usr/bin/env bash
# usage: ab_make_variants.sh <r9700|strix> (GPU box) Builds scratch variants of the fetched package as hard-link trees (no weight copy) with one knob changed each.
set -eu
G=${1:?gpu}; V=${INKV:-$HOME/inkling/inkV}; B=$V/mat-$G
P=$(case $G in r9700) echo chat-r9700 ;; strix) echo chat ;; esac)
mk() { rm -rf "$V/var-$G-$1"; cp -al "$B" "$V/var-$G-$1"; }
mk pc256; python3 - "$V/var-$G-pc256/model.json" "$P" <<'PY'
import json, sys
p, pre = sys.argv[1:]
d = json.load(open(p)); d["presets"][pre]["execution"]["prefill_capacity"] = 256
json.dump(d, open(p + ".new", "w"), indent=1)
PY
mv "$V/var-$G-pc256/model.json.new" "$V/var-$G-pc256/model.json"
for n in 1 3 4; do
  mk xs$n; f=$V/var-$G-xs$n/program/lib/inkling/dense_vec.loom
  python3 - "$f" $n <<'PY'
import sys
p, n = sys.argv[1:]
s = open(p).read()
a = "priority(-1) @inkling_dense_xs_wg_per_unit_default() -> (index) {\n  %value = index.constant 2 : index"
assert a in s
open(p + ".new", "w").write(s.replace(a, a.replace("constant 2", f"constant {n}")))
PY
  mv "$f.new" "$f"
done
mk split2x; f=$V/var-$G-split2x/program/dispatch/hooks_attn.loom
python3 - "$f" <<'PY'
import sys
p = sys.argv[1]
s = open(p).read()
i = s.index("@inkling_attn_split_blocks_gfx1201")
j = s.index("template.return", i)
blk = s[i:j].replace("%upper = scf.select %medium, %c4, %c8", "%upper = scf.select %medium, %c8, %c16").replace("%grown = scf.select %small, %c2, %upper", "%grown = scf.select %small, %c4, %upper")
blk = blk.replace("  %c8 = index.constant 8 : index\n", "  %c8 = index.constant 8 : index\n  %c16 = index.constant 16 : index\n")
open(p + ".new", "w").write(s[:i] + blk + s[j:])
PY
mv "$f.new" "$f"
ls "$V" | grep "^var-$G"
