#!/usr/bin/env bash
# usage: jlink.sh   (local) syncs and links every kernel root of the four areas with loom-link (fast error check, no GPU)
cd "$(dirname "$0")/.."
INKLING_BOX_DIR=inkling/inkK bash scripts/sync.sh 2>&1 | grep -v future
ssh user@gpu-box '. ~/loom-serve/.venv-inkK/bin/activate; cd ~/inkling/inkK/package
LIB=$(ls lib/inkling/*.loom | tr "\n" " ")
CFG="--config=inkling.dense.gemv.cols=8 --config=inkling.dense.gemv.xs_depth=2 --config=inkling.dense.split.waves=4 --config=inkling.dense.f32.depth=3 --config=inkling.dense.f32.k_split=8 --config=inkling.mix.ctrl_chunk=128 --config=inkling.mix.ctrl_tokens=16 --config=inkling.mix.ctrl_bf16=1"
for f in kernel/dense/*.loom kernel/attn/*.loom kernel/experts/*.loom kernel/mix/*.loom; do
  for r in $(grep -o "^kernel.def [a-z() \"_]*@[a-z_0-9]*" $f | sed "s/.*@//"); do
    loom-link $LIB kernel/dense/*.loom kernel/attn/*.loom kernel/experts/*.loom kernel/mix/*.loom --root=@$r $CFG --output=/tmp/jl_$r.loom > /tmp/jl_err.txt 2>&1 && echo "ok $r" || { echo "FAIL $r"; head -c 400 /tmp/jl_err.txt; }
  done
done'
