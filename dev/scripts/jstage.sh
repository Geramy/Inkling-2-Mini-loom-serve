#!/usr/bin/env bash
# usage: jstage.sh <tokens> [link|compile]   (on the box, venv active) builds the inkling_model command for one shape bucket with the serve source list; compile mode targets gfx1201 (ARCH=gfx1151 to change)
cd ~/inkling/inkK/run-sequence-gfx1201 || exit 1
SRC=$(python3 -c "import json;print(' '.join(json.load(open('model.json'))['definition']['sources']))")
T=${1:-1}
CFG="--config=inkling.tokens=$T --config=inkling.spans=2 --config=inkling.context=4096 --config=inkling.pool=65536 --config=inkling.packet_capacity=64 --config=inkling.dense.gemv.cols=$T --config=inkling.dense.gemv.xs_depth=2 --config=inkling.dense.split.waves=4 --config=inkling.dense.f32.depth=3 --config=inkling.dense.f32.k_split=8 --config=inkling.mix.ctrl_chunk=256 --config=inkling.mix.ctrl_tokens=1 --config=inkling.mix.ctrl_bf16=1"
if [ "${2:-link}" = compile ]; then
  loom-compile $SRC --root=@inkling_model --target=amdgpu:${ARCH:-gfx1201} $CFG --output=/tmp/jstage.hsaco 2>&1
else
  loom-link $SRC --root=@inkling_model $CFG --output=/tmp/jstage.loom 2>&1
fi | head -${LINES_MAX:-40} | cut -c1-${COLS_MAX:-260}
