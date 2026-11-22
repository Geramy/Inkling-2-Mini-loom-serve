#!/bin/sh
# Full correctness run on one GPU: run_all.sh gfx1201|gfx1151 (on the box, venv active). Logs in ~/inkling/inkG/logs.
g=${1:-gfx1201}
cd "$(dirname "$0")"
L=$HOME/inkling/inkG/logs
mkdir -p $L
python test_mix.py $g > $L/mix_$g.log 2>&1; tail -1 $L/mix_$g.log
python test_conv.py $g > $L/conv_$g.log 2>&1; tail -1 $L/conv_$g.log
python test_ctrl.py $g 128 all 1 > $L/ctrl_${g}_c128_t1.log 2>&1; tail -1 $L/ctrl_${g}_c128_t1.log
python test_ctrl.py $g 256 all 16 > $L/ctrl_${g}_c256_t16.log 2>&1; tail -1 $L/ctrl_${g}_c256_t16.log
for c in 64 128 256; do python test_ctrl.py $g $c q4 8 > $L/ctrl_${g}_q4_c$c.log 2>&1; tail -1 $L/ctrl_${g}_q4_c$c.log; done
