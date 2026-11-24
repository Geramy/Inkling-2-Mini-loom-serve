#!/usr/bin/env bash
# usage: gates_job.sh r9700|strix   (box; run under flock -x) dev-level skills gates against W/mat-<gpu> of INKR (the oracle server must run)
G=$1
export INKR=$HOME/inkling/integ4 LOOM_VENV=$HOME/inkling/tools/M30-venv SEED=$HOME/inkling/integ2/W/bench-$G
S=$HOME/inkling/integ4/scripts/run_gates.sh
cd $INKR
trap "bash $S $G stop" EXIT
bash $S $G prep || exit 1
bash $S $G serve || exit 1
for c in /sys/class/drm/card[0-9]/device; do echo "$c vram $(cat $c/mem_info_vram_used) gtt $(cat $c/mem_info_gtt_used)"; done > mem_after_load_$G.txt
bash $S $G gates
bash $S $G extra
for c in /sys/class/drm/card[0-9]/device; do echo "$c vram $(cat $c/mem_info_vram_used) gtt $(cat $c/mem_info_gtt_used)"; done > mem_after_ctx_$G.txt
echo GATES_JOB_DONE
