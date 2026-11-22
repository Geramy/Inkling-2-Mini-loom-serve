#!/bin/sh
# Runs every dense test on both GPUs (shared GPU lock). Usage (box, venv active): run_all.sh [outdir]. Exit code 1 if any case fails.
cd "$(dirname "$0")"
OUT=${1:-$HOME/inkling/inkF/results}
mkdir -p "$OUT"
rc=0
for arch in gfx1151 gfx1201; do
  for t in "test_norm_embed.py" "test_gemv.py - xs" "test_gemv.py - split" "test_gemv.py - rows" "test_gemm.py - gemm_q8_0" "test_gemm.py - gemm_wide_q8_0" \
           "test_gemm.py - gemm_wmma_bf16" "test_gemm.py - gemm_bf16" "test_gemm.py - gemm_f32" "test_head.py" "test_oracle_layers.py"; do
    name=$(echo "$t" | tr ' ' '_' | tr -d '.-')
    set -- $t
    f=$1; shift
    if [ "$f" = test_norm_embed.py ] || [ "$f" = test_head.py ] || [ "$f" = test_oracle_layers.py ]; then
      python $f $arch > "$OUT/${arch}_$name.txt" 2>&1 || rc=1
    else
      python $f $arch "$@" > "$OUT/${arch}_$name.txt" 2>&1 || rc=1
    fi
    echo "$arch $t: $(tail -1 "$OUT/${arch}_$name.txt")"
  done
done
exit $rc
