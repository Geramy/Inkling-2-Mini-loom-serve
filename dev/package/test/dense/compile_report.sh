#!/bin/sh
# Compile reports of every dense kernel for both targets (no GPU needed). Usage: compile_report.sh   (run in ~/inkling/inkF/package with the venv active)
P=${P:-$HOME/inkling/inkF/package}
LIB="$P/lib/inkling/dense_common.loom $P/lib/inkling/dense_float.loom $P/lib/inkling/dense_q8_0.loom $P/lib/inkling/dense_abi.loom $P/lib/inkling/dense_vec.loom $P/lib/inkling/dense_mm_wmma.loom $P/lib/inkling/dense_wmma.loom $P/lib/inkling/dense_targets.loom"
K="$P/kernel/dense/norm.loom $P/kernel/dense/embed.loom $P/kernel/dense/gemv.loom $P/kernel/dense/gemv_split.loom $P/kernel/dense/gemm_q8_0.loom $P/kernel/dense/gemm_f32.loom $P/kernel/dense/gemm_bf16_wmma.loom $P/kernel/dense/head.loom"
CFG="--config=inkling.dense.gemv.cols=8 --config=inkling.dense.gemv.xs_wg_per_unit=2 --config=inkling.dense.gemv.xs_depth=2 --config=inkling.dense.split.waves=4 --config=inkling.dense.f32.depth=3 --config=inkling.dense.f32.k_split=8"
for arch in gfx1151 gfx1201; do
  for k in inkling_rms_norm inkling_rms_norm_plain inkling_rms_norm_heads inkling_embed_q8_0 inkling_embed_bf16 inkling_gemv_q8_0 inkling_gemv_bf16 inkling_gemv_f32 inkling_gemv_split_bf16 inkling_gemv_split_f32 inkling_gemv_rows_bf16 inkling_gemv_rows_f32 inkling_gemm_q8_0 inkling_gemm_wide_q8_0 inkling_gemm_wmma_bf16 inkling_gemm_bf16 inkling_gemm_f32 inkling_head_select inkling_head_norm; do
    loom-link $LIB $K --root=@$k $CFG --output=/tmp/cr_$k.loom 2>/tmp/cr_err.txt || { echo "$arch $k LINK FAIL: $(head -c 300 /tmp/cr_err.txt)"; continue; }
    loom-compile /tmp/cr_$k.loom --target=amdgpu:$arch --compile-report=text-summary --output=/tmp/cr_$k.hsaco 2>&1 | awk -v a=$arch -v k=$k '
      /target_resources/ { for (i=1;i<=NF;i++) { if ($i ~ /^vector_final_registers=/) v=$i; if ($i ~ /^scalar_final_registers=/) s=$i; if ($i ~ /^occupancy_percent=/) o=$i; if ($i ~ /^resident_subgroups_per_simd=/) r=$i } }
      /^COMPILE-REPORT: allocation/ { for (i=1;i<=NF;i++) if ($i ~ /^spills=/) sp=$i }
      /static_instruction_mix/ { for (i=1;i<=NF;i++) { if ($i ~ /^wmma=/) w=$i; if ($i ~ /^local_memory=/) l=$i; if ($i ~ /^global_load=/) g=$i } }
      /^COMPILE-REPORT: emission/ { for (i=1;i<=NF;i++) if ($i ~ /^code_bytes=/) c=$i }
      /^COMPILE-REPORT: summary/ { for (i=1;i<=NF;i++) if ($i ~ /^status=/) st=$i }
      END { printf "%-8s %-26s %s %s %s %s %s %s %s %s %s\n", a, k, st, v, s, o, sp, w, l, g, c }'
  done
done
