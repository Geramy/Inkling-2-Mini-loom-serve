#!/bin/bash
# Regenerate oracle dumps and greedy continuations on the box (CPU only).
set -e
ROOT=${ROOT:-$HOME/inkling/inkB}
MD=${MD:-$HOME/models/inkling2-mini/Mainline_Support}
BIN=$ROOT/build/bin/llama-oracle-dump
OUT=$ROOT/oracle
T=${T:-16}
run() { # quant prompt mode gen outdir
    ORACLE_TOKENS=$OUT/prompts/$2.ids.txt ORACLE_OUT=$5 ORACLE_MODE=$3 ORACLE_GEN=$4 \
    $BIN -m $MD/Inkling-2-Mini-UD-$1.gguf -ngl 0 -t $T -ctk f32 -ctv f32 -c 1024 -b 512 -ub 512 > $5.log 2>&1
}
mkdir -p $OUT/dumps $OUT/gen
run Q8_K_XL A all   64 $OUT/dumps/Q8_K_XL_A
run Q8_K_XL B light 64 $OUT/dumps/Q8_K_XL_B
run Q8_K_XL C light 64 $OUT/dumps/Q8_K_XL_C
for p in A B C; do
    mkdir -p $OUT/gen/Q8_K_XL_$p
    cp $OUT/dumps/Q8_K_XL_$p/gen_ids.txt $OUT/dumps/Q8_K_XL_$p/gen_text.txt $OUT/gen/Q8_K_XL_$p/
done
for p in A B C; do
    mkdir -p $OUT/gen/Q4_K_XL_$p
    [ -n "$ONLY_Q8" ] || run Q4_K_XL $p none 64 $OUT/gen/Q4_K_XL_$p
done
