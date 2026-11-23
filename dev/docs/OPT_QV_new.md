# OPT_QV: 4/5/6-bit dense weights (custom requantized files), lane QV

Bead llama-cpp-tg3.36. Base: package_T10b (= package/ after wave 1). Patch: `lanes/QV.patch`. Records: `bench/decisions_QV.jsonl`. Step 1 (the Unsloth file memo, stop rule hit) is kept at the end of this file; step 2 (custom files and kernels, approved by the lead) is the main text.

## Result in one table

| File | attn q/k/v/gate/out/rel | lm_head | KLD vs UD-Q8_K_XL (CPU, wikitext 32 x 512) | same top | bytes per token | decode ceiling R9700 / 8060S | file size |
|---|---|---|---|---|---|---|---|
| UD-Q4_K_XL (base) | Q8_0 | Q8_0 | 0.00335 (1.0x) | 97.64% | 4010.8 MB | 154.6 / 57.3 | 17.63 GB |
| V1 | Q6_K | Q6_K | 0.00457 (1.36x) | 97.38% | 3473.4 MB | 178.5 / 66.2 | 17.30 GB |
| V2x | Q5_K | Q6_K | 0.00687 (2.05x) | 96.65% | 3302.3 MB | 187.7 / 69.6 | 17.22 GB |
| V3x | Q4_K | Q6_K | 0.01434 (4.3x) | 95.22% | 3141.3 MB | 197.4 / 73.2 | 17.14 GB |
| V4x | Q4_K | Q4_K | 0.01631 (4.9x) | 95.01% | 3021.9 MB | 205.2 / 76.1 | 17.02 GB |

(The ceilings are `cost_model.py` of the real inventories of these files, weights only, 620 and 230 GB/s. Unsloth's own table for orientation, other data set: Q4_K_XL KLD 0.0107 and 98.0%, IQ4_XS 0.0109 and 97.7%, Q3_K_XL 0.0174 and 96.3%.) Loop controllers are BF16 in these files (UD-Q8_K_XL source), F32 in the Unsloth Q4 files: 113 MB per token of the gain below comes from that and needs no quality tradeoff.

Measured gain over UD-Q4_K_XL on the published route (ABBA, median of sessions): PERF_SUMMARY

Selection bar from the lead: KLD within about 2x of Q4_K_XL. V1 passes clearly (1.36x), V2x is on the edge (2.05x, same top 96.65% against 97.64%), V3x and V4x fail (4.3x and 4.9x). Kernels exist for all of them; V3x and V4x were not tuned or checked beyond the kernel tests and a speed measurement.

## What was done

1. `llama-quantize` built from `~/inkling/inkB/src` into `~/inkling/optQV/build` (CPU, target llama-quantize only; inkB untouched).
2. Files V1 to V4 made first with the layer 20-23 experts in Q5_K/Q4_K (V2 to V4). The down tensors of the experts have K = 1152, which is not a multiple of 256, so no K-quant fits them; only gate_up could change, which saves 64 MB per token (Q5_K) or 85 MB (Q4_K) for a new expert kernel type. Final files V2x to V4x keep the experts of layers 20-23 in Q8_0 (as UD-Q4_K_XL). V2, V3, V4 were deleted.
3. Quality first (CPU, 8 threads under nice 19): `llama-perplexity` KLD against UD-Q8_K_XL.
4. Kernels Q6_K, Q5_K, Q4_K for the attention projections and the lm_head: GEMV (decode, up to 8 rows) and matrix-op GEMM (prefill); generator switch; card variants; numerics; ABBA performance.

## Files and the exact commands

`scripts_QV/qv_quantize.sh <name> <attn type> <lm_head type>` writes `~/models/inkling2-mini/custom/Inkling-2-Mini-<name>.gguf` from `UD-Q8_K_XL`:

```
bash scripts_QV/qv_quantize.sh V1  q6_k q6_k
bash scripts_QV/qv_quantize.sh V2x q5_k q6_k
bash scripts_QV/qv_quantize.sh V3x q4_k q6_k
bash scripts_QV/qv_quantize.sh V4x q4_k q4_k
```

It runs this command (ATTN and HEAD are the types, SRC is the UD-Q8_K_XL file, nice 19, 8 threads):

```
llama-quantize --allow-requantize --pure \
  --tensor-type 'attn_(q|k|v|gate|output|rel)\.weight=ATTN' \
  --tensor-type 'attn_sink_proj\.weight=bf16' --tensor-type '_conv_proj\.weight=bf16' \
  --tensor-type 'ffn_gate_inp\.weight=f32' \
  --tensor-type 'blk\.(1?[0-9])\.ffn_(gate_up|down)_exps\.weight=nvfp4' \
  --tensor-type 'blk\.2[0-3]\.ffn_(gate_up|down)_exps\.weight=q8_0' \
  --output-tensor-type HEAD --token-embedding-type q8_0  SRC OUT q8_0 8
```

Same-type tensors are byte-copied (the NVFP4 experts of layers 0-19 and their `.scale` tensors); norms, conv taps, clips, `rel_bias`, loop controllers and message queries are not quantizable and are copied. Checked per tensor with `scripts/gguf_inventory.py` against UD-Q4_K_M: the only type changes are attention, lm_head, and the loop controllers (BF16 as in the Q8_K_XL source; F32 in the Unsloth Q4 files). token_embd stays Q8_0. Router F32, sink and conv projections BF16 unchanged.

## Quality (CPU, same GGUF through the oracle build)

`llama-perplexity -m <gguf> -f wiki.test.raw -c 512 --chunks 32 -ngl 0 -t 8 --kl-divergence-base base.kld` once for UD-Q8_K_XL (PPL 6.4143), then `--kl-divergence-base base.kld --kl-divergence` for every file (logs `~/inkling/optQV/kld/*.log`).

| File | PPL (32 chunks) | PPL / PPL(Q8_K_XL) | mean KLD | KLD 99th percentile | max KLD | same top p |
|---|---|---|---|---|---|---|
| UD-Q4_K_XL | 6.4284 | 1.0022 | 0.003354 +- 0.000135 | 0.0343 | 0.62 | 97.64 +- 0.17 |
| V1 | 6.4245 | 1.0016 | 0.004567 +- 0.000165 | 0.0402 | 0.98 | 97.38 +- 0.18 |
| V2x | 6.4310 | 1.0026 | 0.006874 +- 0.000208 | 0.0549 | 0.84 | 96.65 +- 0.20 |
| V3x | 6.4616 | 1.0074 | 0.014339 +- 0.001125 | 0.1104 | 8.80 | 95.22 +- 0.24 |
| V4x | 6.4729 | 1.0092 | 0.016312 +- 0.001124 | 0.1161 | 8.75 | 95.01 +- 0.24 |

PPL differences between the files are inside the PPL error bar (0.17 on 6.4); KLD and same-top are the usable signals. Our KLD scale differs from Unsloth's (their Q4_K_XL 0.0107, ours 0.0034: other text, other base logits), so compare ratios: Unsloth IQ4_XS 1.02x and Q3_K_XL 1.63x of Q4_K_XL; ours V1 1.36x, V2x 2.05x, V3x 4.3x. The V3x and V4x tail (max KLD 8.8) is a few positions where the 4-bit attention flips an expert choice.

## Kernels and package changes (all in lanes/QV.patch)

- `lib/inkling/dense_kq.loom` (Q4_K, Q5_K) and `dense_q6_k.loom` (Q6_K): copies of PR1 `motifs/kq.loom` and `q6_k.loom` (llama.cpp branch hrx-llama-pr1, our MIT code) with the prefix `ggml` renamed to `inkling.dense` and `func.def` made public (as for `dense_q8_0.loom`; the providers `weight.block/values4/superblock/part16/offsets16/wmma16` already used these names). The decode (part16 with the xs GEMV plan) and the matrix-op decode (wmma16) are the tuned PR1 code; PR1 has no per-type tune rows, so the existing `tune.json` rows apply (xs_wg_per_unit 4 on gfx1151, 2 on gfx1201, xs_depth 2).
- `kernel/dense/gemv.loom`: `inkling_gemv_{q4_k,q5_k,q6_k}`, each one call of `inkling_dense_gemv_xs` with the weight type code 2, 5, 6.
- `kernel/dense/gemm_kquant.loom` (new): `inkling_gemm_wmma_{q4_k,q5_k,q6_k}`, 128 x 64 tile, F16 operands, F32 sums, the PR1 generic `ggml_mul_mat_wmma` schedule with the `wmma16` provider staging the weights; same ABI as `inkling_gemm_wmma_bf16`. Compile report: no spills; 100 vgpr (gfx1201) and 96 (gfx1151) for all three (the BF16 matrix GEMM: 136 and 112).
- `scripts_QV/gen_hooks.py`: `linear()` takes `kq`, the name of a config (`inkling.kq_attn`, `inkling.kq_head`); value 0 keeps the old Q8_0/BF16 code unchanged, 6 / 5 / 2 select Q6_K / Q5_K / Q4_K kernels (GEMV for up to 8 rows, matrix GEMM above). Used by the six attention projections (9 call sites) and the lm_head. `dense_type` 2 = "Q8_0 embedding and layer 20-23 experts, BF16 loop controllers" (`Body.is_q8()` replaces the three `dense == 0` tests; `dispatch/visit.loom` expert weight type; `prepare.loom`: `ctrl_bf16 = dense != 0`; ranges 0 to 2). Generated: `hooks_dense.loom`, `hooks_mix.loom`, `hooks_experts.loom`, `kernel_decls.loom` (run `gen_kernel_decls.py` first, then `gen_hooks.py`: the hooks take the kernel declarations from `kernel_decls.loom`).
- `card.json`: three source entries, variants `custom-v1`, `custom-v2x`, `custom-v3x`, `custom-v4x` (chat and capture; local file `Inkling-2-Mini-V1.gguf` etc.; `dense_type` 2 and the kq codes); `card_defaults.loom`: `kq_attn`, `kq_head` default 0. `scripts/publish_build.py` is not edited, so the published tree does not have the new files and variants until the integrator adds them.
- Tests: `test/dense/dtest.py` reads Q4_K/Q5_K/Q6_K with the gguf package dequantizer (independent of our kernels); `test_gemv.py <arch> - kq` (22 cases), `test_gemm.py <arch> - gemm_wmma_{q4_k,q5_k,q6_k}` (9, 9, 10 cases).

Shape facts: attention K is 2304 (9 super blocks) and 3072 (12): both multiples of 256; attn_rel has 384 rows; the lm_head has 201024 rows (200058 used).

## Numerics

- Kernel tests against numpy from the GGUF bytes, both GPUs, all pass: GEMV nmse 1e-14 (lm_head with soft cap 2e-10), matrix GEMM below the 5e-6 gate (F16 rounding of weights and activations, the same behaviour as the Q8_0 matrix GEMM), padding rows untouched.
- End to end on the published serve route (model directory with `loom-serve --preset chat[-r9700]`, prefill capacity 2048): skills `golden.py` against llama.cpp CPU running the SAME custom GGUF (22 prompts: 20 short and documents of 1k and 2k tokens, 64 tokens each, second recording `golden2.json` as `--alt`), `compare.py` near-tie rule:

| File | GPU | prompts | exact 64-token matches | median matched prefix | max gap at first divergence | verdict |
|---|---|---|---|---|---|---|
| V1 | R9700 | 22 | 11 | 63 | 0.849 | PASS |
| V1 | 8060S | 22 | 10 | 50 | 0.849 | PASS |
| V2x | R9700 | 22 | 8 | 50 | 0.436 | PASS |
| V2x | 8060S | 22 | 8 | 50 | 0.436 | PASS |

- Perplexity of the Loom path: capture route (capture id 31 returns the logits of every row) fed with the 512-token chunks stored in the llama-perplexity KL base file (same tokens), PPL over rows n_ctx/2 to n_ctx-2 as llama-perplexity does, first 8 chunks (`scripts_QV/qv_ppl.py`; vocabulary 200058; Loom logits are soft-capped). CPU values are the running PPL of the llama-perplexity logs at chunk 8:

| File | llama.cpp CPU | Loom R9700 | Loom 8060S |
|---|---|---|---|
| UD-Q8_K_XL | 7.6559 | | |
| V1 | 7.6726 | 7.6529 | 7.6701 |
| V2x | 7.6569 | 7.6635 | |

Loom is within 0.3% of the CPU result for the same file (the CPU rounds activations to Q8 for the dot products; Loom keeps F32 activations and is lower on every chunk, in the direction of the unrounded value).

## Performance

PERF_TABLE

## Recommendation and stop reason

RECOMMENDATION

## Incident (my fault, reported to the lead)

`qv_mkvar.sh` first built model directories with `cp -al` from `~/inkling/inkV/mat-<gpu>` and wrote `model.json` in place. `model.json` is hard-linked into other lanes' directories, so for a while the `model.json` of `inkV/mat-r9700`, `inkV/mat-strix` and the directories linked to them (optD1, optD5, integ1 var dirs) pointed to the V1 file, had 3 extra library entries and prefill_capacity 2048. The lead was told at once (affected inodes and repair script `/tmp/qv_repair.py` on the box: restores `dense_type 0`, the Q4_K_XL parameter path and origin, removes the 3 entries, sets prefill_capacity 512). I did not run the repair: the permission system blocked writing the shared files. The script now works on a private copy of the materialized preset (`~/inkling/optQV/mat-<gpu>`, real copies, weights excluded) and writes `model.json` with unlink and create.

## How to rebuild and reproduce

```
bash scripts_QV/sync_qv.sh        # laptop: package_QV, scripts_QV, package_T10b to ~/inkling/optQV on the box
# on the box, in ~/inkling/optQV:
python3 scripts/gen_kernel_decls.py && python3 scripts/gen_hooks.py       # after any kernel or hook change, in this order
scripts/qv_mkvar.sh r9700 v1 ~/inkling/optQV/package ~/models/inkling2-mini/custom/Inkling-2-Mini-V1.gguf 2 6 6 2048   # gpu name package gguf dense_type kq_attn kq_head prefill
scripts/qv_mkvar.sh r9700 base ~/inkling/optQV/package_T10b ~/models/inkling2-mini/Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf 0 0 0 2048
scripts/qv_ab.sh r9700 ~/inkling/optQV/var-r9700-base ~/inkling/optQV/var-r9700-v1 dshort,d8k,d32k,pf2k,c4 3 1   # ABBA, exclusive lock per arm
scripts/qv_arm.sh r9700 ~/inkling/optQV/var-r9700-v2x dshort,d8k,d32k,pf2k,c4 3 2                               # single-arm sessions
scripts/qv_compare.sh r9700 v1 ~/inkling/optQV/var-r9700-v1 out/cmp       # skills compare vs W/bench_v1/golden*.json
scripts/qv_ppl_run.sh r9700 custom-v1 8 out/ppl.json                      # capture route perplexity
(cd package/test/dense && python test_gemv.py gfx1201 - kq && python test_gemm.py gfx1201 - gemm_wmma_q6_k)
```

`qv_mkvar.sh` arguments: gpu (`r9700` or `strix`), name, package dir, gguf, dense_type (0 = Q4_K_XL file, 2 = custom), kq_attn, kq_head (type codes 6 Q6_K, 5 Q5_K, 2 Q4_K), prefill capacity. Variants used: v1 `2 6 6`, v2x `2 5 6`, v3x `2 2 6`, v4x `2 2 2`. Golden recordings of llama.cpp CPU (`-t 8`, nice 19): `~/inkling/optQV/W/bench_v1`, `bench_v2x` (`env.sh` with `V=v1|v2x`). Lock waits dominated the wall time of this lane (a shared or exclusive request waited behind 5 to 9 queued exclusive timed runs of other lanes; a session took 8 to 15 minutes).

## Leftovers to delete (box)

`~/inkling/optQV` (private mat copies, var dirs with symlinks, `kld/` with the 3.3 GB `base.kld`, `build/`, `W/`, `out/`), `~/models/inkling2-mini/custom/*.gguf` (about 69 GB; keep the file the owner chooses), `/tmp/qv_*`, `/tmp/qv_t`.

---

# Step 1 memo (Unsloth files, stop rule)

STEP1
