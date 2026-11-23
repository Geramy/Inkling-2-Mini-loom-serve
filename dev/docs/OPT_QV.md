# OPT_QV: dense-weight quant candidates (decision memo, step 1)

Result: STOP. No Unsloth file gives at least +10% decode ceiling at quality close to UD-Q4_K_XL. Step 2 was not started.

Method: tensor headers of the three files fetched (first 48 MB of each, anonymous range request; sparse files in `~/inkling/optQV/hdr` on the box), `scripts/gguf_inventory.py` then `scripts/cost_model.py` (620 and 230 GB/s, same visit plan, no KV). Inventories on the box: `~/inkling/optQV/inv/UD-*.txt`.
Repo contents (Mainline_Support): IQ1_S, IQ1_M, IQ2_XXS, Q2_K_XL, IQ3_XXS, Q3_K_XL, IQ4_XS, Q4_K_M, Q4_K_XL, Q8_K_XL. There is no Q5 and no Q6 file.

## Table

| File | Size GB | KLD | Same top | Bytes/token MB | R9700 ceiling | gain | 8060S ceiling | gain |
|---|---|---|---|---|---|---|---|---|
| UD-Q4_K_XL (now) | 17.63 | 0.0107 | 98.0% | 4010.8 | 154.6 | - | 57.3 | - |
| UD-Q4_K_M | 16.61 | 0.0108 | 97.9% | 3947.1 | 157.1 | +1.6% | 58.3 | +1.6% |
| UD-IQ4_XS | 15.73 | 0.0109 | 97.7% | 3766.2 | 164.6 | +6.5% | 61.1 | +6.5% |
| UD-Q3_K_XL | 13.63 | 0.0174 | 96.3% | 3537.7 | 175.3 | +13.4% | 65.0 | +13.4% |
| UD-Q8_K_XL (ref) | 22.87 | ref | ref | 5778.3 | 107.3 | -30.6% | 39.8 | -30.6% |

## Why the gain is small: Unsloth keeps the dense tensors at 8 bit

Dense types per file (all 24 layers unless noted):
- Q4_K_M: attn q/k/v/gate/output/rel Q8_0, sink_proj BF16, lm_head Q8_0, token_embd Q8_0. Experts: NVFP4 layers 0-19, layers 20-23 gate_up Q5_K, down Q8_0 (Q4_K_XL: both Q8_0). Only the layer 20-23 experts change (-0.27 GB to 0.21 GB).
- IQ4_XS: attn Q8_0 (a few layers Q6_K/Q5_K), lm_head Q6_K, token_embd Q8_0, layers 20-23 experts Q4_K/Q5_K/IQ4_XS mix. Per token: attn 1275 MB Q8_0 + 66 MB Q6_K + 5 MB Q5_K; lm_head 380 MB.
- Q3_K_XL: attn mostly Q8_0 (1152 MB) with some Q6_K/Q5_K/Q4_K (q is Q6_K in layer 0), lm_head and token_embd Q5_K, experts NVFP4 only in part of the layers 0-19 (1218 MB against 1577 MB); the other expert layers use IQ2_XS, IQ2_S, IQ3_XXS, IQ3_S, Q3_K; layers 20-23 IQ3_XXS/Q3_K mix. The gain comes from the experts, not the dense weights.
- No file uses Q4_K/Q5_K/IQ4_NL/MXFP4 for the bulk of the attention projections. The NVFP4 experts stay identical in the three 4-bit files.

## Kernel coverage in PR1 ggml-hrx motifs (hrx-llama-pr1)

Covered: Q4_K, Q5_K (kq.loom), Q6_K (q6_k.loom), Q8_0 (q8_0.loom); vec/decode and WMMA paths (mul_mat, mm_*), tuned for gfx1151 and gfx1201. Not covered: IQ4_XS, IQ4_NL, IQ3_XXS, IQ3_S, IQ2_XS, IQ2_S, Q3_K. IQ4_XS would need only lm_head (Q6_K: covered) and the layer 20-23 experts (Q4_K, Q5_K covered, IQ4_XS not). Q3_K_XL needs 5 new type families.

## Prefill

Prefill at chunk 2048 is compute bound (docs/ROOFLINE.md: R9700 17.5k-20.5k, 8060S 4.0k-4.8k tok/s ceilings, measured 0.13-0.23 of it). Smaller dense weights do not move the compute bound; only the expert and chunk traffic term changes slightly. Lower-bit types add dequant ALU work in the GEMM (K-quants cost more per weight than Q8_0), so a prefill gain is not expected. Not measured.

## Decision

1. Stop rule: Q4_K_M +1.6% and IQ4_XS +6.5% are below +10%. Q3_K_XL (+13.4%) has KLD 0.0174 (1.6x Q4_K_XL) and same-top 96.3% (-1.7 points), and needs 5 new kernel families (IQ2/IQ3 plus Q3_K). Not "close to Q4_K_XL" quality: not recommended. If the owner wants it anyway, IQ4_XS is the cheapest candidate (+6.5% for almost no quality loss, 1-2 new block decoders) but small.
2. The 8-bit dense tensors are the lever, and no published file changes them. Hypothetical ceilings (same cost model; only attn q/k/v/gate/output/rel, lm_head and layer 20-23 experts change type; sink_proj, controllers, router unchanged; bytes per weight Q8_0 1.0625, Q6_K 0.820, Q5_K 0.6875, Q4_K 0.5625):

| Variant | MB/token | R9700 | 8060S | gain |
|---|---|---|---|---|
| now | 4010.8 | 154.6 | 57.3 | 0 |
| attn Q6_K, lm_head Q6_K | 3586.6 | 172.9 | 64.1 | +11.8% |
| attn Q5_K, lm_head Q6_K, L20-23 experts Q5_K | 3320.0 | 186.7 | 69.3 | +20.8% |
| attn Q4_K, lm_head Q6_K, L20-23 experts Q4_K | 3127.1 | 198.3 | 73.6 | +28.3% |
| attn Q4_K, lm_head Q4_K, L20-23 experts Q4_K | 3007.7 | 206.1 | 76.5 | +33.4% |

All four use only types that PR1 already covers (Q4_K, Q5_K, Q6_K): kernel effort is small (motif copy plus the dense_type switch). They need a self-made GGUF: `llama-quantize` from UD-Q8_K_XL (its attention and lm_head are BF16, a clean source) with per-tensor type overrides; the quality of such a file is unknown (K-quants on attention are known to cost more KLD than Q8_0; measure KLD against UD-Q8_K_XL with llama-perplexity first, before any kernel work). Recommendation: do this quantization experiment as step 2 instead of any Unsloth file, starting with "attn Q6_K + lm_head Q6_K" (+11.8%, smallest quality risk) and "attn Q5_K". The owner's 80 / 170 tok/s figures match a roughly 3.0-3.4 GB per token file, i.e. Q4_K attention, not the Unsloth files.
3. Measured decode is 0.58-0.69 of the ceiling, so the real gain would be about the same percentage if the kernels stay bandwidth bound (K-quant decode ALU cost is a risk on the 8060S).

Candidate rating (gain x quality x kernel effort): custom attn Q6_K/lm_head Q6_K (+11.8%, quality to measure, low effort) > custom attn Q5_K (+20.8%, medium quality risk, low effort) > IQ4_XS (+6.5%, good quality, low-medium effort) > Q3_K_XL (+13.4%, worse quality, high effort) > Q4_K_M (+1.6%).

## Not done

No kernels, no GGUF downloads beyond the headers, no GPU runs, no patch (nothing to patch). Scratch to delete: `~/inkling/optQV` on the box (about 150 MB real size).
