# Roofline for the looped architecture

Script: `scripts/roofline_inkling.py` (standard library only). Output: `bench/<gpu>/roofline.json` (inputs, extras, measured ratios) and `bench/<gpu>/ev_roofline_<variant>.json` (evidence block, schema `loom-card-evidence/2`). `skills/loom-roofline/roofline.py` does not model the loop (Q4: 2.315 GB and 267.8 tok/s on the R9700 at 620 GB/s) and is not edited.

Run: `python3 scripts/roofline_inkling.py` (reads `docs/gguf_inventory_*.txt`). Cross-check, on the box with the tools venv: `python -I scripts/roofline_inkling.py --crosscheck --skills ~/inkling/loom-skills/skills --gguf-dir <dir with the GGUF files>` (output in `docs/roofline_crosscheck.txt`).

## Constants

| GPU | Bandwidth | Matrix peak (F16) |
|---|---|---|
| R9700 (gfx1201) | 620 GB/s, measured read (`docs/DEVICES.md`, `floor_bench.py`; preset 613) | 191.4 TFLOP/s, preset `radeon-r9700` (spec; our kernels reach 30 to 37) |
| 8060S (gfx1151) | 230 GB/s, measured read (preset 228) | 43.8 TFLOP/s, preset `strix-halo` at the 85 W mode (spec 59, 123 W mode 48.4) |

The bandwidth was not measured again in this task. The power mode of the box is not recorded.

## Decode bytes per token

Per token the model runs 48 layer visits (`phys(v)` in `cost_model.py`). Each visit reads its physical layer again: the three loop passes do not share traffic, because pass n+1 depends on pass n and one layer (about 100 MB) does not fit in cache.

| Family | Read per token |
|---|---|
| Attention q, k, v, o, gate, rel, sink; norms, clips, conv taps, rel_bias, res_query | every visit (48x) |
| Dynamic conv projections | 8 global visits |
| Router (F32) | every visit, whole tensor |
| Experts | 8 of 128 per visit: 8 x (gate_up + down) / 128 per tensor; 44 NVFP4 visits (layers 0-19), 4 Q8_0/BF16 visits (layers 20-23); the NVFP4 `.scale` tensors are ignored (512 B) |
| Loop controllers and message queries | passes 2 and 3 only: 24 visits |
| lm_head | once per token (the whole matrix) |
| token_embd | one row |

Totals: Q4_K_XL 4010.77 MB, Q8_K_XL 5778.28 MB (equal to `cost_model_output.txt`).

State per context token, added to the weights (`decode_bytes`):
- Global KV: 8 visits x 4096 B x ctx (K and V, 8 heads x 128, F16).
- Local KV: 40 visits x 4096 B x min(ctx, 1024).
- KV write: 48 x 4096 B. Conv state read and write: 2 x 48 x 79,872 B = 7.7 MB.

Ceiling = bandwidth / bytes. Rows: short (ctx 0), 2000, 8000 (extra), 32000. Short context is 154.3 and 57.2 tok/s on Q4, 107.2 and 39.8 on Q8: 0.2 percent under the documented floors (154.6, 57.3, 107.3, 39.8) because of the conv state and KV write.

## 4 streams (rows 4)

One pass reads the weights once for all 4 rows, plus per row the KV at ctx 230 (the skills' mean context of that row), KV write and conv state. Ceiling = 4 x bandwidth / bytes. The four rows are assumed to use the same 8 experts per visit, the hard bound, as the skills do for MoE (`basis: shared-experts`). With independent routing (expected distinct experts 128 x (1 - (15/16)^4) = 29.1 of 128 per visit) the bound is lower: `independent_routing` in `roofline.json` (Q4 R9700 272.5 against 587.2).

## Prefill

Chunks of `prefill_capacity` tokens (Q4: 2048 since wave 1, Q8: 512; `CHUNKS` in the script). For each chunk of n tokens at context c0:
- FLOPs = n x 8.84 GFLOP (2 x MACs of the weights, expert FFN, controllers on 24 visits, conv; `cost_model.py` constants) + 12,288 x (8 x sum(c0+i+1) + 40 x sum(min(c0+i+1, 1024))) + the lm_head 0.93 GFLOP once, on the last chunk (output row only).
- Bytes = the weights of one decode token with ALL experts of every visit (expected distinct experts 1 - (15/16)^n of 128, so all of them at n = 512) + KV read (global 4096 x (c0+n) per visit, local 4096 x min(c0+n, 1023+n)) + KV write; lm_head once on the last chunk.
- Time = max(FLOPs / peak, bytes / bandwidth), summed over the chunks. Prompts: 1000, 2000, 8000, 16000, 32000 (the skills' decimal sizes).

A chunk reads about 31 GB (Q4) of weights. At 512 tokens per chunk (the first baseline, and Q8 now) prefill is memory bound on both GPUs (R9700 9.8k tok/s memory, 20.9k compute; 8060S 3.6k and 4.8k). At 2048 tokens per chunk (Q4 presets after wave 1) the weight traffic is spread over four times the tokens: the Q4 prefill ceilings are compute bound for prompts of 2k tokens and more (R9700 17.5k to 20.5k, 1k prompt 19.3k memory bound; 8060S 4.0k to 4.8k). Both limits are in `roofline.json` (`compute_only`, `memory_only`). The compute-only ceiling is what `roofline.py` computes (agrees within 0.01 percent at 1000 tokens). Activation traffic is not modelled.

## Efficiency, target, stretch

Copied from `roofline.py` at loom-skills 618f89e (MoE rows): decode 0.78 capped by 35 us x 48 visits per token, KV share at 0.97; 4 streams 0.85; prefill 0.33 of the spec peak (scaled by spec / peak: 0.445 on the 8060S at 85 W). Stretch: 0.88, 0.95, 0.45. `target = ceiling x efficiency`. These are skills defaults from other models; they are not measured on this model.

## Cross-check (`docs/roofline_crosscheck.txt`)

roofline.py's own functions run on the GGUF with the visits substituted (per-layer bytes x visit count, controllers on 24 visits, 48-visit KV geometry): weights agree within 0.003 percent, KV bytes exactly, prefill FLOPs per token within 0.4 percent (conv taps counted as matmul params there), decode ceilings within 0.25 percent (ours is lower by the conv state and KV write).

## Measured Loom against the ceiling (`INTEGRATION_LOG.md`, first measurements)

| | Decode ctx 16 | Decode ctx 8005 | Prefill 520 tokens |
|---|---|---|---|
| R9700 Q4 | 86.6 / 154.3 = 0.56 | 74.9 / 139.4 = 0.54 | 2202 / 7068 = 0.31 |
| R9700 Q8 | 69.7 / 107.2 = 0.65 | 61.9 / 99.7 = 0.62 | 2186 / 6015 = 0.36 |
| 8060S Q4 | 38.9 / 57.2 = 0.68 | 33.3 / 51.7 = 0.64 | 900 / 2622 = 0.34 |
| 8060S Q8 | 29.3 / 39.8 = 0.74 | 26.1 / 37.0 = 0.71 | 833 / 2231 = 0.37 |

All below 0.97. Prefill at 520 tokens is a 512 chunk plus an 8 token chunk, which reads the weights twice.

## Evidence status

`make_info.py roofline` computes from the GGUF with `roofline.py`, so it cannot take this table (it would write the no-loop numbers). The `ev_roofline_*.json` files are assembled by hand in the same block layout (`device`, `tool`, `checkpoint`, `drafting: null`, `metrics` with `ceiling`, `efficiency`, `basis`) and checked with `card_lib._check` against the `roofline` entry of `info_schema.json`: valid; a damaged copy fails. They are not made by the skills tool: `tool.script` names our script and `tool.skills_commit` names the loom-skills commit whose constants it copies. The device name differs from the skills presets (measured bandwidth). Mark this as unverified by the skills' own audit.

## Measured Loom against the ceiling after wave 1 (`PERF_BASELINE.md`, Q4, median of 3 sessions)

`MEASURED_BASELINE` in the script (`measured_loom_wave1` in `roofline.json`). Ceilings with the 2048 token prefill chunk.

| | decode | 2k | 8k | 32k | prefill 1k | 2k | 8k | 16k | 32k | 4 streams |
|---|---|---|---|---|---|---|---|---|---|---|
| R9700 measured (tok/s) | 90.2 | 83.2 | 80.3 | 71.3 | 2576 | 2996 | 3092 | 2801 | 2242 | 218.9 |
| R9700 / ceiling | 0.585 | 0.571 | 0.576 | 0.602 | 0.133 | 0.146 | 0.157 | 0.148 | 0.128 | 0.373 |
| 8060S measured (tok/s) | 39.7 | 36.7 | 34.9 | 29.7 | 894 | 1033 | 1051 | 973 | 812 | 89.0 |
| 8060S / ceiling | 0.694 | 0.678 | 0.675 | 0.677 | 0.186 | 0.220 | 0.233 | 0.225 | 0.202 | 0.409 |

Before wave 1 (512 token chunks, ceiling 9.8k and 3.6k): decode 0.56 / 0.68, prefill 0.14 to 0.21 / 0.16 to 0.23 (`PERF_BASELINE.md`, previous baseline). Decode gained 0.02 to 0.04 of the ceiling; prefill moved from memory bound to compute bound, where the ceiling is about twice as high and the measured value 28 to 63 percent higher on the R9700. All ratios stay below 0.97. The `measured_loom` block (first measurements, INTEGRATION_LOG) is kept for the Q8 rows; for Q4 its prefill ratio is now against the 2048 chunk ceiling and is not comparable with the table above of the first baseline.
