# Optimize plan: HAL profiles, bucket budgets, ranked lanes (T10 steps 1 and 2, inkV, 2026-11-21, bead llama-cpp-tg3.21)

No optimization is implemented here. Everything below is measured (profiles, A/B sessions) or marked as an estimate. Tree: the refreshed publish tree (inkS), presets `chat-r9700` (gfx1201) and `chat` (gfx1151), context 131072, rows 4, prefill 512, Q4_K_XL, same programs as `docs/PERF_BASELINE.md` (`programs_sha256` 51c67d50...). Raw profiles: `bench/<gpu>/profile/raw/`. Full per-family tables: `bench/<gpu>/profile/summary.md` and `.json`. Condensed tables: `bench/<gpu>/profile/condensed.md` (same content for both GPUs, section 2).

## Lane status (2026-11-22, integ3)

| Lane | Status | Where |
|---|---|---|
| 1 P1 larger-n experts pass | kept, in wave 1 (published) | `OPT_P1.md` |
| 3 D1 residual-chain fusion | kept, in wave 1 | `OPT_D1.md` |
| 6 D4 long-context decode attention | kept, in wave 1 | `OPT_D4.md` |
| 8 D5 head and sampler | kept, in wave 1 | `OPT_D5.md` |
| 7 P3 prefill attention K/V tile | rejected, not wired | `OPT_P3.md` |
| QV quantized variants | separate lane, not part of the published package | `OPT_QV.md`, `OPT_QV_new.md` |
| 4 D2 attention front end and projection group | kept, in wave 2 (published) | `OPT_D2.md` |
| 9 D6 experts decode pipelining | kept, in wave 2 | `OPT_D6.md` |
| W64 wave64 down kernel (8060S only) | kept, in wave 2 | `OPT_W64.md` |
| 5 D3 controllers and router | kept, in wave 2 | `OPT_D3.md` |
| M30 hrx-loom 0.3.0 and merge of wave 2 | done, `package/` = `package_M30` | `OPT_M30.md` |
| 2, 10, 11, 12 (P2, P4, P5, prefill variance) | not run in waves 1 and 2 | this plan |

Wave 2 result (merged, one session per GPU): decode short 94.86 (R9700) and 42.80 (8060S) tok/s, 0.61 and 0.75 of the ceiling (`PERF_BASELINE.md`).

## 0. Summary

1. Decode (R9700, short context, 11.64 ms per token): the weight streams (experts, proj, oproj, lm_head) are 7.3 ms of device time against a 6.0 ms floor and are already at 67 to 97 percent of the measured bandwidth. The small kernels (mix, ctrl, norm, conv, router, attention front, sampler) are 4.4 ms in about 1,020 launches (average 4.3 us, 37 percent of the step) against a floor of about 0.5 ms. The largest single family is the mix kernel: 147 launches per token, 1.57 ms, 10.7 us per launch (the isolated benchmark in `KERNELS_mix.md` gives 4.7 to 6.1 us). The attention projection GEMVs are NOT a lever any more: 2.42 ms for 2.22 ms of bytes (92 percent of the bandwidth). The "3.0 vs 2.2 ms" of INTEGRATION_LOG included the launch gaps and the small kernels around them.
2. Launch gaps are not a separate bucket in the HAL accounting. Device busy (sum of dispatch times) is 11.66 ms per step and the profiled wall is 12.59 ms; the unprofiled wall is 11.64 ms. Profiling costs about 1.0 ms per step (R9700) and 1.4 ms (8060S), about 0.7 to 0.9 us per dispatch with 1,569 dispatches per step. Busy is overstated by about the same amount, so idle gap plus host time is between 0 and 1.0 ms (0 to 9 percent) on the R9700 and 0 to 0.5 ms on the 8060S. A trivial dependent launch costs 0.7 to 1.4 us inside its own duration (experts_combine 0.72 us, conv_commit 1.2 us, prep_spans 1.3 us). Launch merging is priced per removed launch (about 1 to 3 us each), not from a gap bucket.
3. Prefill (R9700, 512 token chunk, 220 ms at 2k): experts 125 ms (57 percent; running at 27 TFLOP/s and 240 GB/s, so neither compute nor memory bound), attention 34 ms (16 percent, growing to 167 ms and 48 percent at 32k, 7 TFLOP/s equivalent), small kernels with controllers 34 ms (15 percent), proj and oproj 25 ms (11 percent). The whole drop of the prefill rate with context (2.3k to 1.4k tok/s) is the attention kernel (30, 59, 163 ms per chunk at 2k, 8k, 32k).
4. Measured in this task (R9700 and 8060S): prefill chunk 256 instead of 512 costs 26 percent (R9700) and 21 percent (8060S). A linear model `chunk time = fixed + per-token x n` gives fixed 77 ms and 0.29 ms per token (R9700), 154 ms and 0.81 ms (8060S). The fixed part is the weight traffic of the chunk (29.5 GB of experts). With the present kernel efficiency a chunk of 1024 / 2048 would give about +20 / +34 percent (R9700) and +16 / +26 percent (8060S). This is the largest prefill lever and the proof of the larger-n lane.
5. The "prefill session drift" is session-to-session variance of a fresh server, not a time trend: two sessions of the same build 90 seconds apart gave 2292 and 2223 tok/s (3 percent) on the R9700. Prefill A/B needs ABBA with two sessions per arm and a 4 percent keep threshold; decode A/B is stable to 0.5 percent.
6. Config-only quick wins: none for decode. xs_wg_per_unit 1 ties (-0.6 percent), 4 loses 30 percent (section 6). Chunk 256 loses (see 4).
7. The prefill numbers of PERF_BASELINE are reconciled in section 1.4: `perf.py` uses max_tokens 4 and a 2454 token prompt with a short last chunk, my sessions max_tokens 1 and a 2040 token prompt; the same server gives 2063 (perf.py method), 2128 (max_tokens 1, same prompt) and 2215 (2040 tokens).

## 1. Method, tools, limits

### 1.1 Route

- `scripts/profile_run.sh <r9700|strix> <session>` (run under `flock -x ~/hrx-pr1/gpu.lock`): starts one `loom-serve` (spin URI as the baseline, `--pool_backing=fixed --workers=8`) with `--device_profiling_mode=dispatch-events,executable-metadata --device_profiling_flush_interval_ms=5 --print_device_statistics` on the hard-linked fetch dir (`loom-fetch --materialize=share`, no weight copy; disk stayed at 45 GB free), runs `scripts/profile_driver.py` for the session, stops the server with SIGTERM by PID and waits for the exit. The kernel table is in `<session>.serve.log`. `scripts/profile_all.sh <gpu>` runs the 12 sessions, each under its own exclusive lock, one server at a time on an idle box (`pgrep` check inside the script).
- Sessions: `base` (warm-up request only), `dshort_p/_d`, `d8k_p/_d`, `d32k_p/_d` (decode after a short, 9k and 37k prompt; `_p` = prompt plus one decode step, `_d` = N tokens), `pf2k` (3 prompts of 2040 tokens, each 3 chunks of 512 and one of 504), `pf8k` (9082 tokens), `pf32k` (37006 tokens), `c4_p/_d` (4 concurrent streams, own prompts). A scenario table is the difference of two sessions (`_d` minus `_p`, prefill minus `base`), so weight load, first-use work, the warm-up request and the prompt prefill cancel. Decode rows are per step (a step is one token for all rows), prefill rows per 512 token chunk.
- `scripts/profile_report.py PROF_DIR ROOFLINE_JSON OUT --gemm-tflops X` writes the family tables, `scripts/profile_condense.py` the five-bucket tables of section 2, `scripts/profile_families.json` is our families file (same format as the skills' `profile_families.json`: first regex that matches a kernel name wins; names are ours, `package/kernel/{experts,attn,dense,mix}`). Cross-check: the skills' `profile_stats.py` on the raw `dshort_d` log gives device busy 4843.4 ms, equal to the HAL `dispatch_function_total` (4843.379 ms) and to our row sum.
- No `dropped_records` in any of the 24 profile sessions (5 ms flush is enough for 1,569 dispatches per step). Graceful stop: every server exited after SIGTERM (`alive=no` in `prof/all-*.out`).

### 1.2 Limits of the HAL table (they shape the plan)

- One row per kernel function name, merged over every hook that launches it. `inkling_gemv_q8_0` is q, k, v, gate, rel (proj), oproj and the lm_head. `inkling_gemm_q8_0` is the same in prefill. `inkling_mix_states` is mix_attn, mix_ffn, both message mixes and the head mix. `inkling_rms_norm` is norm_in and norm_ffn. We split by launch count (5 of 6 gemv launches per visit are proj) and by weight bytes (oproj is 26.4 percent of the six matrices), and take the lm_head time from its own row (the prefill sessions have one lm_head launch per request: 1.19 ms on the R9700, 2.34 ms on the 8060S, i.e. 414 and 210 GB/s). A per-hook split of the merged kernels needs the `inkling.stop_after` config (`scripts/jsplit.sh`) or a profile build with per-hook kernel aliases; not done. The proj/oproj/head split is therefore an apportionment.
- The statistics are per dispatch packet; the duration includes the wave launch and the dependency wait at the packet start. This is why a dependent one-block kernel never shows below 0.7 us.
- Profiling overhead: profiled wall minus unprofiled wall is 1.0 ms per decode step on the R9700 (12.59 against 11.64, the latter confirmed today by `perf.py`: 86.0 tok/s) and 1.4 ms on the 8060S (27.27 against 25.91). Prefill is not visibly affected (profiled 2284, unprofiled 2257 to 2292 tok/s).

### 1.3 Floors

- Decode floor per family = bytes per token (`bench/<gpu>/roofline.json`, `bytes_by_family`: experts 1847 MB, proj 1015 MB, oproj 361 MB, controllers 227 MB, router 57 MB, lm_head 492 MB, misc 11 MB, conv state 7.7 MB; attention = KV bytes at the context) divided by the measured bandwidth (R9700 620 GB/s, 8060S 230 GB/s). The sum at short context is 6.49 ms and 17.5 ms (154.3 and 57.2 tok/s: the ceilings of `ROOFLINE.md`).
- Prefill floor per family and chunk = max(bytes / bandwidth, flops / best GEMM rate). Bytes: all 128 experts of every visit are read once per chunk (29.55 GB), proj, oproj, controllers, router once, KV. Flops: 2 x MACs per token x 512 (experts 6.1 of 8.8 GFLOP per token). Best GEMM rate: R9700 120 TFLOP/s (skills `device/rdna4`: q4_k m256 kernels 95 to 125 TFLOP/s on real weights; spec 191), 8060S 30 TFLOP/s (skills `device/rdna3/strix-halo`: real weights and random activations, 85 W power limit; register-only peak 43.8). Attention flops use the same rate (an optimistic lower bound). Activation traffic and norms are not modelled, so small families have a near zero floor.
- 4 rows: the experts floor in the tables is the shared-routing bound (2.98 / 8.03 ms). With independent routing (29.1 of 128 distinct experts per visit) it is 10.8 / 29.2 ms. Measured experts: 9.79 / 28.4 ms. The experts of the 4-row step are therefore at their traffic floor, and the 4-row gap in the table is not a kernel gap.

### 1.4 Reconciliation with the wall-clock baseline (`PERF_BASELINE.md`)

| Row | PERF_BASELINE tok/s | Profiled wall (this task) | Unprofiled today | Note |
|---|---|---|---|---|
| R9700 decode short | 85.9 | 79.4 (12.59 ms) | 86.0 (`perf.py`), 85.4 to 86.1 (A/B) | profiling +1.0 ms/step |
| R9700 decode 8k | 76.3 | 71.7 (13.95 ms) | not repeated | +0.85 ms |
| R9700 decode 32k | 67.0 | 63.2 (15.83 ms) | not repeated | +0.9 ms |
| R9700 4 streams | 214.2 | 195 (20.49 ms) | about 200 (A/B, own prompts, 300 tokens) | prompts differ from `c4_steady.py` |
| R9700 prefill 2k | 2075 (2454 tokens, `perf.py`) | 2284 (2040 tokens, max_tokens 1) | 2257 (A/B mean, 2040 tokens) | see below |
| R9700 prefill 8k / 32k | 1920 / 1401 | 2013 / 1433 | 1965 (A/B mean, 9.1k) | |
| 8060S decode short | 38.6 | 36.7 (27.27 ms) | not repeated | +1.4 ms |
| 8060S prefill 2k / 8k / 32k | 823 / 786 / 595 | 894 / 799 / 598 | 899 (A/B, 2040 tokens) | |

Decode agrees with the baseline once the profiling overhead is added. Prefill differs by method, measured on one server (R9700, `recon` session): `perf.py` method (max_tokens 4, time to the first visible token, 2454 tokens) 2063 tok/s; the same prompt with max_tokens 1 and total time 2128 (the three extra decode steps are 3 percent); a 2040 token prompt (4 full chunks) 2215 (the 406 token last chunk of the 2454 prompt costs a full set of weight reads: 3 percent more). So the baseline `perf.py` rows are about 6 to 7 percent below the pure prefill rate; they stay the rows to beat because they are what the card reports. `PERF_BASELINE.md` rows are not changed.

## 2. Bucket tables (a)

Buckets: experts (gate_up, down, routes, combine), proj+oproj (split), attention+prep (attention tile and merge, attn_prep, conv_commit, kv_append), small (mix, ctrl, norm, post conv, router), head+sampler (embed, head select and norm, lm_head, sampler). `launches` per step or chunk are the same on both GPUs. `gap` = measured minus floor. `share` of the device busy. Decode rows are ms per step, prefill rows ms per 512 token chunk (profiled wall: the profiling overhead is inside the busy values, about 0.65 ms per step on the decode rows, about 1 percent of a prefill chunk). Per-family rows (launches per step, us per launch, floors) are in `bench/<gpu>/profile/summary.md`.

Launch structure per decode step (1,569 dispatches, identical for every row count): per visit 32.7 on average: proj gemv 5 (+ 64 per step split_bf16 for sink and dynamic conv), oproj 1, norm 3.2, mix 3.1, post conv 4, attn_prep 3, attention 3 (two tile launches and one merge), controllers 3 on 24 + 24 visits, router 2, experts routes, gate_up, down, combine 4; plus embed, lm_head, head select and norm, sampler, prep_spans once per step.

**decode short** (ms per step; profiled wall R9700 12.59 ms (79 tok/s), 8060S 27.27 ms (37 tok/s))

| bucket | launches | R9700 ms | floor | gap | share % | 8060S ms | floor | gap | share % |
|---|---|---|---|---|---|---|---|---|---|
| experts | 192 | 3.69 | 2.98 | +0.71 | 32 | 9.84 | 8.03 | +1.80 | 37 |
| proj+oproj | 352 | 2.42 | 2.22 | +0.20 | 21 | 6.99 | 5.98 | +1.01 | 26 |
| attention+prep | 288 | 1.03 | 0.01 | +1.02 | 9 | 1.68 | 0.04 | +1.65 | 6 |
| small: mix, ctrl, norm, conv, router | 731 | 3.13 | 0.49 | +2.64 | 27 | 5.31 | 1.32 | +4.00 | 20 |
| head+sampler | 6 | 1.38 | 0.79 | +0.59 | 12 | 2.61 | 2.14 | +0.47 | 10 |
| total busy | | 11.66 | 6.49 | +5.17 | 100 | 26.43 | 17.51 | +8.93 | 100 |

**decode 8k** (ms per step; profiled wall R9700 13.95 ms (72 tok/s), 8060S 31.22 ms (32 tok/s))

| bucket | launches | R9700 ms | floor | gap | share % | 8060S ms | floor | gap | share % |
|---|---|---|---|---|---|---|---|---|---|
| experts | 192 | 3.80 | 2.98 | +0.82 | 29 | 9.80 | 8.03 | +1.77 | 32 |
| proj+oproj | 352 | 2.41 | 2.22 | +0.19 | 18 | 7.03 | 5.98 | +1.05 | 23 |
| attention+prep | 288 | 2.31 | 0.75 | +1.56 | 18 | 5.54 | 2.03 | +3.52 | 18 |
| small: mix, ctrl, norm, conv, router | 731 | 3.16 | 0.49 | +2.67 | 24 | 5.52 | 1.32 | +4.20 | 18 |
| head+sampler | 6 | 1.41 | 0.79 | +0.61 | 11 | 2.63 | 2.14 | +0.49 | 9 |
| total busy | | 13.10 | 7.23 | +5.87 | 100 | 30.52 | 19.50 | +11.02 | 100 |

**decode 32k** (ms per step; profiled wall R9700 15.83 ms (63 tok/s), 8060S 37.29 ms (27 tok/s))

| bucket | launches | R9700 ms | floor | gap | share % | 8060S ms | floor | gap | share % |
|---|---|---|---|---|---|---|---|---|---|
| experts | 192 | 3.97 | 2.98 | +0.99 | 25 | 10.22 | 8.03 | +2.19 | 27 |
| proj+oproj | 352 | 2.60 | 2.22 | +0.38 | 16 | 7.14 | 5.98 | +1.15 | 19 |
| attention+prep | 288 | 4.69 | 2.23 | +2.47 | 29 | 12.87 | 6.00 | +6.87 | 34 |
| small: mix, ctrl, norm, conv, router | 731 | 3.26 | 0.49 | +2.77 | 20 | 5.35 | 1.32 | +4.04 | 14 |
| head+sampler | 6 | 1.41 | 0.79 | +0.62 | 9 | 2.63 | 2.14 | +0.49 | 7 |
| total busy | | 15.94 | 8.71 | +7.23 | 100 | 38.21 | 23.47 | +14.74 | 100 |

**4 rows decode** (ms per step; profiled wall R9700 20.49 ms (195 tok/s), 8060S 49.68 ms (81 tok/s))

| bucket | launches | R9700 ms | floor | gap | share % | 8060S ms | floor | gap | share % |
|---|---|---|---|---|---|---|---|---|---|
| experts | 192 | 9.79 | 2.98 | +6.81 | 51 | 28.43 | 8.03 | +20.39 | 59 |
| proj+oproj | 352 | 2.77 | 2.22 | +0.55 | 14 | 7.82 | 5.98 | +1.84 | 16 |
| attention+prep | 288 | 1.50 | 0.00 | +1.50 | 8 | 3.01 | 0.00 | +3.01 | 6 |
| small: mix, ctrl, norm, conv, router | 731 | 3.74 | 0.49 | +3.25 | 19 | 5.94 | 1.32 | +4.62 | 12 |
| head+sampler | 6 | 1.40 | 0.79 | +0.61 | 7 | 2.62 | 2.14 | +0.48 | 5 |
| total busy | | 19.21 | 6.48 | +12.73 | 100 | 47.82 | 17.47 | +30.35 | 100 |

**prefill 2k** (ms per 512-token chunk; R9700 2284 tok/s, 8060S 894 tok/s profiled)

| bucket | launches | R9700 ms | floor | gap | share % | 8060S ms | floor | gap | share % |
|---|---|---|---|---|---|---|---|---|---|
| experts | 192 | 124.98 | 47.67 | +77.31 | 57 | 352.89 | 128.51 | +224.38 | 62 |
| proj+oproj | 352 | 24.92 | 10.99 | +13.92 | 11 | 58.20 | 43.97 | +14.24 | 10 |
| attention+prep | 288 | 34.35 | 2.04 | +32.31 | 16 | 85.21 | 8.15 | +77.06 | 15 |
| small: mix, ctrl, norm, conv, router | 731 | 33.95 | 0.62 | +33.33 | 15 | 71.33 | 2.46 | +68.86 | 13 |
| head+sampler | 6 | 1.38 | 0.20 | +1.18 | 1 | 2.51 | 0.54 | +1.97 | 0 |
| total busy | | 219.58 | 61.52 | +158.06 | 100 | 570.14 | 183.63 | +386.52 | 100 |

**prefill 8k** (ms per 512-token chunk; R9700 2013 tok/s, 8060S 799 tok/s profiled)

| bucket | launches | R9700 ms | floor | gap | share % | 8060S ms | floor | gap | share % |
|---|---|---|---|---|---|---|---|---|---|
| experts | 192 | 124.02 | 47.67 | +76.34 | 50 | 354.02 | 128.51 | +225.51 | 56 |
| proj+oproj | 352 | 24.50 | 10.99 | +13.51 | 10 | 57.73 | 43.97 | +13.76 | 9 |
| attention+prep | 288 | 63.74 | 3.87 | +59.86 | 26 | 146.73 | 15.50 | +131.23 | 23 |
| small: mix, ctrl, norm, conv, router | 731 | 32.87 | 0.62 | +32.24 | 13 | 69.18 | 2.46 | +66.72 | 11 |
| head+sampler | 6 | 1.30 | 0.04 | +1.26 | 1 | 2.42 | 0.12 | +2.31 | 0 |
| total busy | | 246.42 | 63.21 | +183.22 | 100 | 630.08 | 190.56 | +439.52 | 100 |

**prefill 32k** (ms per 512-token chunk; R9700 1433 tok/s, 8060S 598 tok/s profiled)

| bucket | launches | R9700 ms | floor | gap | share % | 8060S ms | floor | gap | share % |
|---|---|---|---|---|---|---|---|---|---|
| experts | 192 | 122.54 | 47.67 | +74.87 | 35 | 350.85 | 128.51 | +222.34 | 41 |
| proj+oproj | 352 | 25.05 | 10.99 | +14.06 | 7 | 57.84 | 43.97 | +13.87 | 7 |
| attention+prep | 288 | 167.50 | 9.78 | +157.71 | 48 | 364.66 | 39.12 | +325.54 | 43 |
| small: mix, ctrl, norm, conv, router | 731 | 33.11 | 0.62 | +32.49 | 9 | 70.86 | 2.46 | +68.40 | 8 |
| head+sampler | 6 | 1.27 | 0.01 | +1.26 | 0 | 2.38 | 0.03 | +2.35 | 0 |
| total busy | | 349.47 | 69.08 | +280.40 | 100 | 846.59 | 214.09 | +632.50 | 100 |

Idle gap plus host (profiled wall minus device busy): R9700 decode short 0.93 ms (7 percent), 8k 0.85, 32k 0.95, 4 rows 1.28; 8060S 0.84 (3 percent), 0.70, 1.08, 1.86. Prefill: 1.9 percent (2k), 1.7 (8k), 1.2 (32k) on the R9700 and 0.3 to 0.1 percent on the 8060S. Part of the decode values is the profiling overhead (1.0 and 1.4 ms per step), so the real idle gap is 0 to 1.0 ms on the R9700 and 0 to 0.5 ms on the 8060S. A host-side issue cost cannot be told apart from it; the host does not wait on the device in decode (spin wait, PERF_BASELINE "Wait mode": no effect).

### 2.1 What the tables say

Decode, R9700 short context (11.66 ms busy; floor 6.49):

| Where the 5.17 ms of gap sits | ms | Launches | Note |
|---|---|---|---|
| mix (mix_states) | 1.57 | 147 | 10.7 us per launch; benchmark 4.7 to 6.1 us; floor 0 |
| attention at short context + attn_prep | 1.03 | 288 | tile 6.8 us x 96, merge 2.5 x 48, prep 2.2 x 48, conv_commit 1.4 x 48, kv_append 1.6 x 48 |
| experts (gate_up 49.3 us at 78 percent of the bandwidth, down 25.4 us at 76 percent, routes, combine) | 0.71 | 192 | |
| ctrl (down, z, apply) | 0.52 | 144 | floor 0.37 (bytes) |
| router (logits 5.6 us, select 3.9 us) | 0.46 | 96 | floor 0.09 |
| head (lm_head at 414 GB/s) + sampler (195 us) | 0.59 | 6 | sampler alone is 1.7 percent of the step |
| norm (rms_norm 2.2 us x 152) | 0.34 | 152 | |
| post conv (conv_res 1.5, conv_commit 1.2 us) | 0.24 | 192 | |
| proj + oproj | 0.20 | 352 | at the roof |

Decode, long context: attention is the only bucket that moves: 1.03 ms (short), 2.31 (9k), 4.69 ms (37k) on the R9700 and 1.68, 5.54, 12.87 ms on the 8060S, against KV floors 0.01, 0.75, 2.23 and 0.04, 2.03, 6.00 ms. At 37k the attention gap alone (2.5 ms, 2.2 for the tile kernel) is 15 percent of the step: the 67 against 118 tok/s of the card is 2.2 ms attention + 1.0 ms experts + 1.6 ms mix + the rest of the short-context gaps.

Prefill, R9700 2k (219.6 ms per chunk; the floor sum is 61.5 ms): experts 125.0 (gate_up 76.1 ms = 1.59 ms per visit against 1.17 ms in the isolated benchmark of `KERNELS_experts.md`, down 45.6 ms; 27 TFLOP/s and 240 GB/s; floors 47.7 ms memory and 26 ms compute), attention+prep 34.4, small 33.9 (ctrl 14.7, mix 6.7, norm 5.4, router 4.9, post conv 2.3), proj+oproj 24.9 (53 TFLOP/s against 120 possible). The prefill small kernels sum to 15 percent of the chunk with floors near zero: ctrl alone is 14.7 ms (144 launches, 102 us each, floor 0.37 ms of bytes).

Prefill, attention versus context (R9700 ms per chunk): 30.3 (2k, mean context 1k), 59.2 (9k), 162.9 (37k); slope about 8 us per context token, intercept about 22 ms (the 40 local visits with the window 1024). 8060S: 70.5, 132.2, 349.9 ms.

## 3. Lanes (b), ranked

Gains are estimates from the profile (share of the row x the fraction a design can remove, per `topic/optimize-loop`, "Pricing a design"); the measured kernel-to-model conversion of fusions was 50 to 100 percent of the kernel gain in the skills' campaigns, so the ranges include that. Row = the card row the lane is judged on. ms are R9700 unless named. "Files" are the exclusive write set of the lane (all under `package/`); lanes with disjoint sets can run in parallel.

Verification common to every lane (V0): kernel tests of the area (`package/test/<area>` through `scripts/ktest_adapter.py`, baseline 28 of 28 on both GPUs), a negative control for any A/B kernel test (`ab_kernel.py`), `loom-opt --pass=canonicalize` on all programs, then A/B end to end against a sha base (V1), then `LEVEL=dev` gates once on the winner (V2: `scripts/run_gates.sh <r9700|strix> serve gates extra stop` under `flock -s`; consistency, agent, golden, compare, sampling supplement; context table check if device memory moves). V1 (ABBA): `ab_perf.py` needs both models in one server; the R9700 cannot hold two 17.6 GB weight sets, so use session ABBA: `scripts/ab_abba.sh <gpu> <variant> <metrics> <reps>` (A B B A, one fresh server per arm, hard-link variant tree from `scripts/ab_make_variants.sh`; `ab_session.py` metrics dshort, d8k, d32k, pf2k, pf8k, c4), keep only if the target row wins by more than the noise (decode: 1.5 percent, observed spread 0.1 to 0.7; c4: 2.5; prefill: 4, observed session spread 3) and no other row loses more than 2 to 3 percent. On the 8060S (128 GB unified) `ab_perf.py` with two models in one server at context 8192 is possible. Run `perf.py` with a label for the kept build and write `bench/decisions.jsonl`. Numerics rule: a lane that keeps the arithmetic order (pure launch merging, same per-element math) must give byte-identical golden outputs (smoke sha of `scripts/smoke_publish.sh`, `compare` gap unchanged); a lane that reorders sums (attention, experts, norm fusion with a different reduction tree) is gated before its speed counts and may flip near ties (the skills' `lvs` GEMM failed consistency this way).

| # | Lane | Target row and metric | Expected gain | Risk | Files owned | Verification beyond V0 to V2 | Time box | Stop rule |
|---|---|---|---|---|---|---|---|---|
| 1 | P1 Prefill larger-n experts pass (n 1024 and 2048) and `prefill_capacity` | prefill 1k, 2k, 8k tok/s, both GPUs; first on the R9700 | measured model: +20 percent at 1024, +34 percent at 2048 with today's kernel efficiency (R9700; 8060S +16 / +26); more with better tiles (lane 2) | medium to high: routes kernel (n <= 512, tasks record stride 34 words, tile <= 32), WMMA tile for 64 to 128 routes per expert, activation memory, shape buckets | `kernel/experts/{routes,gate_up_wmma,down_wmma}.loom`, `lib/inkling/experts_{common,stage,wmma}.loom`, `dispatch/hooks_experts.loom`, `prepare.loom` (buckets), `package/card.json` + `scripts/publish_build.py` (`prefill_capacity`) | experts oracle tests with n 1024 and 2048 and skewed routing (`package/test/experts`), consistency gate (chunking invariance: a prompt alone against with prefix), 1k row must not lose (padding at short prompts: skills' flint note: 1k lost 29 percent before padding fixes), ctx table re-run (memory +0.5 to 1 GB), compile time | 8 h | at 2048 tokens the experts reach 90 percent of their memory floor or a compute floor at 60 percent of 120 TFLOP/s; or a measured reason (e.g. 1k row loses more than 3 percent and no padding fix) |
| 2 | P2 Experts GEMM efficiency at fixed n (tile 32 path) | prefill rows; ms per chunk of experts_gate_up and down | 125 to about 85 ms per chunk (-18 percent at 2k; 27 to 40 TFLOP/s) | high: skills' MoE note says the dequant, stage and barrier per k step bound it, not the MMA; no native fp4 WMMA on gfx12, the UE4M3 sub-block scale prevents the fp8 route used for MXFP4 | `kernel/experts/{gate_up_wmma,down_wmma}.loom`, `lib/inkling/{nvfp4,experts_wmma}.loom` (collides with lane 1: same agent, after it) | as lane 1 plus ISA read first (`loom-compile --format=amdgpu-hsaco`) and `ab_kernel.py` against the old kernel with real routing | 6 h | 3 restructurings within 3 percent, or at 60 percent of the compute floor |
| 3 | D1 Decode small-kernel fusion: residual chain (mix_attn, mix_ffn, norm_in, norm_ffn, conv_res, conv_commit) | decode short and 8k, 4 streams, both GPUs | -1.0 ms per step (R9700, 9 percent, 86 to 94 tok/s); -1.5 ms (8060S, 6 percent) | medium: same arithmetic, new fusion boundaries; the mix kernel is 2x off its benchmark: first find why (cache, grid, nb 7 tail) | `kernel/mix/{mix_states,conv_res}.loom`, `kernel/dense/norm.loom`, `dispatch/hooks_mix.loom` (non-ctrl hooks), norm hooks of `dispatch/hooks_dense.loom`, `dispatch/layer.loom` | byte-identical goldens; `package/test/mix`, `test/dense` norm; guards (no write beyond live tokens) | 4 h | the 147 mix + 152 norm + 192 conv launches reach 1.3 us per remaining launch, or after 2 designs each within 2 percent |
| 4 | D2 Decode attention front and projection group: one launch for q, k, v, gate, rel GEMVs (shared xn input), fuse attn_prep + kv_append + conv_commit, merge tile and merge launches | decode short, 8k | -0.7 ms (R9700, 6 percent) | medium: consumer-prologue designs lost 28 to 250 percent in skills' GEMV tests; a grouped GEMV over distinct weight matrices keeps the byte stream | `kernel/dense/{gemv,gemv_split}.loom`, new `gemv5` kernel, `kernel/attn/{attn_prep,attn_kv_append,attn_conv_commit,attn_merge}.loom`, `hooks_attn.loom` prep part, proj/oproj hooks of `hooks_dense.loom`, `lib/inkling/dense_vec.loom` (collides with lane 3 on `layer.loom` and `hooks_dense.loom`: land lane 3 first or split the hook files) | as lane 3; proj bandwidth must stay at 90 percent of the bandwidth | 4 h | grouped GEMV below 90 percent of the bandwidth after 2 designs |
| 5 | D3 Decode controllers and router: ctrl_down + ctrl_z + ctrl_apply into one or two launches, router logits + select + routes in one, combine into the post-ffn conv | decode short, 4 streams | -0.55 ms (R9700, 5 percent) | medium: controllers have F32 weights (227 MB); router select order must stay exact (tie rule: lower index) | `kernel/mix/ctrl.loom`, ctrl hooks of `hooks_mix.loom`, `kernel/experts/{router,routes,combine}.loom` (decode path), `hooks_experts.loom` router part (collides with lane 1 on `hooks_experts.loom`: serialize) | experts and mix tests, golden byte-identical (router ties) | 3 h | launches per visit reach 1 per family or 2 designs within 2 percent |
| 6 | D4 Decode attention at long context (global visits) | decode 8k and 32k, both GPUs | R9700 -1.5 ms at 37k (+10 percent: 63 to 70 tok/s unprofiled 67 to 75), -0.8 ms at 9k (+6 percent); 8060S -3 ms at 37k (+9 percent) | medium: split rows interact with `max_splits` workspace and the merge; bitwise chunk invariance holds for prefill only | `kernel/attn/attn_tile.loom` (decode mode), `attn_merge.loom`, split rows in `hooks_attn.loom` | attn tests (decode tail cases), golden at 2k/16k, A/B at 8k and 32k with prompts of the card; context table unchanged | 3 h | tile kernel at 85 percent of the KV bandwidth (floor 2.2 ms at 37k) or splits within noise (see section 6) |
| 7 | P3 Prefill attention kernel (GQA K/V tile shared by the 3 heads of a KV group, larger query tile, local window skip) | prefill 8k, 16k, 32k, both GPUs; attention ms per chunk | R9700 -18 / -35 / -100 ms per chunk at 2k / 8k / 32k (+9 / +16 / +41 percent tok/s); 8060S -40 / -80 / -200 ms | high: the kernel is at 7 to 9 TFLOP/s with 16 query rows per workgroup and one head per workgroup (`KERNELS_attn.md`); the softmax order must stay chunk invariant (bitwise on gfx1201 today) | `kernel/attn/attn_tile.loom` (prefill mode; same file as lane 6: one owner or lane 6 first), `lib/inkling/attn_{common,matrix}.loom` | attn oracle tests including 512+rest, chunk-invariance cases, golden 16k, consistency; `ab_kernel.py` with serving-shaped spans (several spans, decode anchors in the same epoch) | 6 h | kernel at 25 TFLOP/s equivalent (3.5x), or 3 designs within 3 percent |
| 8 | D5 Decode head and sampler: lm_head GEMV 414 to above 540 GB/s, sampler 195 us to below 40 us (parallel argmax and sampling over the 200,058 logits), head select and norm folded | decode short, all decode rows, R9700 first (8060S head is at 91 percent, sampler 0.26 ms) | R9700 -0.45 ms (4 percent); 8060S -0.25 ms (1 percent) | low to medium: the sampler must keep the same draws (sampler tests, seed reproducibility of `sampling_supplement.py`) | `kernel/dense/head.loom`, `kernel/dense/gemv.loom` (head variant only), `lib/vendor/sampling.loom` use, head hook of `hooks_dense.loom` | sampler tests, sampling supplement 4 of 4, golden | 2 h | lm_head at 90 percent of the bandwidth and sampler under 2 percent of the step |
| 9 | D6 Experts decode (gate_up and down NVFP4 and Q8_0 dense arm) | decode short, 8k | R9700 -0.45 ms (4 percent): gate_up 49.3 to 43 us, down 25.4 to 21.5 us (90 percent); 8060S -0.6 ms | medium: the decode kernel is the PR1-derived `gate_up.loom`/`down.loom`; 4 dense-arm visits at Q8_0 | `kernel/experts/{gate_up,down}.loom`, `lib/inkling/experts_stage.loom` (decode half) | experts tests (5 cases), golden | 3 h | both kernels at 90 percent of the measured bandwidth |
| 10 | P4 Prefill small kernels and controllers (ctrl 14.7 ms, mix 6.7, norm 5.4, router 4.9, attn_prep 4.1, combine 2.6, post conv 2.3 per chunk) | prefill rows | -20 ms per chunk (-9 percent at 2k; R9700), 8060S -35 ms | low to medium: prefill-shaped (ctrl_tokens 16, chunk 128), per-arch tune rows already exist in `kernel/mix/tune.json` | the same files as lanes 3 and 5: run after them, as a separate pass with its own tune rows | as lane 3 plus the prefill goldens | 3 h | ctrl at 3 ms and the others at 2x their byte floor |
| 11 | P5 Prefill proj and oproj GEMM (Q8_0 weights) | prefill rows | R9700 -8 to -10 ms per chunk (-4 percent): 25 ms against a 11 ms flop floor, 53 TFLOP/s; 8060S none (76 percent of the best rate) | medium | `kernel/dense/gemm_q8_0.loom`, `gemm_wide_q8_0` selection thresholds in `kernel/dense/tune.json` (`gemm_wide_min_tiles` 32, `gemm_wide_max_work` 524288) | dense tests, golden | 2 h | 90 TFLOP/s equivalent or 3 variants within 3 percent |
| 12 | Session variance of prefill (3 percent) | prefill A/B noise | none by itself; makes lane 1, 2, 7, 10, 11 decisions safer | low | none (measurement only: log sclk, power, `mem_info_*` at start of each session) | 6 sessions with clocks logged | 1 h | cause found or documented as per-session memory placement |

Order of value (R9700 rows, estimates): decode short 11.64 ms: lanes 3, 4, 5 (about -2.2 ms together), 8 (-0.45), 9 (-0.45) = -3.1 ms at most (8.5 ms = 117 tok/s); a realistic capture of 60 to 70 percent gives 9.7 ms = 103 tok/s. Prefill 2k (220 ms per chunk): lanes 1 and 2 together -70 ms (experts 125 to 55), lane 7 -18, lane 10 -20, lane 11 -8: -116 ms at most (104 ms = 4.9k tok/s); realistic 140 ms = 3.6k tok/s.

## 4. Targets (c)

There are no published Inkling numbers on any device (the only reference is the CPU build of llama.cpp, 14.9 tok/s decode and 58 tok/s prefill: an oracle, not a competitor). The targets are fractions of the roofline ceilings of `ROOFLINE.md` (decode 154.3 / 57.2 tok/s, prefill 9.8k / 3.6k tok/s at the measured bandwidth), the skills' defaults (decode 0.78, prefill 0.33 of peak) being unmeasured on this model. Proposal:

| Row | R9700 now (efficiency) | R9700 target | stretch | 8060S now | 8060S target | stretch | Why |
|---|---|---|---|---|---|---|---|
| decode short | 85.9 (0.56) | 105 (0.68) | 117 (0.76) | 38.6 (0.68) | 43 (0.75) | 45 (0.79) | the weight streams are at 67 to 97 percent of the bandwidth already; the 39 percent small-kernel share can lose 55 to 65 percent; 75 percent of the ceiling needs every lane at its upper bound on the R9700, but is reachable on the 8060S where the gap is smaller |
| decode 8k | 76.3 (0.55) | 92 (0.66) | 104 (0.75) | 33.2 (0.64) | 38 (0.73) | 41 | as above plus the long-context attention lane |
| decode 32k | 67.0 (0.57) | 83 (0.70) | 92 (0.78) | 27.5 (0.63) | 33 (0.75) | 35 | attention gap is the largest single item here (2.2 / 6.5 ms) |
| 4 streams | 214 (0.36 shared bound, 0.79 independent bound) | 235 (0.86 of the independent bound) | 250 | 86.6 | 95 | 100 | experts already at the traffic floor of the real routing; only the small-kernel savings (1.5 ms of 18.7) apply |
| prefill 2k | 2075 (0.21) | 3300 (0.34) | 4900 (0.50) | 823 (0.23) | 1250 (0.34) | 1500 (0.41) | 50 percent needs lanes 1 and 2 at their upper bound (experts at 2048 tokens near a 60 percent of GEMM rate); only physically plausible on the R9700 at n >= 1024 (experts memory floor 12 ms per 512 tokens at n = 2048); on the 8060S prefill is compute bound at 30 TFLOP/s real (floor sum 184 ms per chunk = 2.8k tok/s) |
| prefill 8k | 1920 (0.20) | 2900 (0.30) | 4000 | 786 (0.22) | 1100 | 1300 | adds the attention lane (-35 ms) |
| prefill 32k | 1401 (0.14) | 2400 (0.25) | 3200 | 595 (0.16) | 900 | 1100 | attention is 48 percent of the chunk; 3.5x kernel speedup needed for the target |

The perf.py rows (max_tokens 4, 2454 tokens) read about 6 percent below the pure prefill rate (section 1.4); the table uses perf.py values as the card does.

## 5. Order and parallelism (d)

Timed runs are exclusive across the whole box (one lock, both GPUs): about 10 minutes per ABBA candidate (4 sessions) and 5 minutes per profile pair, roughly 20 to 25 timed candidates in total for the plan, 4 to 5 hours of exclusive time. CPU work (kernel writing, `loom-opt`, compile reports, ISA reading, `loom-test` on CPU) does not need the lock; kernel tests and gates take `flock -s`.

1. Wave 1 (parallel, disjoint files): lane 1 (experts files, `prepare.loom`, card) on the R9700; lane 3 (mix, norm, conv, layer.loom) ; lane 6 (attn_tile decode, hooks_attn split rows: the split rows quick test is done, section 6); lane 8 (head, sampler). Lane 1 first on the GPU: it changes the prefill shapes and every later prefill measurement.
2. Wave 2: lane 4 and 5 after lane 3 has landed (shared `layer.loom`, `hooks_dense.loom`, `hooks_mix.loom`); lane 9 (experts decode) in parallel with lane 2 only if lane 9 touches `gate_up.loom`/`down.loom` and lane 2 the `*_wmma.loom` files; lane 7 after lane 6 (same `attn_tile.loom`) or by the same agent.
3. Wave 3: lanes 10 and 11 after lanes 3 to 5 (same files, prefill tune rows); re-profile with `profile_all.sh`; re-run `perf.py` rows and rebuild the card (T6/T8).
4. After each wave: profile again (the new bucket table is the next budget), V2 gates on the kept build only.
5. The two GPUs: develop each lane's kernel with per-arch rows for both; time the R9700 first (largest gap in launch count terms for decode, only GPU where lane 2 and 11 pay), then confirm on the 8060S once per kept change (its lane gains are comparable or larger in ms: small kernels cost 1.7x more per launch there). The 8060S is also the cheaper place for two-model interleaved `ab_perf.py`.
6. Disk: variants are hard-link trees (`scripts/ab_make_variants.sh`, no weight copy); delete them after the decision.

## 6. Quick wins without kernel changes (measured)

Method: scratch variant trees of the fetched package (hard links, one file edited, nothing committed), `scripts/ab_abba.sh` (A B B A, fresh server per arm, spin URI, GPU lock), `scripts/ab_session.py` metrics. R9700 unless named.

| Knob (edit) | Arm A (base) | Arm B (variant) | Result |
|---|---|---|---|
| `xs_wg_per_unit` default 2 -> 1 (`lib/inkling/dense_vec.loom`; gfx1201 uses the portable default) | decode short 86.06, 85.92 tok/s; 4 rows 198.7, 200.9 | 85.47, 85.41; 200.2, 201.1 | -0.6 percent decode (spread 0.1), 4 rows +0.5 percent: tie, keep 2 |
| same -> 3 | 85.16, 85.75; 200.5, 199.6 | 85.72, 85.70; 201.2, 200.7 | +0.3 percent: tie, keep 2 |
| same -> 4 | 85.74, 85.94; 199.1, 199.3 | 60.29, 60.23; 199.1, 199.1 | decode -29.9 percent, 4 rows equal: the GEMV path of one row is sensitive to the workgroup count; keep 2 |
| global attention `split_blocks` x2 (4 / 8 / 16 blocks at up to 4k / 16k / above; `dispatch/hooks_attn.loom`, gfx1201 provider) | decode 8k 76.40, 76.88; 37k 67.50, 67.85 | 76.73, 76.93; 68.57, 68.58 | +0.25 percent at 9k, +1.3 percent at 37k (two sessions each, spread 0.3): a small real gain, below the 2 percent rule; the A arms drift 0.5 percent between sessions. Candidate for lane 6 |
| global attention `split_blocks` x4 (8 / 16 / 32) | 76.36, 76.52; 67.45, 67.59 | 73.01, 72.59; 66.16, 65.70 | 8k -4.8 percent (72.80 against 76.44), 37k -2.4 percent (65.93 against 67.52): too coarse loses; the optimum is near the present rows or 2x |
| `prefill_capacity` 512 -> 256 (preset) | pf2k 2292.0, 2222.6; pf9k 1992.8, 1938.0 tok/s | 1691.9, 1681.5; 1506.5, 1495.3 | -25 percent (2k), -24 percent (9k): per-chunk weight traffic dominates; model fixed 77 ms + 0.29 ms per token (R9700) |
| same, 8060S | 899.4, 904.0; 804.0, 801.1 | 706.7, 706.7; 634.8, 635.4 | -21.5 percent (2k), -21 percent (9k): fixed 154 ms + 0.81 ms per token |

Chunk model check: R9700 `chunk time(n) = 77 + 0.29 n` ms gives 3.0k tok/s at n = 2048 and 2.7k at 1024 (A mean 2257 at 512); the 9k rows give fixed 80 ms and 0.35 ms per token. The model assumes the present kernel efficiency; a larger tile changes the per-token term in both directions, so the estimate is for planning only.

Decode noise seen: within a session 0.1 to 0.3 percent; between fresh servers of the same build 0.1 to 0.7 percent (decode), 3 percent (prefill), 1 to 1.5 percent (4 streams).

Not run, with the reason: gemv cols (`inkling.dense.gemv.cols`, 8: decode uses the 1 row bucket where it is not read), expert tile (the portable path tile 8 and matrix tile 32 are fixed by the kernels: tile 32 is the limit of `routes`, that is lane 1), spans (span_cap is bucket data, 4 rows), chunk above 512 (the routes kernel limit n <= 512 stops it: lane 1), wait mode (already measured in PERF_BASELINE: no effect).

## 7. Open items

- A per-hook split of the merged kernels (mix_states, gemv_q8_0, rms_norm) by a profile build with per-hook aliases would replace the apportionment; it needs a scratch program copy and is a small, cheap lane-0 task before lane 3 and 4.
- The mix kernel runs at 10.7 us per launch against 4.7 to 6.1 us in isolation: cold caches (weights of 24 different layers evict the mix inputs), nb up to 7 blocks, or the dependency wait counted in the duration; a one-day question that decides lane 3's size.
- Idle gap: 0 to 1.0 ms per step cannot be fixed from the HAL table; a profile of the same build without `executable-metadata` (dispatch events only) would show whether the overhead is per dispatch.
- The 8060S power mode is still not recorded (floors use 230 GB/s and 30 TFLOP/s).
