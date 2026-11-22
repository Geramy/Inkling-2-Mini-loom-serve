# KERNELS_attn: attention prep, KV append, attention (inkE)

Status: ABI v1, all kernels pass the oracle tests on gfx1201 and gfx1151 (2026-11-21). Files: `package/kernel/attn/*.loom`, `package/lib/inkling/attn_*.loom`, tests in `package/test/attn/`. Math: ARCHITECTURE.md sections 3-5, 8, 9. All activations F32 unless stated. nh = 24, nk = 8, hd = 128, group = 3.

## Conventions

- Token-packed, one launch set per epoch shape. Grid sizes come from capacities (launch scalars). The live token count and the spans come from the span table, so workgroups of dead tokens or spans return at once.
- Span table (buffer `spans`, i32), the same table as `KERNELS_mix.md`: header `{span_count, live_tokens, 0, 0}`, then 8 words per span `{first_token, length, read_lo, read_hi, write_lo, write_hi, position, row}`. Words 6 and 7 are the absolute position of the span's first token and the resident row (the extension requested from inkH). The attn kernels use only the `write` offset (`write_lo`, `write_hi`): the absolute BYTE offset of the row's recurrent slot START inside `state` (the arena). A fork copies the slot, so read = write. The buffer must hold `4 + 8 * span_cap` words; `span_cap <= 256`, token capacity `cap <= 32768`, positions below 2^31.
- Spans of one epoch cover the packed tokens contiguously and belong to distinct rows. A span of length 1 uses the decode path, a longer span uses the prefill path (draft spans of 2 to 4 tokens take the prefill path).
- Activations are token-major `[T][width]`. Projection outputs (inkF) go straight into the buffers below.

## Storage (all inside `state`, the arena)

| Region | Address | Layout |
|---|---|---|
| Global KV, visit gv = 0..7 | `kv_origin + page * 262144`, `kv_origin = gv * pool_tokens * 4096` (bytes) | `page` = physical page id from the host page map. A page holds 64 tokens: `[kv head 8][K, V][64 tokens][128] F16`: head h K at `h*32768`, V at `h*32768 + 16384`, token j at `+ j*256`. |
| Local ring, local visit lv = 0..39 | `write + kv_origin + ((pos / 64) % 16) * 262144`, `kv_origin = lv * 4194304` | Same page layout. The ring holds 16 pages = 1024 tokens; position p is in page `(p/64) % 16`, slot `p % 64`. |
| Conv state, visit v = 0..47 | `write + conv_origin`, `conv_origin = 167772160 + v * 79872` | 3 x 6656 F32, `state[row * 6656 + channel]`, row 0 oldest (x[t-3]). k channels 0..1023, v 1024..2047 (post-attn 2048.., post-ffn 4352.. are inkG's). Zero at sequence start. The state holds the pre-conv inputs. |

The page layout is head-major inside a page, so one K or V tile of 64 tokens for one head is 16 KB contiguous. The page size (262144 B = 64 x 4096) agrees with `PACKAGE_PLAN.md` section 2.

Page map: buffer `rows` (library layout). Row record = 5 x i64 at byte `row * 40`; only field 4 is used: the byte offset inside `rows` of an i32 array `map[logical block] = physical page id` (multiple of 4). `max_blocks` is the length of that array (context capacity / 64). Local visits do not read the map.

## Order of the launches in one visit

```
inkling_attn_prep        q_raw, k_raw, v_raw, r_raw (+ dyn_k, dyn_v on global) -> q, r (F32), k_act, v_act (F16)
inkling_attn_tile mode 0 spans with length > 1  (prefill)
inkling_attn_tile mode 1 spans with length == 1 (decode, split KV)
inkling_attn_merge       combines the decode splits, sink, gate
inkling_attn_conv_commit new k/v conv rows into the slot   (after prep; needs k_raw, v_raw alive)
inkling_attn_kv_append   K/V of the chunk into pages or ring (after both tile launches; needs k_act, v_act alive)
```
Both tile launches run on every visit; a workgroup that does not own a span of its kind returns at once. `kv_append` writes the ring only after attention has read it. Keys of the current chunk are always read from `k_act`/`v_act` (F16, the same rounding as the cache), history below the span position from the cache. So the result does not depend on how a prompt is cut into chunks, and a chunk longer than 1024 tokens never needs the ring for its own keys. A local span writes only its last 1024 tokens to the ring.

## Kernels and ABI (Loom export names; launch args in ABI order; `[..]` is the workload)

| Kernel | Workload, grid, workgroup | Launch args |
|---|---|---|
| `inkling_attn_prep[cap]` | grid `(cap, 43, 1)`, 128 threads | `cap, span_cap, global, conv_origin: offset, spans, state, q_raw, k_raw, v_raw, r_raw, dyn_k, dyn_v, w_q_clip, w_rel_clip, w_k_conv, w_v_conv, q, r, k_act, v_act` |
| `inkling_attn_tile[gx, gy, gz]` | prefill `(tile_bound, 24, 1)`, decode `(span_cap, 8, max_splits)`; 4 x subgroup threads | `cap, span_cap, mode, global, split_blocks, max_splits, max_blocks, kv_origin: offset, spans, rows, state, q, r, k_act, v_act, rel_bias, sink, gate, out, partial` |
| `inkling_attn_merge[span_cap]` | grid `(span_cap, 24, 1)`, 128 threads | `cap, span_cap, global, split_blocks, max_splits, spans, sink, gate, out, partial` |
| `inkling_attn_conv_commit[span_cap]` | grid `(span_cap, 8, 1)`, 256 threads | `span_cap, cap, conv_origin: offset, spans, state, k_raw, v_raw` |
| `inkling_attn_kv_append[cap]` | grid `(cap, 1, 1)`, 128 threads | `cap, span_cap, global, max_blocks, kv_origin: offset, spans, rows, state, k_act, v_act` |

Buffers (F32 unless stated; `T` = token capacity):

| Buffer | Shape | Content |
|---|---|---|
| `q_raw`, `k_raw`, `v_raw`, `r_raw` | `[T][3072]`, `[T][1024]`, `[T][1024]`, `[T][384]` | projection outputs (`attn_q`, `attn_k`, `attn_v`, `attn_rel`) before conv and clip |
| `dyn_k`, `dyn_v` | `[T][32]` | `attn_k_conv_proj`, `attn_v_conv_proj` outputs (global visits; ignored on local visits). Tap `lag + 4*head` |
| `w_q_clip`, `w_rel_clip`, `w_k_conv`, `w_v_conv`, `rel_bias` | `attn_q_clip.weight` (24 x 128), `attn_rel_clip.weight` (24 x 16), `attn_k_conv.weight` (1024 x 4), `attn_v_conv.weight` (1024 x 4), `attn_rel_bias.weight` (1024 x 16) | the GGUF tensors of the visit's physical layer as they are (F32) |
| `q`, `r` | `[T][3072]`, `[T][384]` | after L2 clip with scale, and tau on global visits (the oracle `Qcur`, `Rcur`) |
| `k_act`, `v_act` | `[T][1024]` F16 | after conv, clip (k) or rms norm (v) (the oracle `Kcur`, `Vcur` rounded to F16) |
| `sink` | `[T][24]` | raw `attn_sink_proj` output |
| `gate` | `[T][3072]` | raw `attn_gate` output (logits); the kernels apply sigmoid |
| `out` | `[T][3072]` | gated attention output, the input of `attn_output` |
| `partial` | `span_cap * 24 * max_splits * 136` | decode workspace: per (span, head, split) `{m, l, acc[128], 2 pad}` |

Scalars: `global` 1 on global visits, `mode` 0 prefill / 1 decode, `split_blocks` = 64-key blocks per decode split, `max_splits` = grid z of the decode launch and stride of `partial`. The host computes `max_splits = ceil(max_visible_blocks / split_blocks)`: local visits `17`, global visits `context_capacity / 64 + 1`. `tile_bound = sum over spans of ceil(length / 16) + span_cap` (or `ceil(T / 16) + span_cap`).

Composition requirements for inkH:
- `package/lib/inkling/attn_targets.loom` binds `amdgpu.target<gfx11-generic> @inkling_attn_rdna3` (selects the replicated RDNA3 matrix operand). The composition must contain that line once (the library keeps the same line in its config).
- `package/lib/inkling/attn_matrix.loom` is adapted from `hrx-loom-library/motif/tensor/matrix.loom` (MIT). Vendored-file line for `VENDORED.md`: `package/lib/inkling/attn_matrix.loom <- motif/tensor/matrix.loom (renamed symbols @inkling.attn.mma.*)`.
- Link all of `attn_common.loom`, `attn_matrix.loom`, `attn_targets.loom` with the kernel files.
- Workgroup scratch: tile 33,024 B LDS (R9700 and 8060S); tested with wave32 only (the code reads `target.subgroup.size`; wave64 is untested).
- Alignment: activation buffers 16 B, arena offsets of pages multiples of 256 KiB.

Determinism: block boundaries are absolute multiples of 64 positions and every row sees its blocks in the same order, so prefill results are bitwise equal for any chunking on gfx1201 (up to 6e-8 on gfx1151 for local visits, see the table) and equal to the single chunk result within F16 noise for the decode path. The decode split layout depends only on the token position and `split_blocks`, not on the batch.

## Findings about the reference (for the other agents)

- `norm-S` in the oracle dumps is the RMS_NORM output BEFORE the weight multiply. On global visits the three occurrences in order are the k input, the v input, the q input, then the FFN norm (not q, k, v). The inputs of the dynamic conv projections are `norm * attn_{k,v}_inp_norm`, rounded to BF16 as the CPU dot does.
- The attention logits are `q.k / sqrt(128) + bias` and the sink logit is the raw `attn_sink_proj` output (ARCHITECTURE.md 5.7 is right; a first reading of `build_attn_mha` suggests a scaled bias, the numbers say it is not: NMSE 5e-8 against `kqv_out`). `kq_b` (bias minus sink) at an empty cache cell gives `-sink`, which the tests use to check the sink projection (max abs 3e-6).
- Dynamic conv taps: `dyn[t][lag + 4*h]` multiplies `x[t - lag]` of channels `h*128 .. h*128+127`; the divisor is 3 (dynamic) or sqrt(5) (static).

## Config knobs and tune rows (data, no arch strings)

| key | meaning | gfx1201 | gfx1151 | portable default |
|---|---|---|---|---|
| `inkling.attn.split_blocks` | 64-key blocks per decode split | local 1; global 2 up to 4096 keys, 4 up to 16384, 8 above | same | 4 |

(Measured sweep below; not tuned further, M8 owns that.) The kernels have no other knobs; tile rows (16), key block (64) and the 4 subgroups are fixed.

## Test table (oracle comparison, NMSE; see `package/test/attn/attn_test.py`)

Run on the box: `python package/test/attn/attn_test.py run --dev gfx1201|gfx1151` (loom-test runs the real kernels, outputs written with `check.file.write.npy`). Tolerances: q, r 1e-6; k, v (F16) 2e-7; out 1e-6. A = oracle prompt A (16 tokens), L = the 1332 token prompt (`oracle/dumps_long`, B x4 + 100 tokens). "chunks" lists the span cuts; "decode" = prefill then one-token spans; "mixed" = five epochs mixing a prefill span of one row and decode spans of another row.

| case | q | r | k (F16) | v (F16) | out gfx1201 | out gfx1151 | result |
|---|---|---|---|---|---|---|---|
| A slot 0 1chunk | 0.0e+00 | 3.7e-15 | 4.5e-08 | 4.5e-08 | 2.6e-08 | 2.6e-08 | PASS |
| A slot 0 2chunks | 0.0e+00 | 3.7e-15 | 4.5e-08 | 4.5e-08 | 2.6e-08 | 2.6e-08 | PASS |
| A slot 0 decode | 0.0e+00 | 3.7e-15 | 4.5e-08 | 4.5e-08 | 2.5e-08 | 2.5e-08 | PASS |
| A slot 5 1chunk | 1.3e-15 | 3.2e-15 | 4.3e-08 | 4.1e-08 | 3.1e-08 | 3.1e-08 | PASS |
| A slot 5 2chunks | 1.3e-15 | 3.2e-15 | 4.3e-08 | 4.1e-08 | 3.1e-08 | 3.1e-08 | PASS |
| A slot 5 decode | 1.3e-15 | 3.2e-15 | 4.3e-08 | 4.1e-08 | 3.1e-08 | 3.1e-08 | PASS |
| L slot 0 1chunk | 0.0e+00 | 3.7e-15 | 4.3e-08 | 4.7e-08 | 1.3e-08 | 1.3e-08 | PASS |
| L slot 0 3chunks | 0.0e+00 | 3.7e-15 | 4.3e-08 | 4.7e-08 | 1.3e-08 | 1.3e-08 | PASS |
| L slot 0 512+rest | 0.0e+00 | 3.7e-15 | 4.3e-08 | 4.7e-08 | 1.3e-08 | 1.3e-08 | PASS |
| L slot 0 decode | 0.0e+00 | 3.7e-15 | 4.3e-08 | 4.7e-08 | 1.3e-08 | 1.3e-08 | PASS |
| L slot 0 mixed | 0.0e+00 | 3.7e-15 | 4.3e-08 | 4.7e-08 | 1.5e-08 | 1.5e-08 | PASS |
| L slot 5 1chunk | 1.6e-15 | 3.2e-15 | 4.3e-08 | 4.3e-08 | 1.7e-08 | 1.7e-08 | PASS |
| L slot 5 3chunks | 1.6e-15 | 3.2e-15 | 4.3e-08 | 4.3e-08 | 1.7e-08 | 1.7e-08 | PASS |
| L slot 5 512+rest | 1.6e-15 | 3.2e-15 | 4.3e-08 | 4.3e-08 | 1.7e-08 | 1.7e-08 | PASS |
| L slot 5 decode | 1.6e-15 | 3.2e-15 | 4.3e-08 | 4.3e-08 | 1.7e-08 | 1.7e-08 | PASS |
| L slot 5 mixed | 1.7e-15 | 3.2e-15 | 4.3e-08 | 4.3e-08 | 2.0e-08 | 2.0e-08 | PASS |
| L slot 6 1chunk | 0.0e+00 | 3.3e-15 | 4.3e-08 | 5.0e-08 | 1.1e-08 | 1.1e-08 | PASS |
| L slot 6 3chunks | 0.0e+00 | 3.3e-15 | 4.3e-08 | 5.0e-08 | 1.1e-08 | 1.1e-08 | PASS |
| L slot 6 512+rest | 0.0e+00 | 3.3e-15 | 4.3e-08 | 5.0e-08 | 1.1e-08 | 1.1e-08 | PASS |
| L slot 6 decode | 0.0e+00 | 3.3e-15 | 4.3e-08 | 5.0e-08 | 1.1e-08 | 1.1e-08 | PASS |
| L slot 6 mixed | 0.0e+00 | 3.4e-15 | 4.3e-08 | 5.0e-08 | 1.2e-08 | 1.2e-08 | PASS |
| L slot 11 1chunk | 1.7e-15 | 3.4e-15 | 4.3e-08 | 4.3e-08 | 1.7e-07 | 1.7e-07 | PASS |
| L slot 11 3chunks | 1.7e-15 | 3.4e-15 | 4.3e-08 | 4.3e-08 | 1.7e-07 | 1.7e-07 | PASS |
| L slot 11 512+rest | 1.7e-15 | 3.4e-15 | 4.3e-08 | 4.3e-08 | 1.7e-07 | 1.7e-07 | PASS |
| L slot 11 decode | 1.7e-15 | 3.4e-15 | 4.3e-08 | 4.3e-08 | 1.7e-07 | 1.7e-07 | PASS |
| L slot 11 mixed | 1.8e-15 | 3.7e-15 | 4.3e-08 | 4.3e-08 | 2.3e-07 | 2.3e-07 | PASS |

Chunk invariance of our results (max abs difference of `out`, single chunk of 1332 tokens versus the cut; local and global visits):
- gfx1201: 3 chunks (444 each) and 512+512+308: bitwise equal for visits 0, 5, 6, 11. Decode tail (1300 prefill + 32 one-token spans, `split_blocks` 2): max abs 5e-4 to 1.3e-3 against the single chunk (F16 probability rounding differs between the prefill and the split decode path; NMSE against the oracle is unchanged at 1e-8).
- gfx1151: same, except 3 chunks on the local visits 0 and 6 differ by 1.1e-8 and 6e-8 (not bitwise).
- The oracle itself (llama.cpp CPU) is bitwise chunk invariant: the 3 chunk run (`oracle/dumps_long/Q8_K_XL_L3`, `ORACLE_CHUNK=444`) gives `kqv_out` NMSE 0 and max abs 0 against the single chunk run for visits 0, 5, 6, 11 (`oracle/tools/cmp_chunks.py`).

## Per-kernel timing (loom-benchmark, dispatch_complete, GPU locked, random data)

| run | R9700 us | floor us | x floor | 8060S us | floor us | x floor |
|---|---|---|---|---|---|---|
| dec_g_1024_b1 | 23.34 | 6.77 | 3.5 | 36.05 | 18.24 | 2.0 |
| dec_g_1024_b2 | 31.00 | 6.77 | 4.6 | 38.48 | 18.24 | 2.1 |
| dec_g_1024_b4 | 29.84 | 6.77 | 4.4 | 38.34 | 18.24 | 2.1 |
| dec_g_1024_b8 | 52.70 | 6.77 | 7.8 | 71.04 | 18.24 | 3.9 |
| dec_g_1024_b16 | 84.55 | 6.77 | 12.5 | 120.55 | 18.24 | 6.6 |
| dec_g_4096_b1 | 54.82 | 27.06 | 2.0 | 105.86 | 72.94 | 1.5 |
| dec_g_4096_b2 | 55.11 | 27.06 | 2.0 | 97.94 | 72.94 | 1.3 |
| dec_g_4096_b4 | 54.71 | 27.06 | 2.0 | 105.60 | 72.94 | 1.4 |
| dec_g_4096_b8 | 92.94 | 27.06 | 3.4 | 127.62 | 72.94 | 1.7 |
| dec_g_4096_b16 | 92.18 | 27.06 | 3.4 | 130.46 | 72.94 | 1.8 |
| dec_g_16384_b1 | 206.16 | 108.24 | 1.9 | 652.35 | 291.78 | 2.2 |
| dec_g_16384_b2 | 169.07 | 108.24 | 1.6 | 616.83 | 291.78 | 2.1 |
| dec_g_16384_b4 | 157.43 | 108.24 | 1.5 | 568.82 | 291.78 | 1.9 |
| dec_g_16384_b8 | 179.80 | 108.24 | 1.7 | 606.10 | 291.78 | 2.1 |
| dec_g_16384_b16 | 184.82 | 108.24 | 1.7 | 720.78 | 291.78 | 2.5 |
| dec_g_32768_b1 | 410.81 | 216.48 | 1.9 | 1456.42 | 583.56 | 2.5 |
| dec_g_32768_b2 | 364.18 | 216.48 | 1.7 | 1372.06 | 583.56 | 2.4 |
| dec_g_32768_b4 | 353.28 | 216.48 | 1.6 | 1334.94 | 583.56 | 2.3 |
| dec_g_32768_b8 | 348.03 | 216.48 | 1.6 | 1129.10 | 583.56 | 1.9 |
| dec_g_32768_b16 | 402.64 | 216.48 | 1.9 | 1204.64 | 583.56 | 2.1 |
| dec_l_1024_b1 | 25.94 | 6.77 | 3.8 | 34.32 | 18.24 | 1.9 |
| dec_l_1024_b2 | 30.07 | 6.77 | 4.4 | 38.12 | 18.24 | 2.1 |
| dec_l_1024_b4 | 49.08 | 6.77 | 7.3 | 37.33 | 18.24 | 2.0 |
| dec_l_1024_b8 | 53.61 | 6.77 | 7.9 | 68.29 | 18.24 | 3.7 |
| merge_g_32768 | 23.35 | 1.37 | 17.1 | 24.51 | 3.69 | 6.6 |
| merge_l | 3.68 | 0.06 | 58.2 | 3.06 | 0.17 | 18.0 |
| pre_l_512_at0 | 215.13 |  |  | 434.63 |  |  |
| pre_l_512_at1024 | 682.77 |  |  | 1533.77 |  |  |
| pre_g_512_at0 | 218.66 |  |  | 461.50 |  |  |
| pre_g_512_at4096 | 3111.14 |  |  | 6630.89 |  |  |
| prep_1 | 2.95 | 0.06 | 45.5 | 2.23 | 0.17 | 12.8 |
| prep_512 | 47.22 | 33.19 | 1.4 | 109.96 | 89.47 | 1.2 |
| commit_1 | 2.44 | 0.09 | 26.4 | 1.32 | 0.25 | 5.3 |
| commit_512 | 2.44 | 0.12 | 20.5 | 1.27 | 0.32 | 4.0 |
| append_g_1 | 2.77 | 0.01 | 209.9 | 1.80 | 0.04 | 50.7 |
| append_g_512 | 31.87 | 6.77 | 4.7 | 6.69 | 18.24 | 0.4 |
| append_l_512 | 31.88 | 6.77 | 4.7 | 6.62 | 18.24 | 0.4 |

Names: `dec_g_<ctx>_b<n>` global decode tile launch at `ctx` keys with `split_blocks = n`; `dec_l_1024` local decode (full window); `merge_*` merge launch; `pre_<l|g>_512_at<pos>` prefill tile launch of 512 tokens starting at position `pos` (no floor column: compute bound, see below); `prep_n`, `commit_n`, `append_*_n` for n tokens. Floors: tile = `4096 B x keys / bandwidth`; prep = activation bytes read plus written; append = K/V bytes written; bandwidth 620 GB/s (R9700) and 230 GB/s (8060S), the measured values of the brief. Every row is one launch, mean of a serialized batch of 32 with `loom-benchmark --measure=dispatch_complete`, GPU locked, 1 warm-up batch; working sets of 4 MB and below stay in cache (so "x floor" below 1 on the 8060S is cache, not DRAM).

Prefill compute: 512 tokens at position 0 (local, 131k query-key pairs per head) 1.6 GFLOP in 215 us = 7.5 TFLOP/s on the R9700 (435 us, 3.7 TFLOP/s on the 8060S); 512 tokens at 1024 keys per row (6.4 GFLOP) 9.4 TFLOP/s (4.2); global at 4096 history (27 GFLOP) 8.8 TFLOP/s (4.1). I did not measure the matrix peak, so no peak fraction is claimed.

Cost of idle workgroups: decode grid z sized for 32k keys (`max_splits` 128 at `split_blocks` 4) on a 1024 key row: 43 us instead of 30 us (R9700), 49 us instead of 38 us (8060S); at 4096 keys no difference. Rule for inkH: size `max_splits` from the context bucket of the epoch shape, not from the largest context, when buckets allow.

Per decode token the five attention launches of a local visit cost about 3 + 25 + 4 + 2.4 + 2.7 = 37 us (R9700), so 40 local visits are 1.5 ms of the 6.5 ms weight floor; the 8 global visits add 0.5 ms at 4k context and 3 ms at 32k.

## Open problems

- Decode is latency bound: one tile launch costs 25 us (R9700) / 34 us (8060S) at 1k keys against a 6.8 / 18 us bandwidth floor; a decode visit is about 5 launches. M8: fuse prep + conv commit + append for decode, merge by the last workgroup, fewer blocks per barrier round.
- Prefill reaches 7 to 9 TFLOP/s (R9700) with 16 query rows per workgroup and one head per workgroup; the three heads of a KV group could share one K/V tile.
- `kv_append` at 512 tokens takes 32 us on the R9700 (6.7 us on the 8060S, where the 4 MB write stays in cache).
- gfx1151 local-visit prefill is not bitwise chunk invariant (max abs 6e-8); gfx1201 is. Cause not found.

## Changes by inkJ (integration, 2026-11-20)

- `kernel/attn/attn_merge.loom`: `lo_shift` is computed as `select(pos > 1023, pos - 1023, 0)` instead of `index.max` then `index.sub`. Reason: `BACKEND/044` scheduler failure (`s_max_i32` to `s_sub_u32` to `s_cselect_b32` state cycle) for the local specialisation under loom-serve. Same result. `attn_tile.loom` has the same expression and compiles, so it is unchanged.
- `func.def` of `kernel/attn/attn_tile.loom` are `public`.
- The oracle `kqv_out` is the attention output BEFORE the gate multiply; the kernels multiply with `sigmoid(gate)`, so compare `kqv_out * sigmoid(gate)` (the tests of this doc compared with the gate input of the test fixture).
- Dispatch aliasing: `sink` lives in the shared `small` buffer, `dyn_k` in `ob`, `dyn_v` in `y`, `k_act` in `xn`, `v_act` in `x` (INTEGRATION_LOG.md). `max_splits` and the `partial` size come from the shape bucket context (`inkling.context`), `split_blocks` from `@inkling.tune.attn.split_blocks` (rows for gfx1201 and gfx1151 in `dispatch/hooks_attn.loom`).
