# KERNELS_mix: attention-residual mix, loop message and controller, post convs (inkG)

Status: ABI v1 final (2026-11-21), all kernels pass on gfx1201 and gfx1151. Files: `package/kernel/mix/{mix_states,ctrl,conv_res}.loom`, tune rows `package/kernel/mix/tune.json`, tests `package/test/mix/`. Math: ARCHITECTURE.md sections 5.3, 5.6, 5.8, 6.2. All activations F32. E = 2304. Launch args are listed in ABI order; the first args repeat the workload scalars of `kernel.def`.

## Conventions

- Span table (buffer `spans`, i32) is the shared table of `package/dispatch/spans.loom`: header `{span_count, live_tokens, 0, 0}` at byte 0, then 8 words per span at byte 16: `{first_token, length, read_lo, read_hi, write_lo, write_hi, position, row}`. `read`/`write` are absolute BYTE offsets of the row's recurrent slot START in the arena `state`. At most 63 spans per launch. Kernels return at once for `token >= live_tokens`, so grids use the capacity `cap` of the shape bucket.
- Activations `[T][E]` row-major (token stride E). A multi-column output (nq columns) is `[nq][cap][E]`, column stride `col_stride` elements (`cap * E`).
- `blocks[b]` is `[cap][E]` at element offset `b * block_stride` (`block_stride = cap * E`), b < 8. `cur` is `[cap][E]`.

## Visit schedule of my kernels (per token, one layer visit)

```
mix_states   attn mix (mode 0 raw + rinv, or mode 1 with rms_norm * weight)       -> in_q[,in_k,in_v] columns
[passes 1, 2 on layers 6..17:]
mix_states   message mix (nb 1, nq 4, mode 0)                                      -> m[4] columns (+ rinv_m)
ctrl_down -> ctrl_z -> ctrl_apply -> mix_norm   (nc 3: Q, K, V; weights attn_norm / q,k,v_inp_norm)
 ... inkE / inkF kernels (projections, attention, o-proj) ...
conv_res (site attn, push at ip % 6 == 0) + conv_commit
mix_states   ffn mix (mode 1 with ffn_norm, or mode 0 + ctrl_down -> ctrl_z -> ctrl_apply (nc 1) -> mix_norm)
 ... router and experts (inkD / inkF) ...
conv_res (site ffn) + conv_commit
head:  mix_states with sel list, output_res_query, mode 1, norm_w = output_norm   -> rms_norm(mix) * output_norm for the selected tokens
```
Layer 0 attention: `mix_states` with `nb = 0` (the query is not read, `cur` passes through; mode 1 gives `rms_norm(cur) * attn_norm`).

## Kernels

### `inkling_mix_states[cap]` (288 threads, one workgroup per token)

`launch(cap, nb, nq, block_stride, col_stride, mode, w_stride, sel_on, spans, sel, blocks, cur, query, norm_w, out, rinv_out)`
- States are `blocks[0..nb)` and `cur` (S = nb + 1 <= 9, `nb <= 8`). Order is irrelevant for the math. `nq` 1..4. `query` is F32 `[nq][E]` (`attn_res_query` {2304} or {2304,3}, `ffn_res_query`, `loop_msg_query` {2304,4}, `output_res_query`).
- `mode 0`: `out` = raw mix result, `rinv_out[q][t]` (`[4][cap]`) = 1/sqrt(mean(out^2) + 1e-6) for the controller.
- `mode 1`: `out` = `rms_norm(mix) * norm_w[q * w_stride + e]` (`w_stride 0` shares one weight, `E` = one weight row per column: q/k/v_inp_norm on global layers).
- `nb == 0`: out_q = cur for every column (spec 5.4 layer 0).
- `sel_on 1` (head): `sel` is i32 `{n_sel, token_0, token_1, ...}`; workgroup k mixes the states of token `sel[1 + k]` and writes compact row k of `out` (`out[q][k]`), for `k < n_sel`. With `sel_on 0` pass any valid buffer of at least 16 bytes.
- Message mix: `nb = 1`, `blocks` = bank (bank state 0 = the p12 block of the previous pass), `cur` = bank_out (the previous pass output), `nq = 4`, mode 0.

### Controller `inkling_ctrl_down[cap, nc]`, `inkling_ctrl_z[cap, nc]`, `inkling_ctrl_apply[cap, nc]`, `inkling_mix_norm[cap, nc]`

Controller c computes `x' = x - tanh(forget z) x + tanh(gain z) m` with `z = down [rms(x); rms(m)]`. `nc` = 3 (Q, K, V) or 1 (FFN; the weight set is passed as set 0). Weights are one raw tensor per matrix (`loop_{q,k,v,ffn}_ctrl_{down,forget,gain}.weight`): F32 (Q4_K_XL) or BF16 (Q8_K_XL, raw 16-bit words), selected by config `inkling.mix.ctrl_bf16`. Unused sets (nc < 3) may repeat set 0.
- `ctrl_down`: `launch(cap, nc, x_col0, x_col_step, m_col0, col_stride, spans, x, m, down0, down1, down2, part)`. The `x` column of controller c is `x_col0 + c * x_col_step` (local layers: `x_col_step 0`, all three read column 0; global layers: 1); the `m` column is `m_col0 + c` (attention 0, FFN 3). Writes `part[c][cap][4608/chunk][64]` F32 partial sums.
- `ctrl_z`: `launch(cap, nc, x_col0, x_col_step, m_col0, spans, rinv_x, rinv_m, part, z)`: `rinv_x[col][cap]`, `rinv_m[col][cap]` are the `rinv_out` of the two `mix_states` launches. Writes `z[c][cap][64]`.
- `ctrl_apply`: `launch(cap, nc, x_col0, x_col_step, m_col0, col_stride, spans, x, m, forget0, forget1, forget2, gain0, gain1, gain2, z, out)`. `out` is `[nc][cap][E]` (unnormalized x'), must not alias `x` or `m`.
- `mix_norm`: `launch(cap, nc, col_stride, w_stride, spans, x, w, out)`: `out[c] = rms_norm(x[c]) * w[c * w_stride]` (in place allowed; also the plain input norm for layer 0).
- Workspace: `part` = `nc * cap * (4608 / chunk) * 64 * 4` B (chunk 256: 18 KB per token per controller), `z` = `nc * cap * 256` B, `out` = `nc * cap * E * 4` B.
- Config, set per shape bucket with `config_i64`: `inkling.mix.ctrl_chunk` (K per `ctrl_down` workgroup, multiple of 64, up to 512), `inkling.mix.ctrl_tokens` (tokens per workgroup of `ctrl_down` and `ctrl_apply`; the weights stay in registers over the token tile; 1 for decode buckets, 16 for prefill buckets), `inkling.mix.ctrl_bf16`.

### Convs `inkling_conv_res[cap]`, `inkling_conv_commit[span_cap]`

Post-attention conv (site 2048) and post-FFN conv (site 4352): `y = (x_t + sum_{i<4} w[c][i] * xcat[t + i]) / sqrt(5)`, tap 0 oldest, `xcat` = the 3 stored inputs then the span's inputs. `conv_w` = `blk.N.attn_post_conv.weight` / `ffn_post_conv.weight` raw F32 `[2304][4]`.
- `conv_res`: `launch(cap, site, visit, push, push_block, block_stride, bank_on, bank_origin, spans, state, a, conv_w, cur, blocks, bank)`. `a` = pre-conv input `[cap][E]` (o-proj output or MoE output, not modified). `site` = channel offset inside a state row. `visit` = BYTE offset of this visit's conv state from the slot start = `167772160 + v * 79872` (inkE storage table). `push 1` (attention at `ip % 6 == 0`): `blocks[push_block][t] = cur[t]` (the residual), then `cur[t] = y`; `push 0`: `cur[t] += y` (the FFN site always uses 0). `bank_on 1`: also write to `bank[bank_origin + t * E + c]` the stored residual (when `push 1`) or the new `cur` (when `push 0`).
- `conv_commit`: `launch(span_cap, site, visit, spans, state, a)`, one launch per site right after `conv_res`. Writes the new 3 history rows of each live span (from `a` and the old rows) into the span's `write` slot, reading the old rows from the `read` slot. It is a separate launch because tokens of one span read old rows while other tokens replace them; this makes the update exact for any span length (also length < 3 and decode), independent of chunking, and correct for read slot != write slot (fork).
- State layout (agreed with inkE, KERNELS_attn.md "Storage"): 3 rows x 6656 F32 per visit, `state[row * 6656 + channel]`, row 0 oldest; channels k 0..1023, v 1024..2047, post-attn 2048..4351, post-ffn 4352..6655. The pre-conv inputs are stored. Zero at sequence start.

### Block bookkeeping and the loop bank (what the dispatch does)

- Block pushes: visit ip 0 (`push_block 0`), ip 6 (`push_block 1`), ip 12 (`push_block 2`); coda ip 18 pushes `push_block 3` after the last pass. The attention mix at a push visit uses `nb = push_block` blocks (the residual is not stored yet); the FFN mix after the push uses `nb = push_block + 1`. Visit 0 stores x0 as block 0 (`cur` holds x0 on entry).
- Passes 1 and 2 restart from the prelude output: `blocks[0]` (x0) stays, `cur` = the prelude output (the dispatch keeps a copy), blocks 1 and 2 are overwritten by the new pass; the coda uses the blocks of pass 2.
- Bank for the message mix of pass p + 1 (`S = 2`): state 0 = the block pushed at ip 12 of pass p, state 1 = the post-FFN `cur` of ip 17 of pass p. Keep a bank array `bank[set][2][cap][E]` with `set` = pass parity: the attention `conv_res` of ip 12 runs with `bank_on 1, bank_origin = (set * 2) * cap * E` (stores the residual = p12), the FFN `conv_res` of ip 17 runs with `bank_on 1, bank_origin = (set * 2 + 1) * cap * E`. The message mix of the next pass: `nb 1`, `blocks` = `bank[set]`, `cur` = `bank[set] + cap * E` (set 0 for pass 1, set 1 for pass 2; pass 2 writes no bank).

## Tune rows (`package/kernel/mix/tune.json`, data not code)

| key | gfx1201 decode | gfx1201 prefill | gfx1151 decode | gfx1151 prefill | portable default |
|---|---|---|---|---|---|
| `inkling.mix.ctrl_chunk` | 256 | 128 | 256 | 128 | 128 |
| `inkling.mix.ctrl_tokens` | 1 | 16 | 1 | 16 | 1 decode, 16 prefill |

`inkling.mix.ctrl_bf16` follows the checkpoint (0 Q4_K_XL, 1 Q8_K_XL). The rows come from a three-point check (chunk 64/128/256, tokens 8/16/32): differences are at most 10% on both GPUs, so these are defaults, not a tune (M8). Workgroup shapes are fixed (288, 256, 192, 64 threads); the reductions use 32-lane clusters (`cluster_size`), which work on wave32 and wave64.

## Tests (numbers are measured; `package/test/mix/run_all.sh <gpu>` on the box, logs in `~/inkling/inkG/logs`)

Reference: `inkg.py` (numpy, f64). It reproduces the oracle tensors of prompt A before any kernel is involved (`check_ref.py`: attn mix, ffn mix and message mix of all 48 slots NMSE <= 1e-14; controllers vs dumped `norm-S` 1e-6 .. 2e-5 (the oracle rounds activations to BF16 for BF16 weights); post convs and `l_out` of all slots exact). Dump facts used: dumped `norm-S` are the unweighted rms norms, `x0 = norm * token_embd_norm`, `norm-S` occurrences of the three attention inputs of a loop layer are ordered (k, v, q), the message mix of slot X is dumped as `mix_out-(X-1)` occurrence 2 (slot 24: `mix_out-17`, slot 36: `mix_out-35`). No new oracle tensor names were needed.

| Test (both GPUs, identical results) | cases | max NMSE | result |
|---|---|---|---|
| mix_states attn mix, prompt A, 16 slots incl. global nq 3 and all block counts, vs oracle `mix_out` | 16 | 1.5e-14 | PASS |
| mix_states rinv (rms of the mix) vs oracle | 16 | 1.1e-14 | PASS |
| mix_states ffn mix + ffn_norm, 17 slots vs oracle | 17 | 2.1e-14 | PASS |
| message mix S = 2, nq = 4, slots 24..47 vs oracle (+ rinv) | 48 | 1.2e-14 | PASS |
| head mix + output_norm, last token via sel list vs oracle `result_norm`, prompts A, B, C | 3 | 1.7e-14 | PASS |
| head with 4 selected tokens (unordered) vs numpy | 1 | 6.5e-15 | PASS |
| prompt B (T = 308) and C (T = 25): attn mix of 47 slots and 24 message mixes vs numpy (the reference matches the oracle on A) | 142 | 1.3e-14 | PASS |
| synthetic: nb 0..8 (S up to 9), nq 1/3/4, mode 0 and 1 with per-column weights, T = 700 | 37 | 2.9e-14 | PASS |
| guards: no write beyond live tokens or sel rows (mix, ctrl, conv) | 91 | 0 | PASS |
| controller (Q8_K_XL BF16 weights), 24 slots x (Q, K, V, FFN) vs numpy on the same weights, chunk 128 token tile 1 and chunk 256 token tile 16 | 2 x 120 | 5.5e-14 | PASS |
| controller + norm vs oracle `norm-S` (attention Q/K/V and FFN) | 2 x 96 | 2.4e-5 (oracle BF16 activations) | PASS (< 1e-4) |
| controller F32 weights (Q4_K_XL file) slots 24, 29, 41, 47, chunk 64/128/256, tokens 8, vs numpy | 36 | 7.1e-14 | PASS |
| post-attn conv_res vs oracle `ffn_inp` (all 48 slots, incl. block store and bank copy) | 48 + 56 | 0 (bitwise equal) | PASS |
| post-ffn conv_res vs oracle `l_out` (all 48 slots) and state commit vs numpy | 48 + 48 | 0 | PASS |
| conv: spans of length 1..7, 17, 40 in place; 4 spans in 4 slots; fork (read slot != write slot); empty history | 12 | 3.8e-15 (y), 0 (state) | PASS |
| conv: chunking 40 tokens as [13,27], [1x6,34], [2,1,3,4,30], [5x8], [39,1] equals one launch; prefill 30 + 6 decodes equals prefill 36 | 6 | 0 (bitwise) | PASS |

Counts: `test_mix.py` 299, `test_conv.py` 342, `test_ctrl.py` 240 per config (2 configs) + 12 x 3 (Q4 variants), per GPU, all PASS. B and C only dump `l_out`, `result_norm` and `result_output` (mode light), so the convs and controllers cannot be checked against B and C directly; B (T = 308) and C (T = 25) cover mixes and the head, and the chunking tests cover long spans (40 tokens, chunks 1..39).

## Timing (`bench_mix.py`, `loom-benchmark --measure=dispatch_complete --batch-size=100/10`, `flock -x`, p50 per dispatch)

A launch of an empty kernel costs 1.1 us (R9700) and 0.9 us (8060S) in the same batched setup (the first row of each block includes this floor). The benchmark replays the same buffers, so data may stay in L2/MALL: ratios below 1 against the DRAM floor are cache hits, not faster than memory. Floor = bytes / measured bandwidth (R9700 620 GB/s, 8060S 230 GB/s).

Decode (T = 1), one launch each, microseconds:

| Kernel | gfx1201 | gfx1151 | traffic (R9700 / 8060S floor) |
|---|---|---|---|
| mix_states nb 3, nq 1 | 4.7 | 5.7 | 0.05 MB (0.07 / 0.2 us) |
| mix_states nb 3, nq 3 (global attn) | 6.0 | 6.6 | 0.06 MB |
| mix_states nb 7, nq 1 | 6.1 | 7.2 | 0.08 MB |
| message mix S 2, nq 4 | 5.8 | 6.3 | 0.06 MB |
| controller Q, K, V (nc 3), F32 weights: down + z + apply + norm | 2.3 + 1.2 + 2.2 + 1.5 = 7.2 | 4.3 + 1.1 + 3.1 + 1.5 = 10.0 | 7.3 MB (11.8 / 31.7 us) |
| controller Q, K, V, BF16 weights | 6.9 | 10.5 | 3.8 MB (6.1 / 16.4 us) |
| controller FFN (nc 1), F32 | 6.6 | 8.6 | 2.4 MB (3.9 / 10.6 us) |
| conv_res attn / ffn | 1.6 / 1.5 | 1.5 / 1.5 | tiny |
| conv_commit | 1.4 | 1.4 | tiny |

Per decoded token (24 looped visits on passes 1, 2): controllers 24 x (7.2 + 6.6) = 0.33 ms (R9700, F32 weights; 227 MB of weights at 620 GB/s would be 0.37 ms, so the fused controller runs at the memory floor within cache effects), all mixes (96 launches at about 5 us) 0.5 ms, convs 96 launches 0.15 ms: together about 1 ms of the 6.5 ms floor, mostly launch and latency chains (these are small dependent kernels, 3 to 8 us each). Open item for M8: merge `mix_states` into the following norm or controller, and `conv_commit` into the next visit's launch, to remove about 0.3 ms.

Prefill (T = 512), microseconds:

| Kernel | gfx1201 | gfx1151 | traffic |
|---|---|---|---|
| mix_states nb 3, nq 1 / nq 3 | 32 / 38 | 70 / 87 | 23.6 / 33.0 MB |
| mix_states nb 7, nq 1 | 47 | 197 | 42.5 MB |
| controller nc 3, F32: down 110, z 10, apply 297, norm 22 | 439 | 1304 | 120 MB |
| controller nc 3, BF16 | 428 | 1196 | 117 MB |
| controller nc 1 (FFN) | 187 | 312 | 40 MB |
| conv_res attn / ffn | 27 / 25 | 44 / 41 | 38 / 33 MB |
| conv_commit | 6.5 | 2.0 | tiny |

`ctrl_apply` is 4x its floor at T = 512 (one lane of each 8-lane cluster does the element-wise part); a token-interleaved variant is the first M8 item. With the token tile the weights are read once per 16 tokens: the first version without tiling took 2977 us at nc 3.

Compile reports: no register spills in any export on gfx1201 (gfx12-generic) and gfx1151 (gfx11-generic).

## Reproduce

```
. ~/inkling/inkG/.venv/bin/activate          # hrx-loom 0.2.0, numpy, gguf
cd ~/inkling/inkG/package/test/mix
python check_ref.py                           # numpy reference vs oracle (no GPU)
./run_all.sh gfx1201 ; ./run_all.sh gfx1151   # all kernel tests, shared GPU lock
CHUNK=256 TOK=16 python bench_mix.py gfx1201  # timings, exclusive GPU lock
```
Sync from the laptop with `package/test/mix/sync.sh`. Tests use `loom-test` with `check.file.read.npy` fixtures and compare the written outputs in numpy (`harness.py`).

## Changes by inkJ (integration, 2026-11-20)

- `kernel/mix/mix_states.loom`: `inkling_mix_state` (selected a buffer between `cur` and `blocks`) is replaced by `inkling_mix_load`, which loads the vector from `cur` or from `blocks` in two branches. Reason: backend scheduler failure `BACKEND/044` (`s_cselect_b32` state cycle) when the launch scalars (`nb`, `nq`) are exact constants under loom-serve. Results unchanged (visits 0 to 5 match the oracle).
- `func.def` of `kernel/mix/*.loom` are `public` (cross-file inline calls in the serve compile).
- Launch ABI unchanged. The dispatch calls `mix_states` only for visits with `nb >= 1` (visit 0 has no mix; `norm_in` reads `cur`), mode 0 (raw), `rinv_out` and `sel` use the `small` and `xn` buffers (see INTEGRATION_LOG.md aliasing table).

## Changes by inkK (M5, 2026-11-21)

- `inkling_mix_states`: three new launch scalars after `sel_on`: `blocks_off`, `cur_off`, `out_off` (element offsets into `blocks`, `cur`, `out`). Needed for the message mix (bank sets) and the prelude copy (blocks slot 7). `test_mix.py` and `bench_mix.py` pass 0 for all three (not re-run).
- The dispatch does not use `inkling_mix_norm`; the controller output goes to `x` and the normal `norm_in` / `norm_ffn` norms it.
