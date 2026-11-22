# Dense kernels (inkF): projections, norms, embedding, output head

Owner: inkF (bead llama-cpp-tg3.6). Files: `package/kernel/dense/*.loom`, `package/lib/inkling/dense_*.loom`, `package/test/dense/`. Status per kernel is in section 9 (kept current).

## 1. Conventions that all dense kernels share

- Activations are F32, token-packed: row `r` of a tensor with row length `L` is at byte `origin + r * stride * 4` (`stride` in elements, `origin` in bytes). Outputs follow the same rule. Nothing is padded; row `r` is token row `r` of the epoch (`r < T`).
- **Live row count.** Every kernel reads `T` (live rows, `0 <= T <= capacity`) as the first i32 at byte `count_origin` of the `%control` buffer. The grid is sized by the workload value (the capacity of the shape bucket); workgroups past `T` exit at once. One compiled kernel serves every row count. `%control` can be the epoch metadata buffer or any buffer that holds `T` (inkH decides where `T` lives; the head kernels use the selection list of section 4).
- **Weights** are raw GGUF bytes: `%weight: buffer` plus `%w_origin: offset` (bytes into the buffer). Bind a tensor with `command.parameter %w, "blk.{}.attn_q.weight"[%layer] : view<...xi8>` (any element type works, the kernel sees bytes). Row `n` of a `{K, N}` tensor starts at `w_origin + n * row_bytes`; `row_bytes = K/32*34` (Q8_0), `K*2` (BF16), `K*4` (F32). No padding, no repacking (`prepare_weight` returns 0).
- Weight type is fixed by the kernel name (`_q8_0`, `_bf16`, `_f32`). A model variant picks the names with `inkling.dense_type` (Q4_K_XL: Q8_0 for attention, embedding and lm_head, F32 for controllers and the router, BF16 for sink and dynamic conv projections; Q8_K_XL: BF16 for all of them, F32 for the router and small tensors).
- Accumulation is F32. Q8_0 and BF16 weights are decoded exactly. The Q8_0 and BF16 matrix-op GEMMs (`inkling_gemm_q8_0`, `_wide_q8_0`, `inkling_gemm_wmma_bf16`) round weights and activations to F16 (the PR1 WMMA path); the GEMV kernels and `inkling_gemm_{bf16,f32}` do not.
- Epilogue `act` (launch scalar, runtime): `0` none, `1` sigmoid (attention gate), `2` soft cap `cap * tanh(y / cap)` (final logits, `cap` = 30). `cap` is an F32 launch scalar.
- Every dispatch is one kernel launch. Kernels write only rows `< T` and outputs `< n_rows`.
- Tune knobs are `config.decl` values (section 8). A knob changes every launch of that kernel in the stage. Provide the values from the arch row (`package/kernel/dense/tune.json`).
- Names are `inkling_*` (kernels) and `inkling_dense_*` / `@inkling.dense.*` (helpers and templates). Every cross-file call needs the `func.decl` / `template.decl` of the callee in the caller's file (copy them from the head of `gemv.loom`).

## 2. Dense tensors and where each kernel is used

K = input width, N = output rows. `Q4` = Q4_K_XL weight type, `Q8` = Q8_K_XL weight type.

| Tensor (per layer unless noted) | K | N | Q4 | Q8 | Kernel (n = live rows) |
|---|---|---|---|---|---|
| attn_q | 2304 | 3072 | Q8_0 | BF16 | gemv n<=8, gemm n>8 |
| attn_k, attn_v | 2304 | 1024 | Q8_0 | BF16 | same |
| attn_rel | 2304 | 384 | Q8_0 | BF16 | same |
| attn_gate (act = 1 sigmoid) | 2304 | 3072 | Q8_0 | BF16 | same |
| attn_output | 3072 | 2304 | Q8_0 | BF16 | same |
| attn_sink_proj | 2304 | 24 | BF16 | BF16 | gemv (xs) or gemv_split; gemm_bf16 for n > 8 |
| attn_k_conv_proj, attn_v_conv_proj (global layers 5,11,17,23) | 2304 | 32 | BF16 | BF16 | same |
| ffn_gate_inp (router, inkD calls it) | 2304 | 128 | F32 | F32 | gemv_f32 or gemv_split_f32; gemm_f32 for n > 8 |
| loop_{q,k,v,ffn}_ctrl_down (layers 6..17) | 4608 | 64 | F32 | BF16 | gemv_split (K > 4096) |
| loop_{q,k,v,ffn}_ctrl_forget, _gain | 64 | 2304 | F32 | BF16 | gemv_rows (K = 64) |
| token_embd (gather) | 2304 | 201024 rows | Q8_0 | BF16 | embed |
| output (lm_head, selected rows only) | 2304 | 200058 of 201024 | Q8_0 | BF16 | gemv (n<=8) or gemm + act 2 |
| norms (attn_norm, q/k/v_inp_norm, ffn_norm, token_embd_norm, output_norm) | 2304 | | F32 | F32 | rms_norm |

## 3. Launch ABI

All argument lists below are `launch(...)`; the bracket list is the workload (compile-visible, used for the grid).

### 3.1 GEMV: `inkling_gemv_{q8_0,bf16,f32}[n_rows, k_size]` (1 to 8 live rows, K % 256 = 0, K <= 4096)

```
launch(%control: buffer, %count_origin: offset, %k_size: index, %n_rows: index,
       %w_origin: offset, %x_origin: offset, %x_stride: index, %d_origin: offset, %d_stride: index,
       %act: index, %cap: f32, %weight: buffer, %input: buffer, %output: buffer)
```
`y[r][n] = act(sum_k W[n][k] * x[r][k])` for `r < min(T, cols)`, `n < n_rows`. `cols` is the knob `inkling.dense.gemv.cols` (maximum live rows, 1 to 8). `n_rows` is the workload and the launch value (same number). `d_stride >= n_rows`; a larger `d_stride` writes into a wider row (the lm_head output uses `d_stride = 200064`). Plan: 256 workitems per workgroup, `xs_wg_per_unit` x execution units workgroups at most, activations kept in registers (PR1 `ggml_mul_mat_vec_xs`).

### 3.2 GEMV for small and odd shapes (any K % 4 = 0, F32 or BF16 weights, up to 8 live rows)

Same `launch(...)` list as 3.1 (`k_size` and `n_rows` are launch values; the workload is only `[n_rows]`). Two kernels per weight type:

- `inkling_gemv_split_{bf16,f32}[n_rows]`: one workgroup for each weight row, `inkling.dense.split.waves` subgroups split K and combine through workgroup memory (the split-K path). Use it for few rows and long K: the 24, 32 and 128 row tensors, K = 4608 with 64 rows.
- `inkling_gemv_rows_{bf16,f32}[n_rows]`: one subgroup for each weight row, no workgroup memory and no barrier; 256 / subgroup size rows for each workgroup. Use it for many rows and short K: `ctrl_forget` and `ctrl_gain` (2304 rows, K = 64).

Both read activations with 4-wide loads, so `k_size % 4 = 0`.

### 3.3 GEMM (n up to 512 live rows): `inkling_gemm_q8_0`, `inkling_gemm_wide_q8_0`, `inkling_gemm_wmma_bf16`, `inkling_gemm_{bf16,f32}`, all `[n_rows, cols_cap]`

Same `launch(...)` list as 3.1 (`k_size` is the runtime K). `cols_cap` is the row capacity of the shape bucket (tile counts come from it); `T` limits the real work.

| Kernel | Weights | Math | K multiple of | Use |
|---|---|---|---|---|
| `inkling_gemm_q8_0` | Q8_0 | matrix ops, F16 operands, F32 sums, 128 x 64 tile | 64 | prefill |
| `inkling_gemm_wide_q8_0` | Q8_0 | same, 128 x 128 tile | 64 | chosen by the arch dispatch window (section 8) |
| `inkling_gemm_wmma_bf16` | BF16 | matrix ops, weights and activations rounded to F16, F32 sums, 128 x 64 tile | 64 | prefill of the Q8_K_XL variant |
| `inkling_gemm_bf16`, `inkling_gemm_f32` | BF16, F32 | F32 products, subgroups split K | 8 | small N (24, 32, 64, 128), any K, router, controllers |


### 3.4 Norms (`norm.loom`)

```
inkling_rms_norm[rows_cap]         launch(%control, %count_origin, %width, %x_origin, %x_stride, %d_origin, %d_stride, %epsilon: f32, %input, %weight, %output)
inkling_rms_norm_plain[rows_cap]   same without %weight
inkling_rms_norm_heads[rows_cap]   launch(%control, %count_origin, %width, %group, %x_origin, %x_stride, %d_origin, %d_stride, %epsilon: f32, %input, %output)
```
`width % 4 = 0`, `weight` is a F32 `[width]` tensor, `epsilon` = 1e-6. `inkling_rms_norm_heads` normalizes each run of `group` values (128, one attention head) with no weight. Output may alias input. These are the only norm kernels (input norms, ffn_norm, v head norm, controller norms).

### 3.5 Embedding (`embed.loom`)

```
inkling_embed_{q8_0,bf16}[rows_cap]  launch(%control, %count_origin, %vocab: index, %tokens: buffer, %tokens_origin: offset,
                                            %d_origin: offset, %d_stride: index, %epsilon: f32,
                                            %embedding: buffer, %norm_weight: buffer, %output: buffer)
```
`x[r] = rms_norm(token_embd[tokens[r]]) * token_embd_norm` (unpadded ids `< 200058`; `vocab` = 201024, only used for the Q8_0 span). `tokens` is the epoch token buffer (i32 per row, byte `tokens_origin + 4 r`).

## 4. Output head (`head.loom`)

The final mix (inkG) writes the mixed state of every row `cur[r]`. The head runs for the selected rows only (plan flag bit 0).

```
inkling_head_select[spans_cap]  launch(%spans_cap: index, %all_rows: index, %metadata: buffer, %sel: buffer, %sel_origin: offset)
inkling_head_norm[spans_cap]    launch(%sel: buffer, %sel_origin: offset, %spans_cap: index, %width, %x_origin: offset, %x_stride, %d_origin: offset, %d_stride, %epsilon: f32, %logit_valid: index (200058), %logit_stride: index (200064),
                                       %logits: buffer, %logits_origin: offset, %input, %weight, %output)
lm_head  = inkling_gemv_{q8_0,bf16}  (n_sel <= 8)  or  inkling_gemm_*  (n_sel > 8)
           with %control = %sel, %count_origin = %sel_origin, k = 2304, n_rows = 200058, act = 2, cap = 30.0,
           x = head_norm output, d_stride = 200064, output = logits buffer
```
- `inkling_head_select` reads the plan packets in `%metadata` (header `{span count, eos, ...}` at byte 0, packets `view<[spans]x9xi32>` at byte 16, fields `{length, position, source, row, input offset, input length, flags, ...}`). For every span with `flags & 1` it appends `{span index, token row = input offset + length - 1}` to the selection list.
- Selection list (`%sel` at `%sel_origin`): `i32 n_sel`, 3 reserved words, then `view<[spans_cap]x2xi32>` of `{span index, token row}` in plan order. Slot `k` of every head buffer belongs to `sel[k]`.
- `inkling_head_norm` computes `x_sel[k] = rms_norm(cur[row_k]) * output_norm` and also writes `-inf` into the 6 padding entries of `logits[k]` (indexes 200058 to 200063).
- **Logits buffer layout (for the sampler):** `logits[k][v]`, F32, row stride `LOGIT_STRIDE = 200064` elements (800256 bytes, a multiple of 256), `k < n_sel`, `v < 200058` real logits after the soft cap `30 * tanh(z / 30)`, `v` in `[200058, 200064)` = `-inf`. The buffer holds `spans_cap` rows. The lm_head rows 200058 to 201024 (padded vocabulary) are never computed. Row `k` belongs to span `sel[k].span`; the sampler writes its token into `outputs[span]`.
- The mix (inkG) output rows are the head input `%input` of `inkling_head_norm`; the kernel reads row `sel[k].row` of it.
- **Packet layout (current).** `inkling_head_select` reads the packet of `lib/vendor/packet.loom`: header words `{live tokens, spans, chosen count}` (12 bytes), `packet_capacity` six-word descriptors, then `packet_capacity` selection words (the token rows of the flagged spans, in order). `inkling.packet_capacity` is a config value. The selection list written to `%sel` holds `{k, row}`: the first word is the position in the selection list, not the span number (the packet already lists only the flagged spans). `%all_rows` 1 selects every live token (capture runs).
- **Bound proofs.** `range(x, lo, hi)` is inclusive, so the `index.assume` lines of `head.loom` do not prove `k < packet_capacity` or `k < spans_cap`; the proofs hold only when the launch size is a compile-time constant, as in the dispatch command (spans capacity from the config). A `check.case` cannot give a kernel a constant launch argument (no `config.get`, no `index.constant`; `func.def` and `check.scenario` did not fold it either), so `test/dense/test_head.py` links a copy of `head.loom` in which the two `%cap = index.assume %spans_cap ...` lines are replaced by `%cap = index.constant N`. The kernel file is unchanged and no bug was found: the code is correct for every launch size up to the capacity.

## 5. Weight decode and layout notes

- Q8_0: 34-byte blocks `{f16 d; i8 q[32]}`, 2-byte alignment; the kernels load with 16-byte vector loads at 2-byte aligned addresses (as PR1).
- BF16 widens by a 16-bit shift. F32 plain. Row `n` of every weight is contiguous; the GGUF tensor data offset is 32-byte aligned.
- All weight motif code is copied from PR1 (`motifs/{common,float,q8_0,wmma}.loom`, `matmul/lib/mm_wmma.loom`) with the prefix `ggml` replaced by `inkling.dense` / `inkling_dense`. The copies keep the same tuned algorithms; only the ABI around them changed.

## 6. Compile and test

```
laptop: package/test/dense/sync.sh                      # copies lib, kernels and tests to the box (~/inkling/inkF/package)
box:    . ~/loom-serve/.venv-inkF/bin/activate; cd ~/inkling/inkF/package/test/dense
        sh run_all.sh                                   # every test on both GPUs (shared GPU lock), exit code 1 on any failure
        python test_gemv.py gfx1201 [filter|-] [xs|split|rows]
        python test_gemm.py gfx1151 [filter|-] gemm_q8_0|gemm_wide_q8_0|gemm_wmma_bf16|gemm_bf16|gemm_f32
        python test_norm_embed.py gfx1201 ; python test_head.py gfx1201 ; python test_oracle_layers.py gfx1201
        sh compile_report.sh                            # registers, spills, matrix ops of every kernel on both targets (no GPU)
        python bench_dense.py gfx1201 [filter]          # timed runs (exclusive GPU lock), loom-benchmark dispatch_complete
```
Each case is a Loom `check.case` (inputs and outputs as `.npy`, launched with the production kernels), the numpy reference is computed from the exact weight values of the GGUF, and oracle tensors come from `oracle/dumps/Q8_K_XL_A`. Oracle `norm*` tensors carry no weight (the weight is applied after them); `norm-S` occurrences of global layers are ordered k, v, q.
See section 9 for the result table.

## 7. Reuse map (PR1 to dense)

| Dense | PR1 source | Change |
|---|---|---|
| `inkling_gemv_*` | `matmul/mul_mat.loom` `ggml_mul_mat_vec_xs` | weight type per kernel, no batch dims, live columns from the control header, output stride and epilogue, `config` renamed |
| `inkling_gemm_q8_0`, `_wide_q8_0` | `matmul/mul_mat_wmma.loom` `ggml_mul_mat_wmma_q8_0_c64`, `_q8_0` | same ABI change |
| `inkling_gemm_{bf16,f32}` | `matmul/mul_mat_f32.loom` | same ABI change |
| `inkling_gemm_wmma_bf16` | `matmul/mul_mat_wmma.loom` `ggml_mul_mat_wmma_q8_0_c64` | the Q8_0 block loader replaced by a BF16 loader (bf16 to f32 to f16); PR1 has no BF16 matrix-op path |
| `inkling_gemv_split_*`, `inkling_gemv_rows_*` | none (new) | split-K and subgroup-per-row GEMV for small and odd shapes |
| `inkling_embed_*`, `inkling_head_*` | none (new) | gather, selection list, gathered norm |
| `inkling_rms_norm*` | `norm/rms_norm_f32.loom` (algorithm), hrx-loom-library `motif/tensor/rmsnorm.loom` (style) | rewritten for token-packed rows, optional weight, per-head variant |

## 8. Tune rows (data, per arch)

Machine-readable copy: `package/kernel/dense/tune.json` (flat key to value, one object per arch plus `portable`; the `dispatch` sub-object is for the dispatch program, not for `config_i64`). The cold program or the card preset sets the keys with `config_i64`; loom-test and loom-benchmark take `--config=key=value` (the test scripts read the arch row from the file). The PR1 values are copied from `ggml-hrx-arch.cpp` (`mul_mat_vec`, `mul_mat_wmma`, `mul_mat_f32` rows). Nothing is re-tuned. The CU count is not a knob: the kernel reads `target.execution.unit.count` (gfx1151 40, gfx1201 64) and multiplies it with `xs_wg_per_unit` (PR1 `xs_wg_per_cu`).

| Key | gfx1151 | gfx1201 | portable | Source |
|---|---|---|---|---|
| `inkling.dense.gemv.cols` (maximum live rows of the GEMV kernels, 1 to 8) | 8 | 8 | 8 | PR1 `max_cols`, a build with 1 is faster for decode only (section 9) |
| `inkling.dense.gemv.xs_wg_per_unit` | 4 | 2 | 2 | PR1 `xs_wg_per_cu` |
| `inkling.dense.gemv.xs_depth` | 2 | 2 | 2 | PR1 |
| `inkling.dense.f32.depth` | 3 | 3 | 3 | PR1 `mul_mat_f32` |
| `inkling.dense.f32.k_split` | 8 | 8 | 8 | PR1 `mul_mat_f32` |
| `inkling.dense.split.waves` (split-K GEMV) | 4 | 4 | 4 | new kernel, default, not tuned |
| dispatch `gemm_wide_min_tiles` | 32 | 32 | 32 | PR1 `mul_mat_wmma` Q8_0 |
| dispatch `gemm_wide_max_work` (tiles x K) | 524288 | 524288 | 524288 | PR1 `mul_mat_wmma` Q8_0 |

The two dispatch values are for inkH: use `inkling_gemm_wide_q8_0` when `tiles >= gemm_wide_min_tiles` and `tiles * K <= gemm_wide_max_work`, with `tiles = ceil(n_rows/128) * ceil(n_cap/128)`, else `inkling_gemm_q8_0`.

### 8.1 What the package must provide (integration notes for inkH)

1. **Target profiles.** The matrix-op providers (`dense_wmma.loom`) are selected by exact target identity, as in PR1. `config.loom` of the package must define both lines of `package/lib/inkling/dense_targets.loom` (`amdgpu.target<gfx11-generic> @inkling_dense_gfx11` and `amdgpu.target<gfx12-generic> @inkling_dense_gfx12`), like the AMD library does for `@tensor_matrix_rdna3`. Without them the code still compiles and is correct, but the GEMM falls back to the scalar F32 provider and is 5 to 15 times slower (measured, section 9).
2. **Source list** (`definition.sources`): `lib/inkling/dense_{common,float,q8_0,abi,vec,mm_wmma,wmma,targets}.loom` and `kernel/dense/{norm,embed,gemv,gemv_split,gemm_q8_0,gemm_f32,head}.loom`. The linker keeps only the kernels that a command launches.
3. **Config keys** of section 8, from `tune.json`. All keys have a `config.decl`; the unit count comes from the device.
4. Origin and licence: the weight motifs and the GEMV and GEMM kernels derive from the llama.cpp ggml-hrx kernels (branch hrx-llama-pr1, MIT, same authors). No file of the AMD library was copied.

## 9. Status and results (2026-11-20)

### 9.1 Kernels

All kernels below are finished, pass their tests on both GPUs (section 9.2) and compile without spills on both targets (section 9.4).

| Kernel | File | Weights | Notes |
|---|---|---|---|
| `inkling_rms_norm`, `_plain`, `_heads` | `kernel/dense/norm.loom` | F32 weight or none | weighted, unweighted, per head (group 128) |
| `inkling_embed_q8_0`, `_bf16` | `embed.loom` | Q8_0, BF16 | gather + rms norm + token_embd_norm |
| `inkling_gemv_{q8_0,bf16,f32}` | `gemv.loom` | Q8_0, BF16, F32 | 1 to 8 rows, K % 256 = 0, K <= 4096, epilogue act |
| `inkling_gemv_split_{bf16,f32}` | `gemv_split.loom` | BF16, F32 | split-K, any K % 4 = 0 |
| `inkling_gemv_rows_{bf16,f32}` | `gemv_split.loom` | BF16, F32 | subgroup per row (K = 64) |
| `inkling_gemm_q8_0`, `inkling_gemm_wide_q8_0` | `gemm_q8_0.loom` | Q8_0 | matrix ops, two tile sizes |
| `inkling_gemm_wmma_bf16` | `gemm_bf16_wmma.loom` | BF16 | matrix ops |
| `inkling_gemm_{bf16,f32}` | `gemm_f32.loom` | BF16, F32 | F32 products, split-K waves, any K % 8 = 0 |
| `inkling_head_select`, `inkling_head_norm` | `head.loom` | F32 weight | selection list, gathered norm, logits padding |

Not done: a fused loop-controller kernel (inkG owns the mix and the message; the controller GEMVs are `gemv_split` / `gemv_rows` calls), a wide (128 column) BF16 matrix-op variant, wave64 matrix-op providers (both target GPUs are wave32), and any re-tuning (milestone M8).

### 9.2 Test results (320 cases, all PASS)

`sh package/test/dense/run_all.sh` on the box (shared GPU lock), 160 cases on each GPU, exit code 0. "nmse vs numpy" is the sum of squared errors over the sum of squares against a float64 reference computed from the exact (dequantized) weights of the GGUF (gate: 1e-9 for exact kernels, 5e-6 for the F16 matrix-op kernels). "oracle error" is the relative L2 (or nmse for norms) against `oracle/dumps/Q8_K_XL_A` (prompt A, 16 rows; gate 6e-3 for BF16 weights, 5e-2 for Q8_0 weights of a different quantization). The worst GEMV error 2e-10 is the soft-cap case (approximate tanh `<afn>`).

| Test (cases per GPU) | What it checks | worst nmse vs numpy | worst oracle error |
|---|---|---|---|
| `test_norm_embed` (14) | weighted rms (attn_norm 0, ffn_norm 0, q/k/v_inp_norm 5), unweighted, per-head v norm, strides, live rows < capacity, embedding gather + token norm from the full BF16 and Q8_0 tables (ids up to 201023, edge ids) | 7.6e-15 | 4.5e-15 (oracle `norm`, `norm-S`, `Vcur-0`) |
| `test_gemv xs` (17) | Q8_0, BF16, F32 GEMV n = 1, 3, 8; q, k, v, rel, o, gate (act sigmoid), sink 24, conv proj 32, router 128, lm_head with soft cap | 2.0e-10 | 1.6e-3 (`proj_q-0`, `conv_in-0` k v o, `proj_r-0`) |
| `test_gemv split` (25), `rows` (25) | split-K and subgroup-per-row GEMV: 24, 32, 128 rows, K = 4608 (64 rows), K = 64 (2304 rows), n = 1, 3, 8, F32 and BF16, sigmoid | 1.3e-14 / 3.0e-14 | 1.4e-3 |
| `test_gemm gemm_q8_0` (11), `gemm_wide_q8_0` (11) | Q8_0 matrix-op GEMM n = 1, 16, 33, 77, 100, 200, 308, 512, partial tiles, rows beyond the live count untouched | 8.7e-8 | 5.1e-3 (Q8_0 weights against the BF16 oracle) |
| `test_gemm gemm_wmma_bf16` (13) | BF16 matrix-op GEMM, same shapes plus small N (24, 32, 128), K = 4608, K = 64 | 5.0e-8 | 1.4e-3 |
| `test_gemm gemm_bf16` (13), `gemm_f32` (3) | F32-product GEMM: same shapes, router, controllers | 1.3e-13 | 1.4e-3 |
| `test_head` (15) | selection list from a packet (packet capacity equal to and larger than the span capacity), gathered output norm, `-inf` logit padding, logits layout, GEMV and GEMM (WMMA for BF16) lm_head with soft cap, n_sel = 1, 3, 12; oracle chains: a row whose normalization equals `result_norm` as the last row of a one-span and a four-span packet through select, norm and lm_head; 4 lm_head-only oracle cases | 4.4e-8 (GEMM) | `result_norm` nmse 1.5e-14, `result_output` 8.7e-4 (BF16), 2.4e-3 (Q8_0), argmax and top-5 equal |
| `test_oracle_layers` (17) | all projections of layers 0, 1, 2, 3, 4, 5, 6, 8, 11, 12, 17, 18, 23 (local and global, prelude, loop core, coda) with BF16 weights, and layers 0, 5, 17, 23 with Q8_0 weights: q, k, v, rel, gate + o (through `kqv_out`), router logits (F32) | 1.6e-7 | 9.4e-3 (Q8_0 `o`, layer 23), BF16 at most 2.5e-3, router 1.5e-7 |

Raw output per test and GPU: `docs/results_dense/<arch>_<test>.txt` (also `~/inkling/inkF/results/` on the box).

### 9.3 Timing against traffic floors

Method: `loom-benchmark --measure=dispatch_complete` (batch 64, warmed, stop when stable, `flock -x` on the GPU lock), operation time p50 in microseconds, input ring >= 200 MB so weights come from DRAM and not from cache. Floor = weight bytes / measured bandwidth (R9700 620 GB/s, 8060S 230 GB/s). The matrix-op rows also give TFLOP/s against the nominal dense F16 matrix rate (R9700 about 190, 8060S about 59; nominal values are not measured). One dispatch per operation; each row is a single run per kernel (no repeats yet), so differences below 5 percent are noise. Script: `package/test/dense/bench_dense.py`, raw output `docs/bench_dense_gfx1201.txt` and `docs/bench_dense_gfx1151.txt`.

`cols=1` marks a build of the GEMV kernel with `inkling.dense.gemv.cols = 1` (decode only). The other rows are `cols = 8`.

| Shape (weights, rows n) | R9700 time | floor | % of floor | 8060S time | floor | % of floor |
|---|---|---|---|---|---|---|
| attn_q / gate 3072x2304 Q8_0, n=1 | 28.7 | 12.1 | 42 | 64.4 | 32.7 | 51 |
| same, `cols=1` | 15.3 | 12.1 | 79 | 38.8 | 32.7 | 84 |
| attn_q / gate 3072x2304 BF16, n=1 | 29.0 | 22.8 | 79 | 81.4 | 61.5 | 76 |
| same, `cols=1` | 26.8 | 22.8 | 85 | 70.7 | 61.5 | 87 |
| attn_q Q8_0, n=8 | 29.9 | 12.1 | 40 | 55.3 | 32.7 | 59 |
| attn_k / v 1024x2304 Q8_0 / BF16, n=1 | 11.3 / 11.3 | 4.0 / 7.6 | 36 / 67 | 22.6 / 27.2 | 10.9 / 20.5 | 48 / 76 |
| attn_rel 384x2304 Q8_0 / BF16, n=1 | 5.9 / 6.2 | 1.5 / 2.9 | 26 / 46 | 13.8 / 16.1 | 4.1 / 7.7 | 30 / 48 |
| attn_o 2304x3072 Q8_0 / BF16, n=1 | 23.1 / 28.0 | 12.1 / 22.8 | 52 / 82 | 49.3 / 72.9 | 32.7 / 61.5 | 66 / 84 |
| sink 24x2304 BF16, xs / split / rows | 2.7 / 3.0 / 6.4 | 0.2 | launch bound | 3.2 / 3.3 / 6.5 | 0.5 | launch bound |
| dynamic conv proj 32x2304 BF16, xs / split | 2.7 / 3.0 | 0.2 | launch bound | 3.8 / 3.4 | 0.6 | launch bound |
| router 128x2304 F32, xs / split | 4.6 / 4.2 | 1.9 | 45 | 8.6 / 8.0 | 5.1 | 64 |
| ctrl_down 64x4608 F32 / BF16, split | 5.0 / 4.3 | 1.9 / 1.0 | 38 / 22 | 8.3 / 6.7 | 5.1 / 2.6 | 62 / 38 |
| ctrl_forget, gain 2304x64 F32 / BF16, rows | 3.5 / 3.5 | 1.0 / 0.5 | 27 / 14 | 6.6 / 6.0 | 2.6 / 1.3 | 39 / 21 |
| same, split (for comparison) | 9.9 / 9.9 | | | 19.2 / 18.2 | | |
| lm_head 200058x2304 Q8_0, n=1 | 1534 | 794 | 52 | 3156 | 2140 | 68 |
| same, `cols=1` | 798 | 794 | 99 | 2171 | 2140 | 99 |
| lm_head BF16, n=1 | 1550 | 1494 | 96 | 4096 | 4028 | 98 |
| same, `cols=1` | 1490 | 1494 | 100 | 4211 | 4028 | 96 |
| lm_head Q8_0 / BF16, n=8 | 1563 / 1578 | 794 / 1494 | 51 / 95 | 3213 / 4136 | 2140 / 4028 | 67 / 97 |
| lm_head GEMM n=16, Q8_0 / BF16 matrix ops | 1009 / 1605 | 794 / 1494 | 79 / 93 | 2532 / 4419 | 2140 / 4028 | 84 / 91 |

| GEMM attn_q 3072x2304 | R9700 us (TFLOP/s) | 8060S us (TFLOP/s) |
|---|---|---|
| Q8_0 n=16 | 29.1 (7.8) | 53.1 (4.3) |
| Q8_0 n=128 | 52.1 (34.8) | 85.2 (21.3) |
| Q8_0 n=308 | 92.0 (47.4) | 228.4 (19.1) |
| Q8_0 n=512 | 134.3 (54.0) | 333.0 (21.8) |
| Q8_0 wide n=512 | 113.8 (63.7) | 304.2 (23.8) |
| BF16 matrix ops n=16 | 38.1 (5.9) | 79.2 (2.9) |
| BF16 matrix ops n=128 | 49.9 (36.3) | 109.1 (16.6) |
| BF16 matrix ops n=308 | 88.4 (49.3) | 258.1 (16.9) |
| BF16 matrix ops n=512 | 130.3 (55.6) | 367.4 (19.7) |
| BF16 F32 products n=512 | 793.7 (9.1) | 1448.4 (5.0) |
| attn_o 2304x3072 n=512: Q8_0 / BF16 matrix ops | 143.9 / 139.6 | 594.0 / 605.6 |

Norm, embedding (n = rows): `rms_norm` 2304 n=1 3.0 us (R9700) / 2.8 us (8060S); n=512 41.5 / 51.1 us (9.4 MB, 37 / 80 percent of bandwidth); `rms_norm_heads` 8x128 n=512 13.8 / 20.9 us; `embed` n=1 2.2 / 2.2 us; n=512 BF16 33.7 / 10.6 us, Q8_0 6.2 / 10.4 us (table read floor BF16 3.8 / 10.3 us, Q8_0 2.0 / 5.4 us).

Estimated dense time of one decoded token (sum of the measured kernels with one launch each, no gaps, Q4_K_XL types, `cols=1` for q/gate/lm_head): R9700 about 6.1 ms (attention 4.1 ms, lm_head 0.8 ms, controllers 1.15 ms on 24 looped visits, conv proj 0.05 ms) against a dense floor of 3.4 ms (2093 MB: attention, lm_head, controllers); 8060S about 13.3 ms against 9.1 ms. The gap is made of launch-bound small kernels (2.7 to 5 us each, about 12 dense launches for each visit) and the Q8_0 GEMV decode cost; fusing the controller (inkG) and fusing the small projections that share one input (sink, conv proj, rel with q and gate) would remove most of it.

### 9.4 Compile reports

`package/test/dense/compile_report.sh`, full table in `docs/compile_report_dense.txt` (static report of `loom-compile --target=amdgpu:gfx1151|gfx1201`): no spills in any kernel, no private memory. Register use of the main kernels (vgpr count, occupancy of a SIMD): norms and embedding 12 to 21 vgpr (100 percent); `gemv_split` / `gemv_rows` 72 to 76 (100); `gemm_q8_0` / `gemm_wmma_bf16` 112 (gfx1151) and 132 to 136 (gfx1201), 50 percent, 16 matrix instructions in the main loop; `gemm_wide_q8_0` 160 and 184, 37 percent, 32 matrix instructions; `gemm_bf16` / `gemm_f32` 72 to 100, 62 percent (no matrix ops). The GEMV kernels need the device for their launch configuration (`target.execution.unit.count`), so the static report skips them; the benchmark reports of the device runs give 175 (Q8_0), 176 (BF16) and 184 (F32) vgpr for the `cols = 8` build, which is why that build runs at 50 percent occupancy and the `cols = 1` build is faster.

### 9.5 Open problems and notes

1. **Matrix-op providers need the two target profiles in the package** (section 8.1). Verified: with the two `amdgpu.target` lines in the module the gfx11 provider runs on the 8060S and the gfx12 provider on the R9700 (loom-test and loom-benchmark, both pass). Without them the scalar fallback runs (5 to 15 times slower, same results). Whether loom-serve resolves the target symbols of a package the same way is not tested here (inkH M1); the AMD library defines `amdgpu.target<gfx11-generic> @tensor_matrix_rdna3` in the package `config.loom`, so this is expected to work.
2. **`cols` is a per-stage knob.** `gemv` with `cols = 1` is 1.5 to 2 times faster for Q8_0 at decode (lm_head 1534 to 798 us on the R9700, 3156 to 2171 us on the 8060S). Shape buckets are separate stages, so the cold program should set `inkling.dense.gemv.cols` to the largest live row count of the bucket (1 for the decode bucket). A mixed bucket with n up to 8 pays the 50 percent occupancy cost of the 8-column build for Q8_0.
3. **Q8_0 GEMV at `cols = 8` and the GEMM tail.** Q8_0 decode costs more instructions than BF16; with `cols = 1` it reaches 79 to 99 percent of the floor on large tensors. The small tensors (k, v, rel: 1 to 2.5 MB) reach only 26 to 48 percent at one launch of 6 to 23 us; they are latency bound.
4. **8060S `attn_o` GEMM (K = 3072, N = 2304) at n=512** is 1.8 times slower than `attn_q` for the same FLOP count (594 and 606 us against 333 and 367); not investigated (M8).
5. **`embed_bf16` at 512 rows on the R9700** takes 33.7 us (table read floor 3.8 us); the Q8_0 variant 6.2 us. Likely page-table cost of the 926 MB table; not investigated. The cost is 34 us once for each 512-token prefill epoch.
6. `inkling_head_select` runs one thread (serial loop over at most `spans_cap` packets); it is a few us. If the plan layout of inkH differs from the assumed one (fields 0, 4 and 6), change the three loads in `head.loom`.
7. The final mix (inkG) is not part of this area. `result_norm` and `result_output` are checked through a chain test that feeds a row whose normalization equals the oracle `result_norm` (rms norm is scale invariant): `head_oracle_chain_*` (result_norm nmse 1.5e-14, result_output relative L2 8.7e-4 BF16, 2.4e-3 Q8_0, top-5 equal).
8. **Chunk-size independence.** The GEMV kernels and `inkling_gemm_{bf16,f32}` are exact (F32 products, nmse 1e-14 against float64). The matrix-op GEMMs round weights and activations to F16 (PR1 behaviour): nmse 4e-8 to 9e-8 (relative L2 about 3e-4) against float64. A token that goes through GEMV in a decode step and through a matrix-op GEMM in a prefill chunk therefore differs by about 3e-4 relative L2 per projection (far below the 1.4e-3 to 5e-3 relative L2 between the Q8_0/BF16 kernels and the oracle that quantizes its activations). Rows inside the same kernel are independent of the chunk (each output row uses only its own input row and a fixed K order). If bitwise equality between decode and prefill is required, dispatch `inkling_gemm_bf16` / `inkling_gemv_*` for every n (slower prefill: 794 us against 130 us for attn_q at n=512 on the R9700). The loop controllers and the router must always use the exact kernels (`gemv_split`, `gemv_rows`, `gemm_f32`, `gemm_bf16`); the tests cover them for every n.

## Changes by inkJ (integration, 2026-11-20)

- `kernel/dense/gemv.loom`: `inkling_gemv_{q8_0,bf16,f32}` take a third workload value `%per_unit` (workgroups per execution unit), used by the launch configuration. The dispatch passes the result of `template.apply<@inkling.dense.gemv.xs_wg_per_unit>()` (providers in `lib/inkling/dense_vec.loom`: 4 for gfx11, 2 portable default and gfx12). Reason: `template.apply` is not allowed in a launch configuration or in a `pure` function. The config key `inkling.dense.gemv.xs_wg_per_unit` is removed. Test scripts under `package/test/dense` that launch the GEMV need `[n_rows, k_size, per_unit]`.
- `func.def` of `lib/inkling/dense_*.loom` and `kernel/dense/*.loom` are `public`.
- Per-stage config set by `prepare.loom` (`config_kernels`): `inkling.dense.gemv.cols` (rows of the bucket up to 8), `xs_depth` 2, `split.waves` 4, `f32.depth` 3, `f32.k_split` 8.
- Dispatch rule: bucket capacity up to 8 rows uses GEMV (`gemv_q8_0`/`gemv_bf16`, `gemv_split_bf16` for sink and the conv projections), larger buckets use `gemm_q8_0`/`gemm_wmma_bf16`/`gemm_bf16`. `gemm_wide_q8_0` is not selected yet.
- Dense weights are declared with exact `command.parameter` footprints (`N x K/32 x 34 x i8` for Q8_0, `N x K x bf16`).
