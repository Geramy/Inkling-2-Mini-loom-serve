# Expert kernels (inkD): ABI, tests, timings

Router, expert FFN (decode and prefill) and combine for the Inkling 2 Mini MoE, as plain Loom kernels.
Sources: `package/lib/inkling/{nvfp4,experts_common,experts_stage,experts_wmma}.loom`, `package/kernel/experts/{router,routes,gate_up,down,combine,gate_up_wmma,down_wmma}.loom`. Tests: `package/test/experts/` (run on the box, see the end).
All kernels assume wave32 (both targets). Weight buffers are the raw GGUF tensor bytes (bind with `command.parameter`, no repacking). No `config.decl` is used: sizes are workload values, the weight type is a launch scalar.

## Dataflow of one MoE visit (n tokens, 8n routes, route r = token * 8 + slot)

```
x (f32 [n][2304], the ffn-normed vector: norm * ffn_norm.weight)
 -> inkling_router_logits : logits f32 [n][128]  + x16 f16 [n][2304] (slot order, see below)
 -> inkling_router_select : ids i32 [n][8], wts f32 [n][8]   (wts = normalised * 8.0)
 -> inkling_experts_routes: tasks (route records grouped by expert)
 -> gate_up (+scale, SiTU): act f16 [8n][1152] (slot order)
 -> down (+scale)         : dout f32 [8n][2304]
 -> inkling_experts_combine: y f32 [n][2304] = sum_s wts[t][s] * dout[t*8+s]
```
Six launches per visit. `y` is the oracle `conv_in-S` occurrence 3 (the FFN branch before the post conv). Note: the oracle tensor `norm-S` is the RMS norm WITHOUT the weight; the router input is `norm-S * ffn_norm.weight`.

## Launch ABI

`n` is the token count of the launch (workload value and device argument, equal). Rows beyond the live count compute garbage from whatever `x` holds; the ids are always in 0..127, so that is safe but costs time: pass the live count as `n` when it is known at plan time. `n <= 512` (routes kernel). `wtype` is a device scalar argument, pass a constant: 0 NVFP4, 1 Q8_0, 2 BF16. Per layer: layers 0..19 `wtype 0` with the `.scale` buffers; layers 20..23 `wtype 1` (Q4_K_XL file) or `2` (Q8_K_XL file), and then `scales` is not read (pass any 512 B buffer). The dtype choice per layer is therefore the compile-time constant the layer command passes. NOT verified: whether loom-serve folds the constant `wtype` into the kernel (the standalone compile contains all three paths; vgpr 56). If the constant is not specialised the three paths still cost nothing at run time (uniform branch).

| Kernel (workload) | Launch arguments | Grid x workgroup |
|---|---|---|
| `inkling_router_logits[n]` | `n, x:f32[n][2304], rw:f32[128][2304] (ffn_gate_inp.weight), logits:f32[n][128], x16:f16[n][2304]` | (32, ceil(n/4)) x 128 |
| `inkling_router_select[n]` | `n, logits, bias:f32[128] (exp_probs_b.bias), ids:i32[n][8], wts:f32[n][8]` | ceil(n/4) x 128 |
| `inkling_experts_routes[]` | `n, tile:index, ids, tasks` | 1 x 128 |
| `inkling_experts_gate_up[n]` | `n, wtype, x16, tasks, w (ffn_gate_up_exps.weight), scales:f32[128] (ffn_gate_up_exps.scale), act:f16[8n][1152]` | (72, records(n)) x 128 |
| `inkling_experts_down[n]` | `n, wtype, act, tasks, w (ffn_down_exps.weight), scales:f32[128] (ffn_down_exps.scale), dout:f32[8n][2304]` | (144, records(n)) x 128 |
| `inkling_experts_gate_up_wmma[n]` | same as `gate_up` | (18, records32(n)) x 128 |
| `inkling_experts_down_wmma[n]` | same as `down` | (18, records32(n)) x 128 |
| `inkling_experts_combine[n]` | `n, dout, wts, y:f32[n][2304]` | (9, n) x 256 |

Two interchangeable expert paths, selected by the layer command from the (compile-time) token count of the stage shape:
- portable (decode, small chunks): `routes` with `tile = 8`, `gate_up`, `down`. Works on any wave32 target.
- matrix (prefill): `routes` with `tile = 32`, `gate_up_wmma`, `down_wmma`. Uses gfx11 and gfx12 WMMA through providers (`inkling.mma.*`, `experts_wmma.loom`, copied from the PR1 motif); other wave32 targets get the portable provider of the same template (correct, slow).
Both give the same results. `routes` tile is at most 32 (record stride is fixed at 34 words); the matrix kernels require tile 32.

Sizes: `records(n) = min(8n, n + 128)` (tile 8 bound, also a safe bound for tile 32), `records32(n) = min(8n, n/4 + 128)`. Workspace: `tasks` = 16 + records(n) * 136 B (n = 1: 1,104 B; n = 512: 87,568 B; no zeroing needed, `routes` writes the count), `x16` 4,608 B per token, `act` 2,304 B per route, `dout` 9,216 B per route (n = 512: 37.7 MB), `logits` 512 B per token. The six kernels must run in order; each reads what the previous one wrote.

Weight layouts (GGUF bytes): gate_up expert e row r (0..1151 gate, 1152..2303 up) at `(e*2304 + r) * rowbytes`, rowbytes NVFP4 1296, Q8_0 2448, BF16 4608. down row r of expert e at `(e*2304 + r) * rowbytes`, rowbytes NVFP4 648, Q8_0 1224, BF16 2304.

### Slot order of f16 activations

`x16` and `act` hold f16 values in "slot order": inside each group of 16 K elements the dwords are the natural element pairs (0,2) (8,10) (1,3) (9,11) (4,6) (12,14) (5,7) (13,15). The NVFP4 decoder yields operand pairs in this order without any shuffle, and both operands of a matrix operation may use any K order as long as it is the same. Host converters: `refs.to_slots` / `refs.from_slots`. No other kernel may read `x16` or `act`.

## Numerics

- Router logits are f32 FMA (relative L2 8e-8 against the oracle). Select: `sigmoid` (afn variant), `+ bias`, top 8 by 8 rounds of argmax, ties to the lower expert index (the oracle uses `std::partial_sort`, tie order unspecified), `w / (sum + 1e-20) * 8.0` on the unbiased sigmoid.
- NVFP4: a nibble becomes the f16 value real/2^14 by shifts and masks (exact, denormal inputs of `dot2`), 8 `dot2` (f16 operands, f32 accumulate) per 16 elements, then the UE4M3 sub-block scale (code 0x7F = 0) times 2^14 in f32. The per-expert `.scale` multiplies gate and up (together) and the down result in f32.
- Matrix path: weights are decoded to real f16 (NVFP4: exact; Q8_0: `d * q` rounded to f16, relative 5e-4; BF16: rounded to f16), accumulation in f32.
- Activations are rounded to f16 once (`x16`, `act`): relative L2 2e-4 on the result of the gate_up GEMV.
- f16 range, all NVFP4 layers (scan of the 20 layers, 1.27e9 sub-blocks, `package/test/experts/scale_scan.py`): UE4M3 bytes 76..126 (raw scale 6.0 .. 448.0), none zero, none 0x7F. Largest weight before the per-expert scale 6 * 448 = 2688, smallest nonzero 3.0. Largest f16 product exponent is far from overflow (65504) and every weight value is exactly representable in f16 (E2M1 has 1 mantissa bit, UE4M3 has 3). The `.scale` (about 1e-4) is applied in f32 on the output. Activations: max |x| of the ffn-normed vector over all 48 slots of prompt A is 8.8; SiTU outputs are bounded by 4 * 25 = 100. No bf16 or f32-accumulate fallback is needed.

## Tune data (no tuning loops were run; defaults from the first measurements)

| Knob | gfx1201 | gfx1151 | portable default |
|---|---|---|---|
| decode path tile (`routes` tile of the portable path) | 8 | 8 | 8 |
| matrix path tile | 32 | 32 | n/a (use portable path) |
| minimum n for the matrix path (measured crossover, gate_up) | 48 | 48 | never |
| lanes per row, pair rows per workgroup (portable) | 8, 16 | 8, 16 | 8, 16 |

Crossover data (device time, NVFP4 layer 0): gfx1201 n=32 portable/matrix gate_up 387/528 us, down 292/321; n=64 689/554, 527/337. gfx1151 n=32 1204/1596, 842/800; n=64 1882/1676, 1358/822. These rows belong in the arch table of inkH; the kernels have no per-arch code besides the WMMA providers.

## Tests (all PASS on gfx1201 and gfx1151)

Commands (box, `. ~/loom-serve/.venv-inkD/bin/activate; cd ~/inkling/inkD/package/test/experts`): `./run_all.sh gfx1201` and `./run_all.sh gfx1151`; individual: `test_gate_up.py`, `test_oracle.py <arch> [slots]` (env `ROWS=5` slices tokens, `PATHS=portable,wmma`), `test_prefill.py <arch> [n ...]`, `bench_experts.py <arch> [n ...]` (env `WMMA=1`, `LAYER`, `KIND`). Fixtures are `.npy` files, kernels run through `loom-test` (`check.case`), results are compared in numpy. Host reference: numpy decode of the GGUF bytes (`refs.py`), f32/f64 math.

| Test | Shapes | Weights | Result (both GPUs) |
|---|---|---|---|
| router vs oracle `ffn_moe_logits`, ids (`ffn_moe_topk`), `ffn_moe_weights_scaled` | prompt A, 16 tokens, slots 0, 9, 19, 27, 39 (layer 9 in passes 0, 1, 2), 20..23; n = 1 (token 5) and n = 2 (tokens 3, 9) slices | F32 router | logits rel L2 <= 9.1e-8; id sets equal on every row; weights rel L2 <= 1.4e-7; `x16` vs x 2.1e-4 |
| experts vs host reference (f32 activations) | same slots and shapes, portable and matrix path | NVFP4 layers 0, 9, 19 (Q4_K_XL file, identical to the Q8 file); BF16 layers 20..23 (Q8_K_XL file); Q8_0 layers 20..23 (Q4_K_XL file, no oracle dump: host reference only) | act 3.0e-4 .. 3.4e-4, dout and y 2.3e-4 .. 3.4e-4 (all dtypes, both paths); matrix-path Q8_0 same order |
| experts `y` vs oracle `conv_in-S` occ. 3 (the FFN output before the post conv) | slots 0, 9, 19, 27, 39 | NVFP4 | 8.6e-3 .. 1.2e-2 |
| same | slots 20..23 | BF16 (Q8_K_XL) | 1.8e-3 .. 2.4e-3 |
| same | slots 20..23 | Q8_0 experts (Q4_K_XL) against the BF16 oracle: includes the weight quantisation difference | 6.5e-3 .. 8.5e-3 |
| gate_up + SiTU synthetic, n = 1, 3, 8, 37, 16 experts | | NVFP4, Q8_0, BF16 | rel L2 2.0e-4 .. 2.2e-4 (the f16 rounding of the stored `act`) |
| full chain at prefill sizes vs numpy | n = 37, 308, 512 (128 experts, tokens grouped, tile 8 and tile 32) | NVFP4 synthetic with scales in the real byte range; Q8_0 and BF16 at n = 37 | portable: act 2.1e-4, dout 2.8e-7, y 2.9e-7; matrix: dout 6.9e-7 (gfx1201) / 2.6e-6 (gfx1151), y 1.0e-6 / 2.7e-6; Q8_0 matrix 2.1e-4 (f16 rounded weights) |

The oracle `y` gap of 1e-2 on NVFP4 is the oracle's own error: it quantises activations to Q8_0 for the dots. A numpy emulation of that (Q8_0 activations for x and act) lowers the host reference to 3.5e-3 against the oracle, so the kernels are closer to the exact math than the oracle is. Prompt B has only `l_out` and logits in its dump (no router tensors), so the 308-token shape is checked against numpy only. Slots 24..47 reuse the weights and scales of layers 6..17 (the per-pass scale re-share is the same tensor); slots 27 and 39 cover that.

Compile reports (loom-compile, both targets): 8 kernels, 0 spills, all entries compile for gfx1201 and gfx1151. VGPRs: gate_up 56, down 40, router_logits 40, matrix kernels 96 (LDS 27 KB, occupancy limited by LDS), WMMA instructions in the matrix kernels (32 each).

## Timings (device time of one dispatch, p50, `bench_experts.py`: input ring of 4 weight sets, flock -x, final batch profiled)

Floors: bytes / measured bandwidth (R9700 620 GB/s, 8060S 230 GB/s). NVFP4 layer 0 (real weights, real routing of prompt A; for n > 16 prompt A tiled with 5% noise). "Decode" reads 8 experts for n = 1.

| Kernel | n | bytes | floor R9700 / 8060S | R9700 | 8060S |
|---|---|---|---|---|---|
| router logits (F32 1.18 MB) | 1 | 1.18 MB | 1.9 / 5.1 us | 3.2 us | 5.1 us |
| select, routes, combine | 1 | | | 1.3, 1.0, 0.7 us | 1.4, 1.2, 0.7 us |
| gate_up NVFP4 (portable) | 1 | 23.9 MB | 38.5 / 104 us | 37.5 us | 119.9 us |
| down NVFP4 (portable) | 1 | 11.9 MB | 19.3 / 52 us | 19.6 us | 63.8 us |
| gate_up Q8_0 (layer 20, Q4_K_XL) | 1 | 45.1 MB | 72.8 / 196 us | 82.0 us | 226 us |
| down Q8_0 | 1 | 22.6 MB | 36.4 / 98 us | 34.5 us | 116 us |
| gate_up BF16 (layer 20, Q8_K_XL) | 1 | 84.9 MB | 137 / 369 us | 141 us | 454 us |
| down BF16 | 1 | 42.5 MB | 68.5 / 185 us | 69.7 us | 204 us |
| gate_up NVFP4 (portable) | 8 | 64 routes | | 204 us | 602 us |
| down NVFP4 (portable) | 8 | | | 119 us | 357 us |
| gate_up NVFP4 (matrix) | 512 | 382 MB, 43.5 GFLOP | 0.62 / 1.66 ms | 1.17 ms (37 TFLOP/s) | 3.62 ms (12 TFLOP/s) |
| down NVFP4 (matrix) | 512 | 191 MB, 21.7 GFLOP | 0.31 / 0.83 ms | 0.70 ms (31 TFLOP/s) | 1.80 ms (12 TFLOP/s) |
| gate_up NVFP4 (portable) | 512 | | | 4.84 ms | 11.0 ms |
| down NVFP4 (portable) | 512 | | | 3.77 ms | 8.5 ms |
| routes (atomic ranks, tile 32) | 512 | | | 12 us | 12 us |
| router logits | 512 | 1.18 MB + x | | 93 us | 157 us |

Decode on the R9700 is at the traffic floor (1.0x). The ring holds four copies of the layer weights (about 4 x 24 MB touched per dispatch, above the 32 MiB benchmark ring target but possibly partly resident in the large last-level cache), so treat the R9700 numbers as a lower bound of the cold time. On the 8060S decode is 1.15x (gate_up) and 1.23x (down) of the floor. Q8_0 and BF16 are 1.1x to 1.25x of the floor. One NVFP4 decode visit on the R9700 (router, select, routes, gate_up, down, combine) is about 63 us of kernel time against a 60 us floor; the 44 NVFP4 and 4 Q8_0 visits of a token total about 3.3 ms against the 2.98 ms expert floor of `ARCHITECTURE.md` section 12 (R9700). Launch gaps inside a command are not included.
The matrix kernels at n = 512 are 1.9x (gate_up) and 2.3x (down) of the traffic floor and about 30% (R9700) / 20% (8060S) of the f16 matrix peak: they are not tuned (single buffered K loop, one thread per stage row for the weight decode, 2 barriers per K block, scattered 2 B stores of `act`).

## Open problems and notes for inkH

- The loom-serve fold of a constant `wtype` is not checked (the command-program compile is not available in `loom-compile`). Fallback if needed: export one thin kernel per dtype over the same helper functions.
- Padded stages: `n` is the stage capacity; all rows are routed. Prefer the live count when the stage shape allows it.
- `dout` (37.7 MB at n = 512) can be removed by a combine epilogue in the down kernel (each route owner adds `w * value` into `y` with a per-token order); not done to keep the order of the 8 slots fixed.
- Tie behaviour of the router differs from `std::partial_sort` only on exact ties of the biased sigmoid.
- Vendoring: nothing from `hrx-loom-library` was copied (it was read for the route bucketing idea). The WMMA providers come from our PR1 motif (`motifs/wmma.loom`, renamed and reduced to the 2x2 tile).
- Possible M8 work: double-buffered K loop and register-direct NVFP4 decode in the matrix kernels, record tile 64 for n = 512, fewer waves in the routes kernel, folding `act` into f16 pairs on store.

## Changes by inkJ (integration, 2026-11-20)

- The weight type is passed as the exact launch scalar `@inkling_expert_wtype(layer)` (0 NVFP4, 1 Q8_0, 2 BF16); loom-serve folds it (all 48 visits compile, layers 20 to 23 use the dense arm with exact `command.parameter` footprints and no `.scale` tensors). The serve run of layers 20 to 23 is part of M5.
- `func.def` of `lib/inkling/experts_*.loom` and `nvfp4.loom` are `public`.
- Router and experts are two hooks (`router`: `router_logits` + `router_select`; `experts`: `routes`, `gate_up`, `down`, `combine`). `n` is the bucket token capacity (live rows are not known at plan time; rows beyond the live count are computed from stale data and ignored). Matrix path (tile 32, WMMA) from capacity 48, portable path (tile 8) below. Workspace (`tasks`, `act`, `dout`) is allocated per hook call. Results of router and the router weights travel through the shared buffers of the aliasing table in INTEGRATION_LOG.md (`small`, `ab`, `ob`, `gate`).
