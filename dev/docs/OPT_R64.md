# Lane R64: wave64 ("row64") matrix kernels for gfx1201 (experiment, 2026-11-23)

Question (owner): does a wave64 matrix-op schema on the R9700 (gfx1201) speed up prefill? Base: `package_M30` (merged wave 2 on hrx-loom 0.3.0). Work copy `package_R64` (laptop), `~/inkling/optR64` (box). Patch: `lanes/R64.patch` (`diff -ruN` of `package_M30` and `scripts_M30` against the R64 copies). Decisions: `bench/decisions_R64.jsonl`. Nothing was committed, published or copied into `package/`.

## Result

The wave64 kernels are correct and the template route works, but they are not faster. Do not promote.

- Dense Q8_0 GEMM (production tile, 128 rows x 64 columns): equal to wave32 within 2 percent in isolation, 3.6 percent faster per launch in a serve profile, which is 0.35 percent of the prefill step. End to end ABBA on the R9700: prefill 1k, 2k, 8k are 0.6 to 1.0 percent lower than the M30 build, inside the 3 percent drift of the session. Decode and 4 streams are equal.
- Dense wide tile (128 x 128): equal to the wave32 wide kernel at n = 512 (109.5 against 111.0 us).
- Experts down (prefill matrix path): 13 percent (n = 512) and 17 percent (n = 128) slower. Experts gate_up was not ported.
- Why: the GEMM kernels are limited by weight decode, staging through workgroup memory and barriers, not by the matrix layout. Wave64 uses the same number of lanes per SIMD (the occupancy limit is the workgroup memory), the 4-value fragments halve the register cost of a tile but add no throughput. In the serve profile of a 2k prefill the Q8_0 GEMM is 10 percent of device time (343 of 3452 ms) and the two expert matrix kernels are 31 percent (1063 ms), so even a large dense gain is small.

## What was built

All files are gfx1201 only through the per-arch route of `docs/TUNE_ROUTE.md` (no arch strings).

- `lib/inkling/targets_w64.loom`: `amdgpu.target<gfx12-generic> @inkling_wave64_gfx12 {subgroup_size = 64}` (the gfx11 alias of the W64 lane stays).
- `lib/inkling/dense_wmma64.loom`: providers `@inkling.dense.mma64.{zero_acc,k64_2x4,k64_4x4}` (`requires [#target.subgroup.size<64>]`, target `@inkling_dense_gfx12`, 4-value `vector<4xf16>` operands and `vector<4xf32>` accumulators) and `inkling_dense_mm_wmma64_store`. The operand fragments use `vector.fragment.load<lhs|rhs>` with the compact carrier (256 / subgroup size values), as the 0.3.0 library matrix motif (`motif/tensor/matrix.loom`) does. No lhs row permutation is needed.
- `kernel/dense/gemm_q8_0_r64.loom`: `inkling_gemm_q8_0_r64`, the same ABI and staging as `inkling_gemm_q8_0`, 4 waves, each wave 64 weight rows x 32 columns (2 x 4 fragments). The weight decode, activation load and the epilogue are shared with the wave32 kernel.
- `kernel/dense/gemm_wide_q8_0_r64.loom`: `inkling_gemm_wide_q8_0_r64` (4 waves, 64 x 64 each). Not wired into the dispatch (the wave32 wide kernel is not used either).
- `lib/inkling/experts_wmma64.loom`, `kernel/experts/down_wmma_r64.loom`: `inkling_experts_down_wmma_r64` (2 waves, 64 rows x 64 routes each, 16 fragments). Not wired into the dispatch.
- `scripts/gen_hooks.py`: template `@inkling.tune.dense.gemm_wave` (default 32, `target(@inkling_rdna4)` 64) in the dense hook file; `linear()` launches `inkling_gemm_q8_0_r64` when it returns 64. `dispatch/hooks_dense.loom` and `kernel_decls.loom` are regenerated. On other targets the branch is dead and the wave64 kernel is not compiled (checked on the 8060S).
- `card.json`: the two library lists get `dense_wmma64.loom` and `gemm_q8_0_r64.loom`.
- Tests and tools: `test_gemm.py` takes `gemm_q8_0_r64` and `gemm_wide_q8_0_r64`; `test_prefill.py` takes `PATHS=wmma_r64`; `bench_dense.py`, `bench_experts.py` (`WMMA=1 R64=1`); `ab_session.py` gets the metric `pf1k`; scripts `r64_*.sh`, `mk_mat_r64.sh`.

### Fragment layout found (gfx1201, wave64)

The result fragment has 4 values per lane. Row (weight row, the rhs column): `row + lane % 16`. Column (token, the lhs row): `col + 8 * (lane / 16 % 2) + 4 * (lane / 32) + e` for element `e`. The guess `col + 4 * (lane / 16) + e` gave nmse 0.97; a comparison of the output with the wave32 kernel showed lane groups 1 and 2 swapped. The operands are loaded by the fragment loads, so the staging in workgroup memory is unchanged. This is a result of the library layout, not a permutation that the program applies to the lhs.

## Evidence

Compile report (gfx1201, `scripts/r64_cr.sh`): `gemm_q8_0` 132 VGPR, subgroup 32, 8 resident waves, occupancy 50 percent, 16 matrix instructions; `gemm_q8_0_r64` 106 VGPR, subgroup 64, 4 resident waves (the same 256 lanes per SIMD), occupancy 25 percent of the wave32 count, 32 matrix instructions; no spills in both; limited by workgroup memory in both.

Isolated kernel time (device, p50, `loom-benchmark` dispatch_complete, exclusive lock, 2 repeats, spread under 1 percent, attn_q 3072 x 2304 Q8_0):

| n | gemm_q8_0 (w32) | gemm_q8_0_r64 | wide w32 | wide r64 |
|---|---|---|---|---|
| 16 | 28.7 | 30.5 | 39.2 (66.0 in one run) | 41.0 |
| 128 | 51.5 | 48.8 | 52.2 | 55.2 |
| 308 | 89.0 | 87.7 | 101.9 | 100.9 |
| 512 | 130.9 | 128.8 | 111.5 | 109.5 |
| attn_o 512 | 139.5 | 137.0 | | |

Experts down, layer 0 NVFP4 weights, device p50 (us; two A and two B runs, alternating): n = 512 w32 614.0, 611.9, w64 695.9, 691.4; n = 128 w32 412.6, 412.5, w64 482.9, 482.0. gate_up (wave32 in both arms) 1020 to 1026 at n = 512.

Serve profile of a 2k prefill (R9700, HAL dispatch statistics, `scripts/r64_prof.sh`, 3 requests): `inkling_gemm_q8_0` 1728 launches, average 198.2 us, total 342.6 ms; `inkling_gemm_q8_0_r64` 191.1 us, 330.2 ms (-3.6 percent per launch, -12.4 ms of a 3452 ms step); expert gate_up 2.79 ms and down 1.65 ms for 240 launches, equal in both arms.

End to end ABBA, R9700, preset `chat`, prefill capacity 2048, fresh server per arm under the exclusive lock, 3 requests per metric and median per session (`scripts/r64_seq.sh`, log `~/inkling/optR64/ab_log.jsonl`, tags `r64a.r9700.*`). A = `package_M30` build (`~/inkling/M30/mat-all`), B = R64 build (`mat-R64`: dense GEMM only). Order A B B A.

| Row | A sessions | B sessions | B against mean of A |
|---|---|---|---|
| prefill 1k, tok/s | 2649.0, 2571.5 | 2585.2, 2584.1 | -0.9 percent |
| prefill 2k | 3817.6, 3714.4 | 3746.4, 3738.8 | -0.6 percent |
| prefill 8k | 3346.8, 3262.9 | 3286.6, 3281.7 | -0.6 percent |
| decode short | 95.11, 94.59 | 95.26, 95.20 | +0.4 percent |
| decode 8k | 84.90, 84.75 | 84.55, 84.91 | 0 |
| 4 streams, aggregate | 214.7, 214.1 | 213.4, 212.6 | -0.6 percent |

The two A sessions differ by 2.7 to 3.1 percent in prefill (the second one is lower: drift of the eGPU); the B sessions agree within 0.3 percent. A profiled single session pair (`prof.A` 3827.9, `prof.B` 3866.8 tok/s at 2k, B first with a warm GPU) points the other way, so the sign of the difference is not established; the magnitude is below 1 percent. Stop rule (win above 2 percent on prefill rows): not met.

The lock was free for every run; the lock wait did not matter. Total time about 3 hours, most of it kernel writing and the ABBA sessions (about 25 minutes).

## Numerics

- Kernel tests, gfx1201: `test_gemm.py - gemm_q8_0_r64` 11 of 11 (nmse 8.4e-8 to 8.7e-8, oracle relative L2 5.1e-3 and 4.5e-3, padded rows untouched), `gemm_wide_q8_0_r64` 11 of 11, `gemm_q8_0` (production wave32, still built) 11 of 11. `test_prefill.py` `PATHS=wmma_r64` (gate_up wave32, down wave64) n = 37, 308, 512, all three weight types: PASS, dout 6.9e-7 for NVFP4 (equal to the wave32 path), act 2.1e-4. `PATHS=wmma` unchanged and PASS.
- 8060S: `test_gemm.py gfx1151 - gemm_q8_0` 11 of 11 with the wave64 files linked; the card with the template builds and runs, the wave64 kernel is not launched (dead branch). Greedy A, B, C (64 tokens) through the chat route identical to the M30 build (`~/inkling/M30/res/gl_all030_strix.json`).
- R9700: greedy A, B, C, the 8k and the 32k documents identical to the M30 build (`gl_all030_r9700.json`), 64 tokens each, serving the R64 card with the wave64 kernel (the serve profile lists `inkling_gemm_q8_0_r64`, 1728 launches).
- Not run: the capture-route layer check (blocked on 0.3.0, see `docs/OPT_M30.md`), the full adapter (only the changed areas), chunk independence (no state change).

## Notes for the owner

- Side finding (not part of the experiment): the wave32 `inkling_gemm_wide_q8_0` is 15 percent faster than the production tile at n = 512 (111 against 131 us) and slower below n = 308. The M30 dispatch never selects it. At prefill capacity 2048 the GEMM share is 10 percent of the step, so the gain is about 1.5 percent of prefill at best. A window rule (wide for a full 512-row chunk) is a separate candidate.
- The expert matrix kernels (31 percent of the prefill step) and the attention prefill are the larger buckets; a wave64 layout does not change them (down: slower).
- The experts gate_up port was skipped because the down port, the simpler of the two, was 13 to 17 percent slower; a different wave64 tile (4 waves, 32 rows each, staging redesigned) is untested.

## Rebuild and repeat

1. `sh scripts_R64/r64_sync.sh` (laptop) copies the lane tree to `~/inkling/optR64` and rebuilds the kernel-test tree `kt`. On the box: `cd ~/inkling/optR64/scripts && python3 gen_kernel_decls.py && python3 gen_hooks.py` (the generators write to `../package/dispatch`).
2. Kernel tests: `bash scripts/r64_dt.sh dense test_gemm.py gfx1201 - gemm_q8_0_r64`; `PATHS=wmma_r64 bash scripts/r64_dt.sh experts test_prefill.py gfx1201 37 308 512`.
3. Timing: `bash scripts/r64_bench.sh q8_0`, `bash scripts/r64_ebench.sh 512`.
4. Build: `bash scripts/mk_mat_r64.sh R64 ~/inkling/optR64/package`; ABBA: `bash scripts/r64_seq.sh r9700 r64a "pf1k,pf2k,pf8k,dshort,d8k,c4" 3 ABBA A=$HOME/inkling/M30/mat-all:$HOME/inkling/tools/M30-venv B=$HOME/inkling/optR64/mat-R64:$HOME/inkling/tools/M30-venv`; greedy: `GREEDY_LONG=1 ABROOT=$HOME/inkling/tools flock -s ~/hrx-pr1/gpu.lock bash scripts/r64_greedy.sh r9700 <mat dir> <out.json> $HOME/inkling/tools/M30-venv`.

## Leftovers on the box

`~/inkling/optR64` (package, scripts, `kt` test tree, `mat-R64` hard links with no weight copy, `res/`, `ablogs/`, `work/cache` with layer 0 expert weights as npy files, `ab_log.jsonl`). `/tmp/r64cr` holds compile-report scratch. Nothing was denied.
