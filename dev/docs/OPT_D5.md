# Lane D5: lm_head and sampler on the decode path (bead llama-cpp-tg3.27)

Result: the sampler moved from one workgroup to 128 workgroups per row (195 us to about 7 us per step). The lm_head in decode is already at the memory floor, so the other part of the lane has nothing to win. Gain on the R9700: +1.6 percent decode short (about -0.18 ms per step). This is at the stop threshold of the lane (1.5 percent), and the lane stops here.

## Findings

1. lm_head in decode is at the floor. The plan counted 414 GB/s from the 1.19 ms of the prefill-stage launch (cols=8 build, 512-token bucket). The decode stage runs the cols=1 build. Measured on the R9700 with a variant that cuts the lm_head launch to 8 rows (the weight parameter stays, a stage needs it): `inkling_gemv_q8_0` total 1405.2 ms to 1087.0 ms over 406 decode steps, so 0.784 ms per lm_head launch (490 MB, 625 GB/s) against a floor of 0.79 ms. Raw: `prof/nolm2` against `inkV/prof/r9700/dshort_d`. The "head 1.38 ms" of the plan is the apportionment error plus the sampler (0.195 ms).
2. The rest of the head hook is small: `head_select` 1.1 us, `head_norm` 2.8 us, embed gather 3.5 us. Folding them saves under 5 us (0.04 percent). Not done.
3. The sampler was the only item: `inkling_sample_rows` = 195 us per step on one workgroup of 256 threads (one CU of 64). For greedy rows it is one pass over 200,058 logits; for rows with a random draw it is two passes (maximum, then a Philox race per token).

## Change (candidate C1)

Files: `control/chat_sampler.loom`, head hook of `scripts/gen_hooks.py` (output `dispatch/hooks_dense.loom`), `lib/vendor/sampling.loom` (one word), `test/chat/test_sampler.py`.

- Three launches per step instead of one: `inkling_sample_stats` (maximum key and maximum logit per block of 1563 columns, 128 blocks per row), `inkling_sample_draw` (each workgroup reduces the 128 block maxima, then runs the exponential race over its block), `inkling_sample_rows` (combines the block results and writes the token). Scratch: `rows x 128 x 20` bytes from `buffer.alloca` in the head hook.
- Fast rows: greedy, top_k 1, and plain rows (top_k 0, top_p 1; temperature above 0). Other rows (top_k, top_p, nucleus) run the old template `sampling.categorical_excluding_f32` inside `inkling_sample_rows`, as before.
- The result does not change. Reductions are exact (maximum of keys, maximum of the race with the lowest ID on a tie), the race uses the same `sampling_race` function and the same counter (seed, position, token ID). Test: the fast rows draw the same 20,000 tokens as the template rows with top_k 200058 (0 of 20,000 differ, four option sets).
- `@sampling_race` is now `func.def public` in the vendored `sampling.loom` (and its copy under `test/chat/vendor`): the serve link needs a visible body for a call from another file (error LOWERING/044 callee_missing_body otherwise). Do the same edit when the vendor files are updated.
- The packet policy block moved into `@chat_sample_policy` (one function for the three kernels).
- Interface change: `inkling_sample_rows` has one more argument (`%scratch`); the two new kernels have the five arguments of `kernel.decl` in `gen_hooks.py` (`EXTRA_DECLS`).
- `test_sampler.py` was stale (it launched a 6-argument kernel with separate policy and identity buffers). It now builds the packet epoch layout (header word 2, identity records, option records) with `inkling.packet_capacity` set and links `lib/vendor/packet.loom`.

Compiler (gfx1201, `loom-compile --compile-report`): no spills in any of the three kernels; VGPR 10 / 19 / 37, occupancy 100 percent.

## Evidence

Kernel: HAL profile, decode short, R9700 (`prof/var-d5` against `inkV/prof/r9700/dshort_d`, greedy requests): `sample_rows` 195.4 us becomes stats 3.8 + draw 1.3 (skipped rows) + rows 2.1 = 7.2 us.

Numerics:

| Check | R9700 | 8060S |
|---|---|---|
| `test_sampler.py` (39 checks: determinism, support, chi-square, seeds differ, exclusion, fast rows equal template rows) | 39 of 39 PASS | 39 of 39 PASS |
| same-seed tokens equal across the two GPUs | 1.0000 for all nine option sets | |
| greedy 64 tokens, oracle prompts A, B, C through the chat route, d5 against base | identical | identical |
| head kernel tests, logits against the oracle | not affected (the head kernels and the logits are unchanged; the sampler reads them) | |

End to end ABBA (fresh server per arm, `ab_abba.sh`, order A B B A, 3 reps per session, medians; A = base `mat`, B = d5). R9700:

| Row | A (two sessions) | B (two sessions) | Change |
|---|---|---|---|
| decode short, tok/s | 86.35, 85.93 | 87.39, 87.67 | +1.6 percent (11.61 to 11.43 ms) |
| decode 8k | 77.28, 76.98 | 77.94, 78.28 | +1.3 percent |
| 4 streams (aggregate) | 201.09, 198.94 | 201.46, 201.61 | +0.7 percent (noise 2.5) |
| prefill 2k | 2219.8, 2210.7 | 2213.7, 2209.0 | -0.2 percent (noise 3) |

Session spread inside an arm is 0.4 percent for decode. The decode gain matches the 0.19 ms expected from the profile.

8060S (same method, `d5` against `mat-strix`, order A B B A):

| Row | A (two sessions) | B (two sessions) | Change |
|---|---|---|---|
| decode short, tok/s | 38.77, 38.72 | 39.09, 39.06 | +0.85 percent (25.82 to 25.60 ms) |
| decode 8k | 33.51, 33.58 | 34.03, 33.95 | +1.3 percent |
| 4 streams (aggregate) | 82.40, 82.31 | 81.82, 82.44 | -0.3 percent (noise 2.5) |
| prefill 2k | 877.9, 860.2 | 876.8, 880.8 | within the session spread (2 percent) |

The gain on the 8060S is smaller than the plan estimate (0.25 ms): about 0.22 ms per step measured. Decode short on the 8060S is +0.85 percent, below the 1.5 percent threshold, but the change is free and the same code runs on both GPUs.

## Per-arch rows

None. The block count (128) and block width (1563) are fixed by the vocabulary and a 256-thread workgroup; both GPUs use the same values. No tune row, no arch string.

## Stop reason

lm_head at 99 percent of the floor (nothing to tune); sampler at 3 percent of the old cost. The whole lane is worth 1.6 percent on the R9700, which is the stop threshold. Remaining head work (select, norm, embed, the three sampler launches) is about 20 us per step.

## How to rebuild

1. `python3 scripts/gen_hooks.py` (writes `dispatch/hooks_dense.loom`; the other hook files do not change).
2. Sampler test, both GPUs, under `flock -s`: `python3 package/test/chat/test_sampler.py amdgpu://GPU-e0880c05003c324d amdgpu://id/000000e2000000808e270f2b0bbb4e00` (loom tools on the path, numpy and scipy).
3. Patch: `lanes/D5.patch` (diff -ruN of `package` and `scripts`, a/ b/ prefixes).

## Notes for the integrator

- Conflicts: `scripts/gen_hooks.py` head section only (`head_hook`, `EXTRA_DECLS`); `hooks_dense.loom` head hook (the generated lines: scratch alloca and three launches). No other lane owns these lines. D1 owns the norm hooks in the same file; the hunks are far apart.
- The shared scripts `profile_run.sh`, `ab_run.sh` check `pgrep -af 'loom-serve|...'`. That matches the path `/home/user/loom-serve/.venv...` in the command line of any queued `flock ... loom-benchmark` job of another lane and stops with "another server runs". The D5 copies use `pgrep -x`.
- A server session started under `flock -s` can fail with error code 8 (resource exhausted) when another shared holder uses the GPU memory. Start servers under `flock -x`.
