# Wave 1 integration (agent integ1, bead llama-cpp-tg3.29)

Result: `package_T10b/` = `package_base_T10` + lanes D1, D4, D5 and P1 in one tree. It is the base for wave 2 (do not edit it). Merged patch: `lanes/wave1_merged.patch` (package files and the `scripts/` hunks, `a/` and `b/` prefixes, against `package_base_T10` and the pristine generator). The real `package/` is not touched; the promotion into `package/`, the publish rebuild, the gates, the card and the zip come after wave 2.

Verdict in one line: no conflict, all kernel tests pass on both GPUs, layer NMSE equals the base level, greedy output is token-identical except at documented near-ties of the base itself, and the combined end-to-end gain is larger than the sum of the lanes on the R9700 (decode +5.3 percent, prefill 2k +61 percent with `prefill_capacity` 2048) and positive on every row on the 8060S.

## Merge

Generator (`scripts/gen_hooks.py`): the pristine original is `scripts_base_D5/gen_hooks.py` (verified: `scripts_base_P1/gen_hooks.py` is identical). Lane D4 had edited `scripts/gen_hooks.py` in place; its diff against the pristine file is exactly the two hunks of the attention split rows (19 lines). Merge order on the pristine file with `git merge-file` (3-way, no conflict in any step): D1 (`scripts_D1/gen_hooks.py`), then D4 (in place edit), D5 (`scripts_D5/gen_hooks.py`), P1 (`scripts_P1/gen_hooks.py`). Result: `scripts_T10b/gen_hooks.py`. All four lanes touch disjoint regions: D1 the mix, norm and conv hooks and the kernel decl list, D4 the `inkling.tune.attn.split_blocks` rows and `max_splits`, D5 the head hook and `EXTRA_DECLS`, P1 two lines of the experts hook (`o264`, `chain(64, ...)`).

Program files: lane files copied over the base copy (none of the four lanes edits the same program file; `hooks_*.loom` and `kernel_decls.loom` are generated). Then `gen_kernel_decls.py` and `gen_hooks.py` regenerated the dispatch files. Checks:

| Check | Result |
|---|---|
| regenerated `hooks_attn.loom` vs D4's hand-edited file | identical (no diff) |
| `hooks_mix.loom` vs D1, `hooks_experts.loom` vs P1 | identical |
| `hooks_dense.loom` vs D1 / vs D5 | differs only by the D5 hunks (sampler scratch and three launches) / only by the D1 hunks (norm_cols, mode 1 norms) |
| `kernel_decls.loom` vs D1 | differs by the three D5 sampler decls only |
| `diff -rq package_T10b package_<lane>` for each lane | only files of the other lanes (and generated files with the other lanes' hunks) differ |

`scripts/gen_hooks.py` in the repo is now the merged generator (it was D4's in-place edit; the pristine original stays in `scripts_base_D5/gen_hooks.py`). `scripts/capture_all.py` got the one-line capture fix below. Both are in the merged patch. Tools of this agent: `scripts_integ1/` (ccheck, ktest_run with the lock shim, make_var, gsession, abba3, capture_job, lgap/lg_job, ab_session with the P1 metrics).

Capture layout (D1 finding): `x` (id 10) is not written in an uncontrolled local visit because `mix_states` mode 1 produces `xn` directly. Measured: at visit 1 (local) only `x` is BAD (nmse 8e7), every other tensor passes, including id 24 (`v`, nmse 2.3e-5): D1's note about id 24 does not hold (v is written by `attn_prep`). Fix: `capture_all.py` skips `x` for visits below 24 whose layer is local (`slot < 24 and not glob`); layout and ids unchanged (the generator is not touched: global visits and visit 0 keep their `x` compare). Controlled visits never compared `x`.

## Compile check (no GPU)

`scripts_integ1/ccheck.sh` links every kernel root with the area config and runs `loom-compile --compile-report` for gfx1151 and gfx1201, base package and merged package. 72 of 72 compiles status OK, 0 spills. Changes against base (VGPR, merged vs base): `inkling_attn_tile` 256 vs 256 (gfx1201), 251 vs 240 (gfx1151), occupancy 25 vs 18 percent; `mix_states` 68 vs 61 (gfx1201), 60 vs 60 (gfx1151); `experts_gate_up_wmma` and `down_wmma` 144 vs 96 (gfx1201), 168 vs 80 (gfx1151); `routes` 24 vs 18; `sample_rows` 37 vs 35, new `sample_stats` 10, `sample_draw` 19 (all arches, same as the lane reports). The three `gemv` roots do not link with the generic area config in either package (target error, same in base, unchanged files).

## Numerics

All runs on `package_T10b` (4 lanes). Kernel tests ran without nested locks (`scripts_integ1/ktest_run.py`: one outer `flock -s` per target, a no-op `flock` shim on PATH for the tests' own lock calls). Dense, attention, mix and chat targets ran on the three-lane tree: lane P1 changes only experts files, so they hold for the merged tree. Experts re-ran on the four-lane tree (n up to 2048).

Progress line (1): numerics done (kernel tests, capture layer checks both GPUs, chunking, greedy, compare). Progress line (2): ABBA done both GPUs.

### Kernel tests (28 adapter targets: experts 5, dense 11, attn 1, mix 7, chat 4)

| Area | gfx1201 | gfx1151 |
|---|---|---|
| experts (`test_gate_up` 12, `test_oracle` 140, rows variants, `test_prefill` n 37..2048) | 5 of 5 | 5 of 5 |
| dense (norm_embed incl. `rms_cols`, gemv x3, gemm x5, head, oracle_layers) | 11 of 11 | 11 of 11 |
| attn (`attn_test run` with `ATTN_LONG` set: 130 rows incl. long prompt 1 chunk, 3 chunks, 512+rest, decode, mixed) | 130 of 130 | 130 of 130 |
| mix: `test_mix` 299 cases, `test_conv` 342, `test_ctrl` 240 + 240 + 3 x 12 | 7 of 7, 0 FAIL | 7 of 7, 0 FAIL |
| chat (plain 84, echo 84, script 29, policy 37 cases, incl. the three-launch sampler path) | 4 of 4 | 4 of 4 |

Notes: the adapter's default `attn_test` call skips the long cases (the oracle path points into the kt copy); run with `ATTN_LONG=<oracle_long>/Q8_K_XL_L1`. `test_ctrl_q4_c128` failed once on the R9700 only because two adapters shared one scratch dir; the rerun passed (12 of 12). The chat suite starts servers outside the lock and shares build dirs and ports: run one GPU at a time under one `flock -x` with the shim. `test_sampler.py` (D5, 39 of 39 on both GPUs in OPT_D5) was not re-run separately.

### Capture route (layer NMSE against the Q8_K_XL oracle dumps)

Visits 0, 1, 5, 6, 17, 18, 23, 24, 29, 30, 47 on both GPUs, id 99, merged server at prefill_capacity 2048, base server at 512. Prompt A (16 tokens), B (308), long prompt (1332, ring wrap). NMSE ratio merged/base over every tensor and prompt (about 37 pairs per visit):

| visit | R9700 max ratio | 8060S max ratio |
|---|---|---|
| 0, 1, 5, 6 | 1.00 to 1.02 | 1.00 to 1.04 |
| 17, 18, 23 | 1.11 / 1.02 / 1.00 | 1.02 / 1.13 / 1.19 |
| 24, 29, 30, 47 | 1.06 / 1.01 / 1.00 / 1.03 | 1.02 / 1.27 / 1.13 / 1.11 |

The largest ratios are on tiny tensors (`sink`, `logits`); the typical ratio is 1.00. Absolute levels equal docs/INTEGRATION_LOG.md (visit 1 `l_out` A 3.0e-5, visit 5 7.0e-5; `y` of visit 6 on the long prompt 2.8e-3 = the known router top-8 flips). Deep loop visits exceed the 2e-4 default gate in base and merged alike (oracle runs the CPU Q8 path; e.g. long prompt `l_out` visit 47 7.0e-2 both), so the BAD flags are the same set in both.

Chunk independence (capture route, `l_out` NMSE per visit): prompt B whole, 103+103+102, 256+52, 304 then 4 single tokens; long prompt whole (1 chunk of 1332, only possible at capacity 2048), 1024+308, 512+512+308, 444 x 3, 1328 then 4 single tokens. The spread across chunkings is 1.000 to 1.002 (max/min) at every visit on both GPUs. Base was run at capacity 512 with A, B whole and 512+512+308 only. The attn kernel tests give prefill chunkings bitwise equal to 1 chunk (8060S 3 chunks 1e-8) and decode 6e-4..9e-4 as in base.

### Greedy 64 tokens, chat route (A, B, C, 8k document, 32k document)

`scripts_integ1/gsession.sh`; base = `inkV/mat-*` (capacity 512); merged at capacity 2048 (and 512 on the 8060S).

| GPU | A | B | C | 8k | 32k |
|---|---|---|---|---|---|
| R9700 | same | same | same | differs at char 118 | same |
| 8060S | same | differs at char 247 | same | differs at char 118 | differs at char 107 |

8060S determinism: base run twice (fresh server each) is identical on all five prompts; merged run twice is identical; merged at capacity 512 equals merged at 2048 (P1 is not the cause). Near-tie check with llama.cpp (CPU oracle build, Q4_K_XL, same rendered prompt, top-10 log-probs, `scripts_integ1/lgap.py`):

| Case | flip | llama.cpp log-probs | verdict |
|---|---|---|---|
| B, 8060S, char 247 | base " GP" (-2.22) to merged " stored" (-1.04, llama.cpp top) | gap 1.17, merged agrees with llama.cpp | near-tie; R9700 base already says " stored", merged 8060S now equals R9700 |
| 8k, char 118 | " about" (-1.04) and " set" (-1.35) swap | gap 0.31 to 0.38 | near-tie; the two GPUs already differ here in base |
| 32k, char 81 (base R9700 vs base 8060S) | " documentation" -1.05 / " content" -0.94 | gap 0.11 | the bases of the two GPUs already differ at a near-tie |
| 32k, 8060S, char 107 (base vs merged) | " Loom" to " docs" | "Loom" -0.0005, " docs" not in top 10 | not explained by llama.cpp; see below |

Reorder experiment: the BASE package with only the global decode split changed (hooks_attn: context above 16384 uses 10 blocks per split instead of 8, same kernels) flips at the same char 107 to the same text; with 12 blocks per split it is identical to base. A pure change of summation order of the base kernels flips this position, so the " Loom" / " docs" decision is within Loom's own float noise at 37k tokens (llama.cpp is sharper there: base Loom is flatter at this context, an existing difference). Verdict: variance, not a D4 defect; base is deterministic run to run. loom-serve has no logprobs (HTTP 400), so Loom's own gap could not be read.

Skills `compare.py` (golden.json + golden2.json incl. the 18k document, 23 prompts, cached llama.cpp recordings, `LOOM_KIT` set, no llama.cpp server): 8060S base 12 full matches, 0 bad, max gap 0.296, PASS; merged 14 / 0 / 0.461 PASS (same at capacity 512). R9700 base 12 / 0 / 0.296 PASS; merged 12 / 0 / 0.582 PASS. consistency, agent, ctx gates not run (final integration).

## End-to-end ABBA (fresh server per arm, order A B C C B A, 3 requests per metric, mean of the two arms; tok/s)

A = base (capacity 512), B = merged at 2048, C = merged at 512 (isolates P1). `scripts_integ1/abba3.sh`. Floor in ms (lower is better).

R9700 (gfx1201):

| Row | Base | Merged 2048 | gain | Merged 512 | gain | Lane gains |
|---|---|---|---|---|---|---|
| decode short | 85.91 | 90.44 | +5.3 % | 90.42 | +5.2 % | D1 +2.9, D5 +1.6, D4 +0.25 |
| decode 8k | 76.69 | 80.68 | +5.2 % | 80.78 | +5.3 % | D1 +2.5, D5 +1.3, D4 +0.7 |
| decode 32k | 67.82 | 71.81 | +5.9 % | 71.97 | +6.1 % | D1 +2.2, D4 +2.4 |
| prefill 1k | 2038 | 2452 | +20.3 % | 2141 | +5.1 % | P1 +21 |
| prefill 2k | 2211 | 3561 | +61.1 % | 2416 | +9.3 % | P1 +55, D4 +5.5 |
| prefill 8k | 1934 | 3123 | +61.5 % | 2181 | +12.8 % | P1 +52, D4 +10.2 |
| 4 streams | 199.7 | 206.3 | +3.3 % | 206.6 | +3.4 % | D1 +1.5, D4 +1.2 |
| floor (ms) | 55.0 | 54.7 | -0.6 % | 54.4 | -1.2 % | |

8060S (gfx1151):

| Row | Base | Merged 2048 | gain | Merged 512 | gain | Lane gains |
|---|---|---|---|---|---|---|
| decode short | 38.61 | 39.50 | +2.3 % | 39.64 | +2.7 % | D1 +1.8, D5 +0.85, D4 +0.3, P1 -0.4 |
| decode 8k | 33.56 | 34.91 | +4.0 % | 35.06 | +4.5 % | D4 +2.2, D1 +1.8, D5 +1.3 |
| decode 32k | 27.82 | 29.77 | +7.0 % | 29.89 | +7.4 % | D4 +5.1, D1 +1.6 |
| prefill 1k | 803 | 828 | +3.2 % | 789 | -1.7 % | P1 +4.7 (2048) |
| prefill 2k | 870 | 1154 | +32.7 % | 891 | +2.4 % | P1 +31.4 |
| prefill 8k | 782 | 1037 | +32.7 % | 808 | +3.4 % | P1 +28.8 |
| 4 streams | 81.7 | 82.3 | +0.8 % | 82.6 | +1.1 % | |
| floor (ms) | 135.4 | 134.0 | -1.0 % | 134.6 | -0.6 % | |

Guard rows from P1: its 8060S decode loss (-0.4 to -1.5 %) is gone in the merged build (all decode rows positive). The 8060S 1k prefill at 2048 is not below base (+3.2 %) and stays the weakest prefill row; at capacity 512 it is -1.7 % but the two base arms differ by 3.5 % (817 / 789), so that row is inside noise. Prefill 32k not run. `memv` is not usable (R9700 24.9 to 25.1 GB in all arms, 8060S counter 0.06): P1's +0.43 GB is not re-measured. Cold start: stage compile time grows with capacity (P1: 41 to 75 s R9700, 75 to 135 s 8060S, summed over workers), first start after a program change only.

## Issues and open problems

- 8060S 32k greedy differs from base at char 107 (variance, above); 8060S prompt B and the 8k prompt on both GPUs differ at documented near-ties. Base Loom looks flatter than llama.cpp at 37k context (existing).
- `prefill_capacity` 2048 must be set in the card preset by the integration step. Q8 presets not measured.
- Attn kernel tests cover at most 21 blocks per visit, so D4's dynamic split (up to 64 splits) is covered only by greedy and capture checks.
- consistency, agent, ctx and sampling gates not run here.
- Tool findings: `compare.py` needs `LOOM_KIT` and runs without `python -I`; ktest_adapter nests locks and misses the lane-hardcoded work path of P1's experts tests (`~/inkling/optP1/work`, redirected in the box copy `ktest_run.py`); chat_run_all starts servers outside the lock.

## Scratch left on the box (not deleted)

`~/inkling/integ1/` (package copies; `var-*` model dirs are hard links of `inkV/mat-*`, no weight copy; real data small: `out/`, `ablogs/`, `W/`, capture txt files), `~/inkling/integ1base/`, `~/inkling/i1kt_r9700/` and `~/inkling/i1kt_strix/` (1.5 GB each), `~/inkling/tools/integ1-venv` (symlink). Disk free at the end: 42 GB.
