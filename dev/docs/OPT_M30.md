# Lane M30: hrx-loom 0.3.0 move and wave-2 re-merge (agent M30, 2026-11-22)

Base: `package_T10b` (wave 1 merged, published). Work copy `package_M30` (laptop), `~/inkling/M30/package` (box). Patch: `lanes/M30.patch` (`diff -ruN package_T10b package_M30` followed by `diff -ruN scripts scripts_M30`). Decisions: `bench/decisions_M30.jsonl`. Candidate for promotion: `package_M30` (= `package_T11` candidate: T10b on 0.3.0 plus D2, D3, D6, W64).

Owner orders that shaped the run: no 0.2.0 reference runs (the first reference kernel-test and A/B jobs on 0.2.0 were stopped by PID; the 0.2.0 greedy references that had already finished are used below), merge the four wave-2 lanes without extra gating debates, measure the merged build once per GPU against the published wave-1 numbers, then the gates.

## Result

- hrx-loom 0.3.0 needs no source change in our package. The unmodified wave-1 package (`package_T10b`) registers and runs on 0.3.0 and gives token-identical greedy output (prompts A, B, C, 64 tokens) to 0.2.0 on both GPUs.
- The requested `offset` to `index` rename is NOT applied: it breaks the build on 0.3.0 (details below). The 0.3.0 library itself still declares `offset` types.
- The gfx12 lhs row permutation in the wave32 matrix providers must stay: removing it makes the R9700 bf16 WMMA test fail (NMSE about 1.6). The 0.3.0 layout change is wave64 only; we have no wave64 matrix kernel.
- D2, D3, D6 and W64 apply to 0.3.0 (one hand merge in `scripts/gen_hooks.py`), compile with 0.3.0 (D3's ballot, broadcast, cttz and alloca included), pass their own kernel tests on both GPUs (33 of 33 targets of the adapter on each GPU, 4 of them new) and give greedy output identical to the wave-1 build on 0.2.0 on both GPUs for A, B, C and the 8k and 32k character documents.
- Merged build on 0.3.0 against the published wave-1 numbers: R9700 decode short 94.86 tok/s (+5.2 percent over 90.2), 8060S 42.80 (+7.8 percent over 39.7). Against the ceilings of the Q4_K_XL file (154.6 and 57.3): 0.61 and 0.75.

## What 0.3.0 required

Install (new venv, `~/loom/.venv` untouched): `~/loom030/.venv` with `pip install "hrx-loom[hf]==0.3.0"` (loom-serve 0.3.0, commit e2c41a72), plus numpy 2.5.5, gguf 0.19.0 and requests 2.34.2 (the same versions as the kernel-test venv of the earlier lanes; the tests need them). `~/loom030/library` is a shallow clone of ROCm/hrx-loom-library tag v0.3.0 (commit 279afe36), `~/loom030/.tools` links to the tools venv of `~/loom`, `~/loom030/loom-install.json` records hrx_loom 0.3.0 with the sha256 of the downloaded wheel (the card tools read it). Lane tool path without the string loom-serve: `~/inkling/tools/M30-venv`.

| Item | Finding |
|---|---|
| `offset` to `index` | Tried a bulk rename of every standalone `offset` token (1,897 lines in the package, the vendored library files, the tests and the generators; the regenerated dispatch files matched). Registration then fails: `prepare.loom:140:55: TYPE/001: 'lhs' type offset does not match 'type annotation' type index`. `buffer.length` and the other buffer and view operations still return the `offset` type in 0.3.0, so a mixed program is rejected. The library tag v0.3.0 (the matrix and ggml format motifs, the Qwen3.8 model programs including `command.launch ... : [offset]`) still uses `offset` in declarations. The unmodified T10b sources compile and run on 0.3.0. The renamed tree is kept as a scratch copy (`scratch_M30_offset_rename_attempt/`, not part of the patch). If the notes mean a specific API, it is not one that we call. |
| Pins and cards | `package/card.json` and `model.json` carry no hrx-loom version pin; nothing to update. `lib/VENDORED.md` still says the vendored library files come from tag v0.2.0: unchanged and still valid (the files compile on 0.3.0). The v0.3.0 library versions of `packet`, `feedback`, `epoch`, `storage` and `sampling` differ from the v0.2.0 copies (for example about 700 changed lines in `epoch.loom`); we did not re-vendor. |
| Card and README tooling | `scripts/make_card.sh` with `LOOM_ROOT=~/loom030` on a copy of the published tree: info.json, README.md and REPORT.md written, `schema: loom-card/3 errors: 0`, `render_card.py --check` OK, scrub PASS. `loom-fetch` (0.3.0) of the presets `chat` and `chat-r9700` with `--materialize=share`: 70 of 70 files each. The recorded SDK identity comes from `$LOOM_ROOT/loom-install.json` (created for 0.3.0). Not re-run: the Q8 presets and the gates evidence step (inputs from wave 1). |
| Controller contract version | Our card keeps the old controller contract version 1; it registers and serves on 0.3.0 without change (no contract change made). |
| Tooling | `ab_run.sh` and `m30_greedy.sh` now stop when the registration is `failed` (they used to wait 15 minutes), the guard uses `pgrep -x` (exact process name), both use the preset `chat` on both GPUs (the materialized wave-1 model has only that preset). `serve.sh` (capture route) adds the two W64 files when they exist. `ktest_adapter.py`: work path rewrite for the `opt*` lanes, new targets `test_gemv_group`, `test_router_decode`, `test_ctrl_msg` and the two `CTRL_FOLD=1` runs of `test_ctrl` (only when the lane files exist), the chat suite uses the 0.3.0 venv. |
| Stale processes | An orphan loom-serve of the first (numpy-less) kernel-test attempt and a failed-registration server of my own were stopped by PID. |

### gfx12 compensation (searched: permutation and transposed-load workarounds)

Found in two places, both wave32 providers (`requires [#target.subgroup.size<32>]`, target gfx12-generic):

- `lib/inkling/dense_wmma.loom`: `inkling_dense_mma_lhs_lane_gfx12` (lhs row order that puts the gfx12 result in the gfx11 layout), used by the providers `k64_2x2`, `k64_4x2`, `k64_1x2`.
- `lib/inkling/experts_wmma.loom`: `inkling_mma_lhs_lane_gfx12`, used by `k64_2x2`.
- `lib/inkling/attn_matrix.loom` has no permutation (it uses `target.subgroup.size` and wave32 only, "tested with wave32 only" in `KERNELS_attn.md`).

Test on 0.3.0, gfx1201: with the unchanged permutation all matrix tests pass (dense `gemm_wmma_bf16` 13 cases, experts `test_prefill`, the oracle layer tests). With the permutation replaced by the identity row map (`row = lane mod 16`) `test_gemm.py gfx1201 - gemm_wmma_bf16` gives 13 cases, 12 failed, NMSE 1.2 to 1.7. So the 0.3.0 fix does not change the wave32 gfx12 layout and the compensation stays. Nothing was removed. A wave64 matrix kernel for gfx1201 would use the new ISA-defined layout without a permutation (no such kernel exists in the package; the W64 lane found no wave64 win on gfx1201).

## Verification of the port (package_T10b unmodified, 0.3.0)

| Check | gfx1151 (8060S) | gfx1201 (R9700) |
|---|---|---|
| Greedy A, B, C, 64 tokens, 0.3.0 against 0.2.0 (`res/g_t10b020_*`, `res/g_t10b030_*`) | identical | identical |
| Kernel tests, adapter, first 17 targets (experts 5, dense 11, mix `test_mix`) | 17 pass, 0 fail | 17 pass, 0 fail |

The reference run was stopped after `test_mix` by the owner order; the merged run below is the full suite.

## Merge of the wave-2 lanes

Order D2, D6, W64, D3. Each patch was applied with `scripts/apply_lane.pl` (splits the lane patch per file, maps the lane paths, skips the generated `dispatch/*` files which are regenerated). Dry runs of all four against the T10b text: 0 failures. Applied in sequence: one conflict.

| Lane | Merge | Own tests on 0.3.0 | Greedy | Decision |
|---|---|---|---|---|
| D2 attention front end and projection group | clean | `test_gemv_group` 6 of 6, `attn_test` 130 of 130 (long cases on, chunk invariance as before) on both GPUs | merged build identical (A, B, C, 8k, 32k) both GPUs | keep |
| D6 experts decode pipelining | clean | `test_oracle` (3 variants), `test_prefill`, `test_gate_up`: pass both GPUs | same | keep |
| W64 wave64 down kernel (gfx1151 only) | clean (card.json, two new files, generator) | no area test (bit-identical by the lane); the wave64 kernel runs on 0.3.0 and the greedy output is identical on the 8060S; the R9700 links the dead gfx11-only kernel and runs | same | keep, gfx1151 only |
| D3 decode controllers and router | one conflict in `scripts/gen_hooks.py` `experts_file` (`chain` helper and the decode branch, against W64's `chain`). Hand merge: `chain(tile, gu_k, dn_k, routes=True)` keeps W64's tuple for `dn_k`; the decode branch (`router_decode`, then the chain without the routes launch) sits inside the `wide` else-branch. The generated `hooks_mix.loom` equals D3's own file; `hooks_experts.loom` equals D3's plus the W64 hunks. | `test_router_decode` pass, `test_ctrl_msg` 160 of 160, `test_ctrl` (chunk 128, 256, q4 64/128/256; plus both `CTRL_FOLD=1` runs 240 of 240) pass on both GPUs. The D3 primitives (`kernel.subgroup.vote.ballot`, `kernel.subgroup.broadcast`, `scalar.cttzi`, workgroup `buffer.alloca`) compile and run on 0.3.0. | same | keep |

No candidate was dropped: none failed numerics and none lost under 0.3.0 in the merged measurement. Lane-by-lane end-to-end re-measurement under 0.3.0 was not done (owner order: one merged measurement); the lane docs give the per-lane gains on 0.2.0.

Full adapter run of the merged package (`kres/all_<gpu>/kernel_tests.json`): 33 of 33 on each GPU (experts 6, dense 12, mix 10, attn 1 with 130 rows, chat 4 variants; 2 targets skipped as before: the capture-route scripts).

Greedy of the merged build (`res/gl_all030_*` against `res/gl_t10b020_*`, one fresh server each, 64 tokens): A, B, C, L8k (9,071 prompt tokens), L32k (36,995) all IDENTICAL on both GPUs. The 8060S 32k flip seen in wave 1 did not occur.

## Measurement (merged build on 0.3.0, once per GPU)

Method: `scripts/m30_seq.sh` (fresh server per arm under the exclusive lock, `ab_session.py`, 3 requests per metric, median, preset `chat`, prefill capacity 2048). B = `package_T10b` on 0.3.0 (the free 0.3.0 effect without lanes), C = merged on 0.3.0. Published = `docs/PERF_BASELINE.md` (wave 1, 0.2.0, `perf_session.sh`). The published prefill and 4-stream rows use a different measuring script, so for those rows the same-tool wave-1 arms of the lane docs (T10b on 0.2.0, `ab_session.py`: R9700 pf2k 3556 to 3561, 4 streams 205.2 to 206.3; 8060S pf2k 1116 to 1191, 4 streams 82.2 to 83.9) are the reference.

| R9700 (gfx1201) | published wave 1 | B: T10b on 0.3.0 | C: merged on 0.3.0 | C vs published | C vs B |
|---|---|---|---|---|---|
| decode short | 90.2 | 87.45 | 94.86 | +5.2 % | +8.5 % |
| decode 8k | 80.3 | 78.41 | 84.77 | +5.6 % | +8.1 % |
| decode 32k | 71.3 | 70.12 | 75.38 | +5.7 % | +7.5 % |
| prefill 2k (pf2k) | same tool 3556 to 3561 | 3550 | 3704 | +4.0 % | +4.3 % |
| 4 streams | same tool 205.2 to 206.3 | 202.66 | 213.22 | +3.7 % | +5.2 % |

| 8060S (gfx1151) | published wave 1 | B: T10b on 0.3.0 | C: merged on 0.3.0 | C vs published | C vs B |
|---|---|---|---|---|---|
| decode short | 39.7 | 39.48 | 42.80 | +7.8 % | +8.4 % |
| decode 8k | 34.9 | 35.09 | 37.81 | +8.3 % | +7.8 % |
| decode 32k | 29.7 | 30.56 | 32.44 | +9.2 % | +6.2 % |
| prefill 2k (pf2k) | same tool 1116 to 1191 | 1196 | 1221 | about +3 % | +2.1 % |
| 4 streams | same tool 82.2 to 83.9 | 83.69 | 88.06 | +5.4 % | +5.2 % |

Run spread inside an arm: under 0.5 percent for decode, under 1.3 percent for 4 streams (`ab_log.jsonl`). Single arm per build and GPU: no ABBA, so session-to-session variance (about 1 percent, more on prefill) is not separated from the effect.

Reading:

- The free 0.3.0 gain on T10b is not visible. 8060S: B equals published within 0.5 percent at short and 8k context, +2.9 percent at 32k, prefill and 4 streams equal to the old same-tool arms. R9700: B is 3.1, 2.4 and 1.7 percent BELOW published at short, 8k, 32k; prefill 2k equals the old arms (the RDNA4 layout change does not show: our matrix kernels are wave32). A second session (tags `m1b.*`) reproduced it: B 87.45, 78.57, 70.25 (decode short, 8k, 32k), prefill 3553, 4 streams 202.2; and the merged build again 95.17, 84.85, 75.41, 3701, 212.7 (within 0.4 percent of the first session). So the R9700 deficit of T10b on 0.3.0 is not a slow session. Whether 0.3.0 causes it cannot be said without an A arm on 0.2.0 (not run, owner order); the merged build shows no such deficit against the lane sums.
- The merged build gains +8.5 percent (R9700) and +8.4 percent (8060S) over B at short context. The lane gains on 0.2.0 add up to about +4.9 percent (R9700: D2 2.4, D6 1.4, D3 1.1) and +7.7 percent (8060S: D2 4.1, D6 1.5, W64 1.0, D3 1.1). The 8060S matches; on the R9700 the merge is 3 percent above the sum, which fits a B arm that is itself low by about 3 percent.
- Distance to the roofline ceiling (Q4_K_XL file, `docs/ROOFLINE.md`): R9700 94.86 of 154.6 (0.61), 8060S 42.80 of 57.3 (0.75). The remaining gap is 4.4 ms on the R9700 (10.5 ms per token against 6.5 ms) and 6.0 ms on the 8060S.

## Capture-route layer check: blocked on 0.3.0

`scripts/capture_all.py` talks to a `sequence` deployment (`dispatch/sequence.loom`, serve route `scripts/serve.sh` with `INKLING_CAPTURE`). On 0.3.0 that deployment does not register, on both GPUs, at every visit tried (0, 5, 24, 47): `registration failed ... "invocation argument count mismatch"` (`res/capture_job_*.out`). The v0.3.0 library moved its text controller to `text_epoch_v2` (token count in place of the shape id; `text_epoch_v1` is gone from the library). Our text route keeps the old `_v1` controller names and works (all runs above use it); the sequence kind has no such compatibility names, and our vendored `epoch.loom` and `dispatch/sequence.loom` still have the 0.2.0 argument lists. Fixing it means porting the sequence control and the vendored epoch to the 0.3.0 contract (not attempted: it touches the code shared with the text route). The capture-route layer check and the chunk-independence runs through it were therefore NOT run for the merged build. Substitutes that were run: the kernel tests (the attention and mix tests compare with the oracle dumps and check chunk invariance: prefill with 3 chunks and 512+rest bitwise equal to 1 chunk on the R9700 and within 1e-8 on the 8060S local visits, decode against 1 chunk 6e-4 to 9e-4, all as in wave 1), the greedy checks including the 8k and 32k documents, and the compare gate against the CPU llama.cpp oracle. The serve route (`serve.sh`) has the W64 files added for later use.

## Gates (LEVEL=dev, once per GPU, merged build on 0.3.0)

`scripts/gates_job.sh` (`run_gates.sh` stages serve, gates, extra) against the materialized merged model `mat-all`, exclusive lock, evidence in `bench/M30/` (`gates_<gpu>.json`, `context_admit_complete_<gpu>.json`). The CPU llama.cpp oracle (`scripts/oracle_server.sh`, started for the second GPU and for the R9700 re-run, stopped by PID afterwards) answered the cache misses.

| Gate | R9700 (preset chat, prefill 2048) | 8060S |
|---|---|---|
| consistency | PASS | PASS |
| agent (name fix mirror) | PASS, 6 turns, 0 failing (first pass without the oracle server failed with "llama.cpp is not answering"; the re-run `gates_quick.sh` passed) | PASS, 6 turns, 0 failing |
| golden | PASS | PASS |
| compare (dev, 14 prompts) | PASS: 7 full, median matched 55, max gap 0.582, 0 bad | PASS: 9 full, median 64, max gap 0.296, 0 bad |
| compare (golden + golden2, 23 prompts) | PASS: 12 full, median 55, max gap 0.582, 0 bad | PASS: 14 full, median 64, max gap 0.461, 0 bad |
| sampling | FAIL (temperature 0 and top_k 1 against greedy: the default effort check, by design as in wave 1) | FAIL (same) |
| agent, unmodified script | 1 turn, 3 failing (as in wave 1: tool messages without a name) | same |
| agent, effort none | 6 turns, 0 failing | 6 turns, 0 failing |
| context admission 131072 | 130,991 tokens ADMITTED; COMPLETED in 174 s (wave 1: 179 s) | 130,991 tokens; COMPLETED in 591 s (wave 1: 626 s) |

The compare numbers equal the wave-1 gate numbers (R9700 dev 7 full, max gap 0.582; 8060S dev 9 full, max gap 0.296), the sampling and unmodified-agent failures are the same known ones. Gate verdict stays "FAIL" for the same reason as in wave 1 (the sampling gate). The full-level audit (`finalize.sh`) was not run.

## Command programs (reusable launch graph, 0.3.0): where we launch one by one, and what a prototype takes

Sources read: the 0.3.0 SDK guide `guide/command-programs.md` (laptop copy of the SDK docs), the library model `model/Qwen/Qwen3.8-27B/program/model_decode.loom` and `epoch.loom` (v0.3.0 tag), the dialect definition in the installed 0.3.0 wheel. PRs #2475 and #2081 were NOT read (no web fetch tool in this lane; a web search returned only the release pages). Everything about those PRs below is therefore unknown.

What we have already:

- The dispatch is already a command program. `dispatch/model.loom` defines `command.def public target(@inkling_execution) @inkling_model()` with one `command.serial` region: `kernel.launch @inkling_prep_spans`, `command.launch @inkling_hook_embed`, then `command.launch @inkling_visit[layer, visit, pass]` for each of the 48 visits, then `@inkling_hook_head`. `dispatch/layer.loom` defines `@inkling_visit`, which composes the 15 hook programs (`mix_attn`, `mix_msg`, `ctrl_qkv`, `norm_in`, `proj`, `attn_prep`, `attn`, `oproj`, `post_attn`, `mix_ffn`, `ctrl_ffn`, `norm_ffn`, `router`, `experts`, `post_ffn`) with exact visit coordinates as specializations, so global or local attention, loop controllers and weight types fold at plan time. Weights come through `command.parameter` (named blocks `blk.{}.…[%layer]`).
- The launches are in the hooks: `scripts/gen_hooks.py` (`klaunch(...)`, 29 call sites in the generator, expanding to about 106 `kernel.launch` sites in `dispatch/hooks_dense.loom` (66), `hooks_mix.loom` (22), `hooks_experts.loom` (14), `hooks_attn.loom` (4)). Each decode step runs one specialization of this graph per shape bucket (the stage capacity decides the bucket: tokens, spans, `n` of 1 to 8 are the decode buckets). So the graph is reusable already; what is not expressed is concurrency: everything sits in one serial schedule, one dispatch and one dependency edge per `kernel.launch`.
- Launch count: 1,569 dispatches per decode step at baseline (HAL profile, `docs/OPTIMIZE_PLAN.md`), about 900 in the merged build (wave 1 -194, D2 -336, D3 -144, D5 +2; derived from the lane docs, not re-profiled). Measured cost of a dispatch: 0.7 to 0.9 us (profiling overhead per dispatch in the plan) and the lanes realised about 0.75 us per removed launch (D2: 336 launches, 0.25 ms). So the whole per-launch overhead of the merged step is about 0.7 ms of 10.5 ms on the R9700 (7 percent) and about 0.7 to 1 ms of 23 ms on the 8060S (3 to 4 percent). That bounds what any launch-graph change can win; it is not a path to the ceiling gap (4.4 and 6.0 ms).

What a decode prototype would take (not implemented here):

1. Measure first, no code: HAL dispatch statistics of one decode step give busy time against wall time (wave 1: busy 11.66 ms against profiled wall 12.59 ms, of which about 1 ms is profiling cost). If the idle gap between dependent dispatches is under 0.3 ms, stop.
2. `command.concurrent` for independent branches inside a visit. Candidates are few after D1 to D3: the attention conv commit against the router chain of the previous FFN (different buffers), `ctrl_down_msg` against the norm of the other half, and the K and V projections that D2 already fused. Each candidate needs a hazard check by hand (buffer aliasing does not create edges inside `command.concurrent`: `sink` lives in `small`, `dyn_k` in `ob`, `k_act` in `xn` per `KERNELS_attn.md`). Expected gain: below 1 percent each.
3. A fixed decode program for bucket n = 1 (`@inkling_decode_step`): unroll the 48 visits (already exact) and make the three dynamic inputs (context length per span, the token, the sampler seed) storage reads only, so one planned artifact is replayed per token. This is what `@inkling_model` already specializes per bucket; the open question is whether loom-serve 0.3.0 records and replays it as one HAL command buffer or re-enqueues per step. That is a runtime property (check the host CPU time per step with `perf`, and the 4-stream aggregate, which already hides host gaps: 213 tok/s on the R9700 is 2.2 times the single stream). A prototype for this question is one afternoon: a tiny package with 48 identical `command.launch` of one small kernel, timed with and without `command.concurrent`.
4. Effort for the real thing: 2 to 3 days (generator change for concurrent regions, hazard review, per-bucket tests: the kernel tests and the capture-route layer check already cover the numerics path). Expected total gain: the 0.7 ms launch overhead bound above, realistically a quarter of it.

## Open issues and leftovers

- No 0.2.0 A arm was measured (owner order); the R9700 B arm sits 1.7 to 3.1 percent below published decode numbers. The repeat session of both arms (`m1b.r9700.*`) agrees with the first (R9700 B 87.45 again). A single 0.2.0 arm of `mat-t10b` in the same session would settle whether T10b decode on the R9700 lost 3 percent on 0.3.0 (one arm, about 6 minutes: `bash scripts/m30_seq.sh r9700 x dshort,d8k 3 A A=$HOME/inkling/M30/mat-t10b:$HOME/loom/.venv`).
- `lib/VENDORED.md` still names library v0.2.0; the v0.3.0 library has newer motifs (`attention` kernels, `q4_k_f32_panel`, `gdn_projection`) that were not evaluated for our kernels.
- Scratch left on the box (mine, not deleted: delete denied on the laptop, left on the box for the next agent): `~/inkling/M30` (work copies, mat dirs `mat-t10b`, `mat-all` are hard-link copies of `optD2/mat-base` with program files replaced, results in `res/`, `kres/`, `ab_log.jsonl`, logs in `ablogs/`), `~/inkling/M30r`, `~/inkling/M30s`, `~/inkling/M30r2`, `~/inkling/M30s2` (kernel-test work dirs, hard-linked fixtures; each can be several GB), `~/inkling/M30g` (gates root), `~/inkling/M30base`, `~/inkling/M30/cardwork` and `pubcopy` (card tool check), `~/loom030` (the new venv; keep). Laptop: `scratch_M30/` (patched snapshots of the merge steps and the rejected `.rej`), `scratch_M30_offset_rename_attempt/` (the renamed trees).
- The adapter chat suite starts its own servers; run it with no other server running (it conflicts with the `pgrep` guard of the timed scripts).
