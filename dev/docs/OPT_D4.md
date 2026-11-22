# OPT_D4: long-context decode attention (lane D4)

Result: R9700 decode 32k +2.4 percent, 8060S decode 32k +5.1 percent, short context neutral. Prefill gains on the R9700 (+5.5 percent at 2k, +10.2 percent at 8k) as a side effect. The plan target (+10 percent at 37k) is not reached: the global decode tile kernel is at about 0.74 of the bandwidth in the bench and the rest of the attention launches did not change (see Stop reason).

## What changed

| File | Change |
|---|---|
| `package/kernel/attn/attn_tile.loom` | (1) the K/V stage loop is unrolled. (2) The m/l stage uses the score stage and the rescale stage uses the r stage (both are dead at that point): workgroup memory 33,024 B to 32,000 B, so 4 workgroups per CU can run instead of 3 (compile report: `resident_subgroups_per_simd` 3 to 4, no spills, 256 VGPR). (3) The decode split size comes from `inkling_attn_decode_split`. |
| `package/kernel/attn/attn_merge.loom` | uses the same `inkling_attn_decode_split` for its split count. |
| `package/lib/inkling/attn_common.loom` | new `inkling_attn_decode_split(min_split, blocks)`: `max(min_split, ceil(blocks / 64))`, so a visit never has more than 64 splits and a global visit with 512 blocks runs 512 workgroups (8 KV heads x 64 splits). |
| `package/dispatch/hooks_attn.loom` and `scripts/gen_hooks.py` | the `inkling.tune.attn.split_blocks` rows of gfx1151 and gfx1201 return 1 (they are now the smallest split; the context table 2/4/8 is gone); `max_splits` is capped at 64. The portable default row (4 on global visits) is unchanged. |
| `package/test/attn/attn_perf.py` | new rows `dec_g_<ctx>_dyn` (dynamic split, grid 64 splits). |

Local visits are unchanged (at most 17 blocks, so one block per split). Prefill mode computes the same values; only its register and workgroup memory schedule changed (LDS aliasing, unrolled stage). The ABI of the kernels did not change. `split_blocks` keeps its name in the kernel signature; it now means the smallest split.

## Evidence

Kernel bench (`test/attn/attn_perf.py`, R9700, one decode token, mean dispatch time):

| Row | Base us | New us |
|---|---|---|
| dec_g 32768 (best static split b4) | 352.3 | 294.4 |
| dec_g 16384 (b4) | not measured | 133.6 |
| dec_g 4096 (b4) | 55.1 | 48.2 |
| dec_l 1024 | 25.6 | 23.0 |
| pre_g 512 at 4096 | 2936 | 2159 |
| merge_g 32768 | 23.0 | 22.2 |

End to end ABBA (`ab_abba.sh`, A B B A, fresh server per arm, 3 requests per metric, median per arm; tok/s):

| GPU | Row | Base | New | Gain |
|---|---|---|---|---|
| R9700 | decode short | 85.90, 86.09 | 86.11, 86.32 | +0.25 percent |
| R9700 | decode 8k | 76.85, 77.21 | 77.40, 77.75 | +0.7 percent |
| R9700 | decode 32k | 67.82, 67.99 | 69.44, 69.63 | +2.4 percent |
| R9700 | prefill 2k | 2293.6, 2241.7 | 2391.5, 2394.2 | +5.5 percent |
| R9700 | prefill 8k | 1991.7, 1939.8 | 2166.6, 2164.3 | +10.2 percent |
| R9700 | 4 streams | 201.9, 199.4 | 202.9, 203.1 | +1.2 percent |
| 8060S | decode short | 38.34, 38.74 | 38.65, 38.63 | +0.3 percent |
| 8060S | decode 8k | 33.32, 33.64 | 34.21, 34.24 | +2.2 percent |
| 8060S | decode 32k | 27.80, 27.89 | 29.23, 29.28 | +5.1 percent |

8060S guard rows (same ABBA): prefill 2k 864.55, 866.63 to 892.68, 888.16 (+2.9 percent); prefill 8k 772.42, 772.41 to 805.01, 800.82 (+4.0 percent); 4 streams 81.91, 81.25 to 81.63, 81.18 (-0.2 percent, noise). Spread inside one session 0.1 to 0.3 percent. Decisions: `bench/decisions_D4.jsonl`.

## Numerics

- Attention kernel tests (`package/test/attn/attn_test.py run --dev <gpu>`, includes the long-prompt oracle dumps of slots 0, 5, 6, 11 for 1 chunk, 3 chunks, 512 + rest, decode and mixed): 130 of 130 pass on gfx1201 and gfx1151; the chunk-invariance lines are identical to the base results (prefill bitwise, decode versus 1 chunk 5e-4 to 1.3e-3 as before). These tests have at most 21 blocks per visit, so the dynamic split is not active there; the long-context path is covered by the greedy checks below.
- Greedy 64 tokens through the chat route, base against new (`scripts/greedy_check.py`, prompts A, B, C of `oracle/prompts`, an 8k and a 32k character document): R9700 5 of 5 identical. 8060S: A, B, C and 8k identical; 32k differs at about token 25 (see below).
- The oracle layer checks through the capture route (`capture_all.py`) were not run: the capture route serves a separate deployment and the attention kernel tests above already compare against the same oracle dumps. The integrator should run `jcheck.sh`/`jchecklong.sh` once at the final gates.

## Per-arch rows (TUNE_ROUTE)

`inkling.tune.attn.split_blocks(context, global)`: gfx1201 1, gfx1151 1, portable default 4 on global visits, 1 on local visits. The 64 split limit is the constant in `inkling_attn_decode_split` and in the host `max_splits` cap.

## Stop reason

Three kernel and schedule changes were kept as one arm (the ABBA lock wait was about 20 to 40 minutes per arm because of the shared lock, so there was no time to separate them end to end). The R9700 decode kernel runs at 456 GB/s at 32k in the bench (floor 216 us at 620 GB/s: 0.74 of the bandwidth, the base was 0.61) but the in-model attention at 32k is still above the floor (the 8 global visits, merge, prep, append and the 40 local visits). Further gains need a different tile (K and V as two streams with double buffering needs more workgroup memory than the 4 workgroups per CU allow; a vector dot decode tile without the 16 row matrix tile) or fusion of the local-visit launches (prep, conv commit, append, merge: 5 launches, about 37 us per local visit): both are larger than the lane time box and are left open.

## Rebuild

Copy the changed files into a package tree, run `scripts/gen_hooks.py` only if the hook rows change, and serve with the chat preset. The kernel bench: `python3 package/test/attn/attn_perf.py --dev gfx1201 --only "dec_g_.*_dyn|pre_g_512_at4096"` (needs the loom tools venv and the GPU lock).

## Greedy difference on the 8060S at 32k

Base: "...The content is a large set of Loom documentation pages. This is a benign request: summariz..." New: "...The content is a large set of docs about Loom (an AMD/ROCm-related compiler/runtime). This..." (first difference at about token 25 of 64; both continue fluently). The new split changes the order of the softmax partial sums, so a near tie between two plausible tokens can flip. This is not verified: the server returned no logprobs (the logprob rerun, `scripts/greedy_lp.py`, got an HTTP error), and no second base run was made to check that the base is repeatable at 32k on that GPU. The R9700 runs of the same prompt are identical. Treat as open for the final gates (`compare` and `consistency` at 16k/32k).
