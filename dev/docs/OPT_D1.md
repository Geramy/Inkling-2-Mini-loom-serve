# Lane D1: decode residual-chain fusion (agent optD1, bead llama-cpp-tg3.25)

Base: `package_base_T10`. Working copy: `package_D1` (laptop), `~/inkling/optD1/package` (box). Patch: `lanes/D1.patch` (package files, generated hooks, and the `scripts/gen_hooks.py` hunks; the generator is the source of `dispatch/hooks_mix.loom`, `dispatch/hooks_dense.loom` and `dispatch/kernel_decls.loom`).

## What changed

| # | Change | Launches per decode step | Files |
|---|---|---|---|
| 1 | `conv_commit` rides in the next `mix_states` launch (extra workgroups past the token workgroups; same arithmetic). Only the commit of layer 23 keeps its own launch (the head follows). The commit of a visit moves into the first mix launch of the next executed visit (layers 0 to 17, then loop passes 6 to 17, then 18 to 23), or into the layer-5 extra mix. | -95 | `kernel/mix/conv_res.loom` (`inkling_conv_commit_item`), `kernel/mix/mix_states.loom`, generator `prev_commit`, `conv_commit` |
| 2 | `mix_states` mode 1 (already present: mix, rms norm, weight) for uncontrolled local attention and for every uncontrolled FFN visit. `norm_in` and `norm_ffn` launch nothing for these visits. | about -43 | generator `mix_attn`, `mix_ffn`, `norm_ffn` |
| 3 | `inkling_rms_norm_cols`: one launch for the three columns of `norm_in` (controlled loop visits and global visits, one weight per column). | about -56 | `kernel/dense/norm.loom`, generator `rms_norm_cols`, `norm_in` |
| 4 | `nq` of `mix_states` is a specialization (workload) value. No speed change measured (see below); kept because it removes dead query work and the kernel stays one source. | 0 | `kernel/mix/mix_states.loom` |

Launches per step: rms_norm 152 to 25, rms_norm_cols 28, conv_commit 96 to 1, total about -194 of 1569 (profile, `dshort_d`, R9700). New `mix_states` ABI: workload `[cap, ncommit, nq]`, scalars `nb, block_stride, col_stride, mode, w_stride, sel_on, blocks_off, cur_off, out_off, commit_site, commit_visit`, buffers as before plus `commit_state, commit_a`.

Layout rule that the integrator must keep: in an uncontrolled local visit `xn` now comes from `mix_states`, and `x` is not written, so the debug capture ids 10 (x) and 24 (x) are stale for those visits (capture 11, 23 and 28 are unchanged). The commit of visit V needs `y` (FFN) or `ob` (attention) unchanged until the next mix launch: no hook between them writes those buffers.

## Why the mix kernel runs at 2x its benchmark (finding)

Profile `dshort_d`, R9700, `mix_states` split by hook with scratch aliases (`scripts_D1/split_mix.pl`, not in the patch):

| Call | avg us | with a hot query weight (query read from `blocks`) | cold-weight cost |
|---|---|---|---|
| ffn mix (nq 1) | 9.43 | 6.35 | 3.1 us |
| attn mix (nq 1 or 3) | 11.12 | 8.91 | 2.2 us |
| message mix (nq 4, nb 1) | 10.26 | 9.61 | 0.65 us |
| all, base | 10.66 | | |

A cold read of the query weight costs 2 to 3 us per launch: the weights are streamed once per step (17.6 GB) and are not in cache when the small kernel starts, while the isolated benchmark repeats the same launch with warm weights. Instruction count is not the cause: with `nq` as a specialization (1 query chain instead of 4) the average only fell from 10.66 to 10.46 us. The rest (6 to 9 us against 4.7 to 6.1) is dependent-launch latency, not ALU. A cache prefetch in the preceding kernel would move the miss to that kernel (about 1.5 us net, uncertain); there is no prefetch instruction in Loom, so it was not tried.

## Evidence (R9700, gfx1201, Q4_K_XL, chat-r9700, ABBA, fresh server per arm, `scripts_D1/abba.sh`, median of 3 runs inside each arm)

| Row | A (base) | B (d1b) | A | B | Delta mean |
|---|---|---|---|---|---|
| decode short | 85.97 | 88.39 | 85.92 | 88.54 | +2.9 % |
| decode 8k | 76.78 | 78.58 | 76.80 | 78.78 | +2.5 % |
| prefill 2k | 2216 | 2259 | 2214 | 2260 | +2.0 % |
| 4 streams | 200.5 | 203.0 | 199.7 | 203.0 | +1.5 % |

Second ABBA on the R9700 (`d1c`, the final code with the `nq_r` assume, arms A B B A, 2 reps each): decode 32k 68.07 / 69.33 / 69.17 / 67.45 (+2.2 %), prefill 8k 1934 / 1980 / 1981 / 1935 (+2.3 %).

8060S (gfx1151, chat, `d1c`, ABBA, 3 reps per arm):

| Row | A | B | B | A | Delta mean |
|---|---|---|---|---|---|
| decode short | 38.69 | 39.35 | 39.38 | 38.77 | +1.8 % |
| decode 8k | 33.60 | 34.18 | 34.19 | 33.68 | +1.8 % |
| decode 32k | 27.95 | 28.37 | 28.38 | 27.91 | +1.6 % |
| prefill 2k | 882.8 | 883.0 | 886.2 | 881.3 | +0.3 % |
| 4 streams | 82.02 | 83.20 | 82.86 | 81.87 | +1.2 % |

The 8060S gain is below the 2 percent rule on every row (no row loses). The R9700 target row is the one that counts for this lane; the integrator decides whether the 8060S keeps the change (it costs nothing there and no row is below base).

Profile busy time of the changed families: rms_norm + conv_commit 0.448 ms per step to 0.135 ms; mix_states 1.568 to 1.538 ms; conv_res 0.131 to 0.141 ms. Wall decode time 400 tokens: 5.025 s to 4.844 s (profiled run).

## Numerics

- Greedy A, B, C (64 tokens, chat route, `scripts_D1/greedy_run.sh`): R9700 identical to base for `d1b` and `d1c`. 8060S: A and C identical, B differs at character 233 ("Now they want me to explain" in base against "The final instruction: explain"). This is the known arch near-tie: the base outputs of the two GPUs already differ at the same position (`inkM/old_1151.json` says "Now they want me", `old_1201.json` says "The final instruction"), and the fused build on the 8060S now gives the R9700 text.
- Kernel tests (run under the shared GPU lock, new ABI): `test_conv.py` with the commit in the mix launch: 342 of 342 PASS on gfx1151 (gfx1201 run not finished); `test_conv.py` standalone commit kernel: 342 of 342 PASS gfx1201; `test_ctrl.py`: 240 of 240 PASS gfx1201; `test_norm_embed.py` (16 cases incl. the two `rms_cols_*` cases): 0 failed on both GPUs; `test_mix.py` (oracle slots, nq 1 to 4, mode 0 and 1): 91 of 91 cases PASS (0 FAIL) on gfx1201 when I stopped waiting for the lock, 299 planned (the rest, and the gfx1151 run, are not done; the scripts are `scripts_D1/ktests.sh`). Test harness changes: `harness.py` drops a `func.decl` that another concatenated source defines; the mix tests pass the new workload and scalar list.

## Per-arch rows

None needed: the fusions are arch-neutral (no tune value). If the 8060S ABBA shows a loss, gate change 2 and 3 by a `template.def` provider on `@inkling_rdna3`.

## Stop reason and what is left

Stop rule: not hit, +2.9 % decode short on the R9700 after the three candidates (1 explained, 2 and 3 done). Left (not done): global uncontrolled visits still use one norm launch per column (4 visits); fuse `norm_ffn` into `ctrl_apply` for controlled visits (24 launches); cold query-weight prefetch.

## Rebuild

```
# box
cd ~/inkling/optD1; python3 scripts/gen_kernel_decls.py; python3 scripts/gen_hooks.py      # writes package/dispatch/*
bash scripts_D1/make_var.sh r9700 NAME                                                        # hard-link tree with the changed program files
bash scripts_D1/abba.sh r9700 NAME dshort,d8k,pf2k,c4 3                                       # ABBA, exclusive lock per arm, retry on the run guard
bash scripts_D1/greedy_run.sh r9700 var-r9700-NAME g.json; route_check.py --diff g_base.json g.json
```

Kernel tests: `scripts_D1/ktests.sh gfx1201|gfx1151` (mix, conv with commit in the mix launch and standalone, controller) and `test/dense/test_norm_embed.py` with `PKG` pointing at the package (case `rms_cols_*`).
