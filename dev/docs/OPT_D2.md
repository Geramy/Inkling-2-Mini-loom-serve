# Lane D2: decode attention front end and projection group (agent optD2, bead llama-cpp-tg3.31)

Base: `package_T10b` (wave 1 merged). Working copy `package_D2` (laptop), `~/inkling/optD2/package` (box). Patch: `lanes/D2.patch` (package files, the regenerated `dispatch/*`, and the `scripts/gen_hooks.py` hunks). Decisions: `bench/decisions_D2.jsonl`.

## What changed

| # | Change | Launches per decode step | Files |
|---|---|---|---|
| 1 | `inkling_gemv_attn_group`: one launch for q, rel, gate, k, v (Q8_0) and sink, k_conv_proj, v_conv_proj (BF16; the two conv projections only on global visits). All tensors read the same `xn` rows; every tensor gets whole workgroups with the same pass count, so one grid covers all of them. Used when the stage capacity is 8 or less and the dense type is Q8_0; larger stages and the BF16 type keep the old GEMV/GEMM launches. | about -5 per attention visit (-240) | `kernel/dense/gemv.loom` (`inkling_dense_gemv_xs` split into `inkling_dense_gemv_xs_at` with workgroup index and pass count as parameters, the old function is a wrapper; new `inkling_dense_group_plan`, `inkling_gemv_attn_group`), generator `attn_group`, hook `proj` |
| 2 | `inkling_attn_post`: merge of the decode splits, KV append and conv commit in one launch (1-D grid: 24 merge workgroups per span, one append workgroup per token, 16 commit workgroups per span). The kernel bodies are now inline functions (`inkling_attn_merge_wg`, `inkling_attn_append_wg`, `inkling_attn_commit_channel`); the old kernels remain as thin wrappers (used by prefill tests and kept ABI stable). The commit moved from the `attn_prep` hook to the `attn` hook (it only needs `k`, `v` raw, unchanged until the next `proj`). | about -2 per attention visit (-96) | `kernel/attn/attn_{merge,kv_append,conv_commit}.loom`, generator `attn` / `attn_prep` hooks |

The prefill path is unchanged: stages with capacity above 8 use the old GEMM launches, and `inkling_attn_post` serves prefill spans too (merge workgroups of prefill spans return at once, the append and the commit run for every span as before, with the same ordering after both tile launches).

Order rule found: a hook must reference weight tensors in the same order in every stage. The first group version referenced k and v before sink and failed registration with "stage parameter placement 6 differs". The group launch now references q, rel, gate, sink, then k, v, (k_conv, v_conv), the order of the old chain.

Not done: prep + append fusion (append must follow the tile on local rings for prefill chunks shorter than 1024 tokens; the hook cannot know that a stage has only one-token spans), folding the gate multiply into the o-proj (skills' GEMV tests: consumer prologues lost), skipping the mode-0 tile launch in decode stages (tile belongs to lane P3).

## Evidence (ABBA, fresh server per arm, preset `chat` with prefill_capacity 2048, median of 3 requests per arm; `scripts_D2/abba.sh`)

R9700 (gfx1201):

| Row | A (T10b) | B (D2) | B | A | Delta |
|---|---|---|---|---|---|
| decode short | 89.49 | 91.93 | 92.01 | 90.21 | +2.4 % |
| decode 8k | 79.74 | 81.56 | 81.84 | 81.03 | +1.6 % |
| prefill 2k | 3561 | 3560 | 3542 | 3556 | -0.2 % (noise, same kernels for n > 8) |
| 4 streams | 206.3 | 210.2 | 209.1 | 205.2 | +1.9 % |

The measured gain is about 0.25 ms per step against the plan's 0.7 ms (336 fewer launches; the group GEMV costs some of it back).

8060S (gfx1151), same method (ABBA, 3 requests per arm):

| Row | A (T10b) | B (D2) | B | A | Delta |
|---|---|---|---|---|---|
| decode short | 38.98 | 40.78 | 41.04 | 39.62 | +4.1 % |
| decode 8k | 34.28 | 35.81 | 35.92 | 35.06 | +3.5 % |
| prefill 2k | 1156.9 | 1185.1 | 1190.7 | 1190.8 | +0.1 % (the two A arms differ by 3 %: session variance of prefill; same kernels for n > 8) |
| 4 streams | 82.16 | 84.32 | 84.95 | 83.86 | +1.9 % |

Not measured (lock queue of 1 to 2 hours per session when this lane ran, so the time box went to the main rows): decode 32k, prefill 8k, the long-document greedy checks (8k and 32k character documents) and the capture-route layer check. The integrator's final gates (`compare`, `consistency`, 16k/32k) cover them; the changed kernels do not depend on the context length (the post launch only adds work per token and per span). A script for the long greedy check is in `scripts_D2/greedy_long.sh` (with `greedy_check.py`).

## Numerics

- `test/dense/test_gemv_group.py` (new): the group kernel against numpy on the exact Q4_K_XL weights, layers 5 (global, 8 tensors) and 0 (local), 1, 3 and 8 live rows: 6 of 6 PASS on gfx1201 and gfx1151, NMSE 1e-14 (same arithmetic as the single GEMV).
- `test/attn/attn_test.py` (the launch chain uses `inkling_attn_post`; `ATTN_SEPARATE=1` runs the old three kernels): 130 of 130 PASS on gfx1201 and gfx1151, chunk-invariance lines identical to the base (prefill bitwise, decode versus 1 chunk 5e-4 to 1e-3). The tests include the decode and mixed (prefill span of one row beside decode spans of another) cases, so the conv state commit order is covered by the outputs of the later epochs.
- Greedy A, B, C (64 tokens, chat route): identical to the base on both GPUs (`g_base_*.json` against `g_g2_*.json`).
- Not run: the capture-route layer check (`capture_all.py`; the group GEMV is arithmetic-identical to the single GEMV and the attention kernel tests compare with the oracle dumps).

## Per-arch rows

None. Both changes are arch neutral (no tune value). The group kernel takes its pass count from `inkling_dense_mv_xs_plan` with `xs_wg_per_unit` of the device (gfx1201 2, gfx1151 4); a separate tune row for the group (pass count 8 to 32) was not tried.

## Stop reason

The stop rule (below +1.5 percent decode short after the candidates) did not hit on the R9700 (+2.4 percent). Open: the group pass count tune, the BF16 dense type (keeps the old launches) and the 32k and prefill 8k guard rows.

## Rebuild

```
# box
cd ~/inkling/optD2; python3 scripts_D2/gen_kernel_decls.py; python3 scripts_D2/gen_hooks.py    # writes package/dispatch/*
bash scripts_D2/make_var.sh g2                 # hard-link tree of mat-base (T10b, prefill_capacity 2048) with the changed program files
bash scripts_D2/abba.sh r9700 g2 dshort,d8k,pf2k,c4 3
bash scripts_D2/locked.sh bash scripts_D2/greedy_run.sh r9700 $PWD/var-g2 g.json
```
Kernel tests: `PKG=~/inkling/optD2/package python package/test/dense/test_gemv_group.py gfx1201` (dense venv), `ATTN_ROOT=~/inkling/optD2 python package/test/attn/attn_test.py run --dev gfx1201` (inkE venv).
