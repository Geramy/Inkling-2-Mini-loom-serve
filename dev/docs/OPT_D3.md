# Lane D3: decode controllers and router (agent optD3, bead llama-cpp-tg3.32)

Base: `package_T10b` (wave 1 merged). Working copy: `package_D3` (laptop), `~/inkling/optD3/package_D3` (box). Patch: `lanes/D3.patch` (package files, the three generated dispatch files, the tests, and the `scripts/gen_hooks.py` hunks; the generator is the source of `dispatch/hooks_mix.loom`, `dispatch/hooks_experts.loom` and `dispatch/kernel_decls.loom`). Stopped on the owner's order (move to hrx-loom 0.3.0): the last code state (`d3i`) has kernel-level evidence and kernel tests, but no end to end ABBA, no greedy run and nothing on the 8060S; see "Not done".

## What changed

| # | Change | Launches per decode step | Files |
|---|---|---|---|
| 1 | `inkling_router_decode` (new export): top 8 of all rows and the expert route table in ONE workgroup for stage rows <= 8. The select code is an inline function (`inkling_router_select_one`) that the old `inkling_router_select` also calls; the route code is an inline function (`inkling_experts_route_groups`) that the old `inkling_experts_routes` also calls (large-n path: same code, tested to n 2048). `router_select` moves from the router hook to the experts hook (the router hook is the logits kernel only). | -48 | `kernel/experts/router.loom`, `routes.loom`, generator `experts_file` |
| 2 | Select with one cross-lane reduction per round: `kernel.subgroup.vote.ballot` of the lanes that hold the maximum, `cttz` gives the lowest lane (lowest expert index: same tie rule), `kernel.subgroup.broadcast` gives index and probability. The old code used three reductions per round (max, min index, sum of the winner). Same ids and weights bit for bit. Wave32 only, as before (a lane owns 4 of the 128 experts). | 0 (about -0.6 us per visit) | `router.loom` |
| 3 | `inkling_router_logits`: the 18-step weight loop is fully unrolled. The f16 copy of x is packed by 5 threads of every workgroup instead of 128 threads of the first workgroup. | 0 (-1.9 us per visit) | `router.loom` |
| 4 | `inkling_ctrl_apply_z` (new export, stage rows <= 8): `ctrl_z` is folded into the apply kernel (64 threads add the `ctrl_down` partials in the order of `ctrl_z` into an LDS vector, a barrier, then the old math; the x and m loads are issued before the z sum). | -48 | `kernel/mix/ctrl.loom`, generator `ctrl_hook` |
| 5 | `inkling_ctrl_down_msg` (new export, stage rows <= 8): the message mix and `ctrl_down` in one launch (one token per workgroup, 288 threads). Workgroups of the x half of the 4608 inputs run as `ctrl_down`. Workgroups of the m half build the message column of the controller in LDS with the arithmetic of `inkling_mix_states` (nb 1, mode 0, one query; they call its softmax, coefficient and axpy inline functions), write their slice of the column and (the first one) the rms factor `rinv_m`. The state rows for the final sum are loaded before the first barrier. The `mix_states` launch of the message mix is gone for decode (59682 to 40194 launches in the 400 token profile). Message columns and `rinv_m` are bit equal to `mix_states`. | -48 | `kernel/mix/ctrl.loom`, generator `ctrl_hook` |
| 6 | `inkling_ctrl_apply` for the prefill shape (rows > 8): was 8 lanes per output row with one lane doing the element-wise part, now one thread per output row. The 2 x 64 weights of a row stay in registers (16 vec4 each), the z values of the token tile are staged in LDS, no cross-lane reduction, one tanh pair per element, x, m and the output are accessed in rows of consecutive elements. Tile = `inkling.mix.ctrl_tokens` (16 in the tune table, at most 64, LDS 16 KB). (P4 of the plan.) | 0 | `kernel/mix/ctrl.loom` |

Decode launches per step: -144 (select and routes -48, ctrl_z -48, message mix -48). The decode controller chain is now `ctrl_down_msg`, `ctrl_apply_z`, the norm; the FFN router chain is `router_logits`, `router_decode`.

New ABI (all arch neutral, no tune value): `inkling_router_decode()` launch `(n, tile, logits, bias, ids, wts, tasks)`, n <= 16 (the generator uses it for stage rows <= 8); `inkling_ctrl_apply_z[cap, nc]` launch `(cap, nc, x_col0, x_col_step, m_col0, col_stride, spans, x, m, forget0..2, gain0..2, rinv_x, rinv_m, part, out)`; `inkling_ctrl_down_msg[cap, nc]` launch `(cap, nc, x_col0, x_col_step, m_col0, col_stride, blocks_off, cur_off, spans, x, bank, query, down0..2, msg, rinv_m, part)`. `inkling_ctrl_apply` keeps its ABI. `ctrl_down_msg` needs the link set of the mix area (`mix_states.loom` defines the softmax, coefficient and axpy functions it calls; `ctrl.loom` declares them).

## Evidence (R9700, gfx1201, Q4_K_XL, chat-r9700 card, prefill_capacity 2048)

Kernel times from the HAL dispatch profile (`prof/<variant>.r9700/dshort_d.serve.log`, 400 token decode after a short prompt, per launch, 48 launches of each family per step):

| Kernel | base us | d3b | d3e | d3g | d3h | d3i (last) |
|---|---|---|---|---|---|---|
| router_logits | 5.51 | 3.60 | 3.63 | 3.63 | 3.58 | 3.68 |
| router_select + experts_routes (base) / router_decode | 4.01 + 1.40 | 4.06 | | 4.18 | 4.26 | 4.18 |
| mix_states, message mix (D1: 10.3) + ctrl_down 4.81 / ctrl_down_msg | 10.3 + 4.81 | 10.3 + 4.79 | | 11.70 | 8.88 | 8.37 |
| ctrl_z 1.18 + ctrl_apply 4.79 / ctrl_apply_z | 5.97 | 4.96 | | 5.00 | 4.94 | 5.00 |
| profiled decode wall, 400 tokens (s) | 4.751 | 4.687 | 4.682 | 4.587 | 4.512 | 4.521 |

Variants: d3a = changes 1 and 4 (first version); d3b = 1 to 4 with ballot select and unrolled logits; d3e = d3b + pack spread + change 6; d3g = d3e + change 5; d3h = d3g with `unroll` in the message loops; d3i = d3h with the state rows prefetched before the first barrier. The profiled wall time falls 4.8 percent from base to d3i (profile runs are about 8 percent slower than the unprofiled serve).

Prefill (pf2k profile, one 2040 token chunk of stage capacity 2048; per launch, 48 per chunk): `ctrl_apply` 773 us (base) to 479 us (8 lanes with lane exchange through LDS, d3b) to 242 us (one thread per output row, d3e, change 6): 111.3 ms to 34.9 ms for the three chunks of the run, about -25 ms per chunk (-3.7 percent of the 680 ms chunk). `router_logits` at 2040 rows: 226 us to 237 us (+5 percent, 11 us per chunk of 48 launches: noise level). `ctrl_down` 312 us and `ctrl_z` 68 us are unchanged (ctrl_down is FMA and L1 bound near 3x its floor; WMMA is out because the controllers are F32 by design).

End to end ABBA, `scripts_D3/abba.sh` (fresh server per arm, base = `package_T10b` at prefill_capacity 2048, 3 runs per arm, medians):

| Row | A base | B d3e | B d3e | A base | Delta |
|---|---|---|---|---|---|
| decode short (tok/s) | 90.33 | 91.49 | 91.06 | 90.30 | +1.1 percent |
| decode 8k | 80.58 | 81.60 | 81.45 | 80.54 | +1.2 percent |
| prefill 2k (tok/s) | 3600.9 | 3738.2 | 3725.6 | 3547.3 | +4.4 percent |
| 4 streams | 205.7 | 207.8 | 208.5 | 205.8 | +1.2 percent |

d3b (changes 1 to 4 only) ABBA: decode short 89.91 / 91.05 / 91.51 / 89.90 (+1.5 percent), decode 8k +1.2, 4 streams +0.6. d3e is the last build with an end to end ABBA. The decode gain of changes 5 and the loop tuning (d3g to d3i) is only measured at kernel level and in the profiled wall time (about -3.5 percent more); an unprofiled ABBA of d3i is the first job of whoever continues (`scripts_D3/chain_all.sh d3i`).

## Numerics

- Greedy A, B, C (64 tokens, chat route, `scripts_D3/greedy_run.sh`, R9700): token-identical to the base for d3a, d3b, d3e and d3g (`route_check.py --diff`). d3h and d3i were not run (they change only loop unrolling and a load position of d3g, and the message kernel test below is bit exact).
- Kernel tests, R9700, code state d3e (before change 5 and the loop tuning): `test_router_decode` (new, 80 cases: select, route table and weights of the fused kernel against the split kernels and a host reference, with tied logits, n 1 to 8), `test_oracle` (router vs oracle; 3 variants), `test_gate_up`, `test_prefill` n 37 to 2048 (large-n route path unchanged): 6 of 6 experts targets, `test_ctrl` 240 of 240 at tile 1, 8, 16 and 24 (the rewritten prefill apply), Q4 F32 weights 12 of 12 at chunk 64, 128, 256, fold variant (`CTRL_FOLD=1`: `ctrl_down` + `apply_z`) 240 of 240 and 12 of 12 for chunk 64, 128, 256. `test_mix` 299 of 299. The final form of `test_router_decode` (it adds the route table check against the host reference) was not run; the earlier form passed.
- Kernel test of change 5 (`test_ctrl_msg.py`, new): message mix + ctrl_down + ctrl_z + ctrl_apply against `ctrl_down_msg` + `ctrl_apply_z` on random data, F32 and BF16 weights, nc 3 and 1, rows 1 to 8: 160 of 160 at chunk 256, 128 and 64 on the R9700 and at chunk 256 on the 8060S (code state d3i). The message columns and `rinv_m` are bit equal, the controller output agrees to 1e-14 (nmse), guards intact.
- 8060S: kernel tests of the d3b to d3d states passed (20 of 20: experts, mix incl. fold and t8, t24); nothing of d3e to d3i except the msg test above. Greedy and ABBA were not run.
- Not run for any state: the capture-route layer check (`scripts_D3/capture_job.sh`, visits 6, 24, 29, 47 are prepared in `~/inkling/optD3/capt`) and the chunk-independence runs. The prefill apply is the only kernel whose tile and chunk behavior changed; `test_ctrl` at tile 8, 16, 24 and the 240 case oracle comparison cover it.

## Candidates

| Candidate | Result |
|---|---|
| ctrl_chunk 128 / 64 for the decode buckets (ABBA R9700 dshort) | rejected: 90.32 / 89.95 against base 90.76, 89.87 |
| ctrl_down + ctrl_z + ctrl_apply into one launch | not possible without a grid barrier (loom-kb: 1.3 to 1.9 us per phase against 0.6 to 0.7 us per launch). Done instead: ctrl_z into apply (change 4) and the message mix into ctrl_down (change 5). Rank 64 stays in the global partials `part` and in LDS. |
| norm_ffn into ctrl_apply | not done: needs a row reduction across the 72 workgroups; the only consumer of the normalized FFN input is `router_logits`, so a norm prologue there is the better design (about -1.5 us for 24 visits, 0.3 percent) and was not worth a second logits variant. |
| router_logits weight bank (18 loads first, 92 VGPR) | rejected: 4.22 us against 3.71 us for the plain unroll |
| router_logits with 1 wave per workgroup (128 workgroups) / 2 waves (64) | 3.85 / 3.47 us against 3.52 us: no gain, kept 4 waves |
| router_decode keeps the ids in LDS only (no global ids) | 4.13 us against 4.06 us: no gain, reverted (the ids stay in global memory for diagnostics) |
| ctrl_apply_z with x and m loads before the z sum | kept (inside change 4) |
| ctrl_apply: 8 lanes per row with the element-wise exchange through LDS (coalesced, d3d) | 481 us against 479 us: the instruction count (cross-lane reduce and 8x redundant tanh) bounds it, not the memory access. Replaced by change 6. |
| combine into the post-ffn conv (plan row 5) | not done: `y` is also the commit source of the next visit, so the conv must write it anyway; about 0.8 us per visit |

## Per-arch rows and constraints

None: no tune value was added. `router_select_one` and `router_decode` assume wave32 (as the old select did); `ctrl_apply` needs 16 KB LDS, `ctrl_down_msg` 11.5 KB with 288 threads. The decode kernels are used for stage rows <= 8 (`decode_rows` in the hooks, the same threshold as `prepare.loom`); ctrl_tokens must be 1 for those stages.

## Integrator notes

- Generator hunks (`scripts/gen_hooks.py`): `ctrl_hook` (one `scf.if decode_rows` with the two decode launches in the then branch and the old chain in the else branch) and `experts_file` (the router hook launches logits only; the experts hook launches `router_decode` for rows <= 8 and `router_select` otherwise, and skips `experts_routes` for rows <= 8). Lane D6 edits the same `experts_file`; the experts launches themselves are unchanged.
- Debug capture ids: `ab` (ids) and `ob` (weights) are written one hook later than before (in the experts hook); capture 29 (`small`, the logits) is unchanged. `msg` columns that the visit does not use (column 3 for Q/K/V, columns 0 to 2 for the FFN) are no longer written in decode.
- `ctrl_down_msg` replicates the message mix of `mix_states` (bit equal in the test). If `mix_states` changes its message arithmetic, `test_ctrl_msg.py` fails.
- The tests changed: `test_ctrl.py` (`CTRL_FOLD=1`), new `test_router_decode.py` (work dir `~/inkling/optD3/work`: change before reuse) and `test_ctrl_msg.py`.

## Not done / next

1. Unprofiled ABBA of d3i against `package_T10b` on both GPUs (decode short, 8k, 32k, pf2k, 4 streams), greedy A, B, C on both GPUs, capture-route layer check, final kernel test run on both GPUs: `bash scripts_D3/chain_all.sh d3i` in `~/inkling/optD3` does the first three plus capture; the test run is `scripts_D3/ktd3.py` with `KT_ONLY=test_router_decode,test_oracle,test_prefill,test_gate_up,test_ctrl`. A test run of the last state on the R9700 (`ktres_f1`, `kt_f1.out`) was still queued behind other lanes when this was written.
2. Remaining decode headroom in this lane (kernel level): `router_decode` 4.2 us (the select is bound by the cold bias read and the dependent rounds), `ctrl_down_msg` 8.4 us (message chain plus two barriers), `ctrl_apply_z` 5.0 us, `router_logits` 3.6 us (floor about 1.9 us), norm_ffn 2.5 us x 24.
3. Prefill: `ctrl_down` 312 us per launch (about 15 ms per 2k chunk) is the next controller item; it needs an LDS-staged x tile to cut the L1 traffic (64 rows read the same x chunk).

## Rebuild

```
# box
cd ~/inkling/optD3/scripts_D3 && python3 gen_kernel_decls.py && python3 gen_hooks.py      # writes package_D3/dispatch/* (the scripts of this lane point at package_D3)
cd ~/inkling/optD3 && bash scripts_D3/make_var.sh r9700 d3i package_D3 2048              # hard-link tree on top of var-r9700-base (the pristine base, prefill_capacity 2048)
bash scripts_D3/abba.sh r9700 d3i dshort,d8k,pf2k,c4 3                                   # ABBA against var-r9700-base
bash scripts_D3/chain_g.sh r9700 d3i                                                     # greedy A/B/C, then a decode profile
bash scripts_D3/ck.sh $PWD/package_D3 inkling_ctrl_down_msg                              # compile report (CHUNK=256 TOK=1 for the decode config)
```

Notes for the box: `inkV/mat-<gpu>/model.json` was changed in place by another lane (a `dense_kq.loom` entry); `make_var.sh` therefore starts from `var-<gpu>-base` (made from the original mat before that change). The run scripts wait up to 15 minutes for foreign servers instead of exiting at once (the shared lock does not stop other lanes from starting servers while another lane holds it).

Leftovers on the box (`~/inkling/optD3`, mine, small: hard-linked var trees and logs): `var-*` (hard links, no weight copies), `prof/`, `pk_*` package copies, `kt/`, `ktres_*`, `work/`, `ablogs/`. Stale process: `ktd3.py r9700` (pid 259321, writes `ktres_f1`), stop it by pid when not needed.
