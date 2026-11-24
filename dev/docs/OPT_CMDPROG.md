# Lane CMDPROG: HAL timeline, bandwidth table, command program replay (agent optCMD, 2026-11-24)

Base: `package/` (= `package_M30` + CHATFIX, hrx-loom 0.3.0). Work copy `package_CMD` (laptop), `~/inkling/optCMD` (box). Patch: `lanes/CMDPROG.patch` (`diff -ruN package package_CMD`, then `diff -ruN scripts scripts_CMD`). Decisions: `bench/decisions_CMDPROG.jsonl`. Nothing was published, committed or pushed. No HRX, Loom, SDK or compiler file was touched.

## Result in one paragraph

One kept change: the grouped attention projection (`inkling_gemv_attn_group`) gets its own workgroup-per-unit tune value, 7 on gfx1201 (workgroups 269 to 434, passes 32 to 20). Decode on the R9700 gains +2.6 percent (short, 8k) and +2.3 percent (32k), prefill and 4 streams unchanged; the 8060S code is unchanged. Everything else in the lane is measurement: every dispatch of a decode step has a 0.5 to 0.75 us barrier gap (about 0.55 ms per step, 5 percent on the R9700, 2 percent on the 8060S), the host takes about 0.4 ms between steps on the R9700, the command buffer IS replayed (one record, one execute per step), hardware counters do not exist on gfx1201 and gfx1151 in hrx-loom 0.3.0, and `command.concurrent` does not overlap independent kernels on the queue (two experiments).

## Part 1: HAL timeline of one decode step

Method: `scripts_CMD/prof_run.sh` starts the server with `--device_profiling_mode=dispatch-events,queue-events,executable-metadata --device_profiling_output=<bundle>.irpf` (flush every 5 ms, no dropped records), runs `profile_driver.py dshort_d` (preset `chat-r9700` / `chat`, warm-up, 120 to 400 tokens), stops by PID. `scripts_CMD/irpf.py` reads the bundle (file format 2.0, struct layouts from `profile_schema.h` of hrx-system v0.2.0; sizes check against the per-record length). `step_report.py` takes the steady steps (first 3 skipped), `queue_seq.py` the host queue events, `step_timeline.py` one step as a timeline. The device clock is 100 MHz (10 ns ticks). Profiling itself costs about 4 percent (profiled period 10.97 ms against 10.5 ms unprofiled on the R9700).

| Per decode step, bucket 1, short context | R9700 (gfx1201) | 8060S (gfx1151) |
|---|---|---|
| dispatches per step | 882 | 882 |
| device wall, first start to last end | 10,594 us | 23,859 us |
| sum of kernel time | 10,043 us | 23,332 us |
| sum of gaps between dispatches | 552 us (5.2 percent of wall) | 526 us (2.2 percent) |
| gap per dispatch | 0.5 to 0.75 us, 880 of 880 gaps, mean 0.63 us | same, mean 0.60 us |
| idle between the last dispatch of a step and the first of the next, median | 372 us | 86 us |
| host EXECUTE submit period | 10,969 us | 23,988 us |
| wall minus device busy per step | 926 us (gaps 552 plus turnaround 372) | 612 us |

The M30 estimate (about 900 launches at 0.7 to 0.9 us, 0.7 ms) was close: measured 0.55 ms. There is no gap above 1 us inside a step (the histogram has 880 of 882 gaps in the 0.5 to 0.75 us bin, one at 7 to 10 us before the sampler, one before the first kernel). Which boundaries create the gaps: every boundary the same (0.6 to 0.65 us), so the sum follows the launch count: mix to ctrl and post_conv to mix (96 each per step), attention to attention, gate_up to down, router to router and so on 48 each (`scratch_CMD/step_r9700_v1.txt`, `step_strix_v1.txt`). Per-kernel table (R9700, launches per step, device us per step, gap before in us per step): `scratch_CMD/step_r9700_v1.txt`.

### Per-kernel rows (device time per step, gaps included separately)

R9700 (base `package/`, before the kept change):

| Kernel | launches | device us | gap before us |
|---|---|---|---|
| experts_gate_up | 48 | 2266 | 28 |
| gemv_attn_group | 48 | 2017 | 30 |
| gemv_q8_0 (48 o_proj, 1 lm_head 775 us) | 49 | 1436 | 31 |
| experts_down | 48 | 1193 | 31 |
| mix_states | 99 | 1008 | 59 |
| attn_tile | 96 | 536 | 55 |
| ctrl_down_msg + ctrl_apply_z | 96 | 627 | 61 |
| router_logits + router_decode | 96 | 377 | 58 |
| conv_res | 96 | 149 | 59 |
| attn_prep + attn_post | 96 | 218 | 63 |
| rms_norm, rms_norm_cols | 53 | 137 | 32 |
| experts_combine | 48 | 42 | 31 |
| sampler, head, embed, prep | 10 | 21 | 4 |

8060S: gate_up 6149, group 4656, gemv_q8_0 3937 (lm_head 2147), down_w64 3181, mix_states 1382, ctrl 1564, attn_tile 857, router 687, rest 1.2 ms.

## Part 2: hardware counters (L2 ingress): not available

`--device_profiling_mode` families on both GPUs (loom-serve 0.3.0, `scripts_CMD/prof_run.sh ... <mode>`):

| Family | gfx1201 | gfx1151 |
|---|---|---|
| dispatch-events, queue-events, executable-metadata | works | works |
| counters (0x10), counter-ranges (0x400), device-metrics (0x100), executable-traces (0x40), command-region-events (0x200), host-execution (0x2), memory-events (0x80), device-queue-events (0x4) | `UNIMPLEMENTED; unsupported GPU profiling data families` (`runtime/src/iree/hal/driver/amd/gpu/profile.c:633`) | same (counters probed; the others share the check) |

`loom-benchmark --profile-data=counters --profile-counter=...` is the same HAL path (documented as UNIMPLEMENTED for gfx11 and gfx12 in the skills, topic `sdk`). There is therefore no TCC read request, EA read or L2 ingress number from the HAL. `rocprofv3` was NOT used (owner rule); the skills note that the `amdgpu` device is the native libamdf driver, not ROCr, so rocprofv3 would not see these dispatches in any case. What would give the numbers: HAL counter support for gfx12 and gfx11 in the HRX runtime (proposal 1 below).

What the HAL gives instead: device time per launch site and the byte model of `docs/ROOFLINE.md`. The table below uses the expected bytes per step (cost model, Q4_K_XL, short context) and the profiled time; "counter bytes" and the ratio column are n/a. A kernel above the ceiling reads from cache (hit), a kernel far below it is latency or occupancy bound.

R9700 (ceiling 620 GB/s), base build:

| Kernel family | launches/step | device us | expected MB | achieved GB/s | % of roofline |
|---|---|---|---|---|---|
| experts gate_up | 48 | 2266 | 1231.1 | 543 | 88 |
| attn q/rel/gate/k/v/sink group | 48 | 2017 | 1015.4 | 504 | 81 |
| experts down | 48 | 1193 | 616.2 | 517 | 83 |
| lm_head Q8_0 | 1 | 775 | 492.1 | 635 | 102 |
| attn o_proj Q8_0 | 48 | 660 | 361.0 | 547 | 88 |
| loop controllers (down_msg, apply_z) | 96 | 627 | 227.4 | 363 | 59 |
| router logits F32 | 48 | 170 | 56.6 | 333 | 54 |
| mix_states, attn_tile, router_decode, conv_res, attn_prep, attn_post, norms, combine, sampler, head | 566 | 2,200 | under 15 together | latency bound | n/a |
| total | 882 | 10,024 | 3,994 | 398 | 64 |

After the kept change the group row is 1757 us, 578 GB/s, 93 percent.

8060S (ceiling 230 GB/s), base build (the change does not touch it):

| Kernel family | launches/step | device us | expected MB | achieved GB/s | % of roofline |
|---|---|---|---|---|---|
| experts gate_up | 48 | 6149 | 1231.1 | 200 | 87 |
| attn group | 48 | 4656 | 1015.4 | 218 | 95 |
| experts down (wave64) | 48 | 3181 | 616.2 | 194 | 84 |
| lm_head Q8_0 | 1 | 2147 | 492.1 | 229 | 100 |
| attn o_proj | 48 | 1790 | 361.0 | 202 | 88 |
| loop controllers | 96 | 1564 | 227.4 | 145 | 63 |
| router logits | 48 | 398 | 56.6 | 142 | 62 |
| small kernels | 566 | 3,400 | under 15 together | latency bound | n/a |
| total | 882 | 23,278 | 3,994 | 172 | 75 |

Reading: nothing re-reads (no row above 102 percent; lm_head sits at the measured ceiling, so the 620 GB/s figure is about 635 GB/s of cache-free streaming). The streaming kernels are at 83 to 95 percent; the loss is split over many kernels, not one. Only the R9700 group GEMV stood out (81 percent), which is the kept change. Without counters this is a rate estimate, not a measurement of bytes: a re-read inside a kernel would show up as a rate under the ceiling and cannot be told from a latency loss.

## Part 3: command program replay

Findings (measured, `scratch_CMD/queue_r9700.txt`, the queue tail in `scratch_CMD/step_strix_v1.txt`):

- loom-serve 0.3.0 records the decode program ONCE per shape bucket and replays it: the decode bucket is one HAL command buffer (id 1) with 882 dispatches, executed 206 times in the R9700 run (156 on the 8060S); the dispatch events carry the command-buffer flag and the same command index every step. The other two command buffers (1282 dispatches) are the warm-up and the first request buckets.
- Per step the host queue sees five operations within 6 us: transfer (88 B, the packet), alloca (1.1 MB transient), execute (the replay), dealloca, copy (4 B, the sampled token). Host encode time per step is therefore about 6 us, not a cost.
- What the host does between steps is wait for the step, read the token, build the next packet and submit: 372 us median on the R9700 (3.5 percent of the 10.97 ms period), 86 us on the 8060S (0.4 percent). The device is idle then. This is a runtime property (see proposal 2).
- PR #2081 (merged 2026-10-20): command programs run through HAL scenarios, "one equivalent command buffer per physical input-ring slot", the shared recorder records fills, copies, direct and indirect dispatches and "barrier waves"; serving uses the same recorder. This matches the replay seen. PR #2475 (C++ authoring) changes nothing at run time. PR #1849 and #1863 (storage-backed grids) merge the generated count-producer kernels of launches whose grid depends on storage: our decode grids are exact, so there is nothing to merge (no residual count kernels appear in the 882 dispatches). The 0.3.0 release notes were not read (no web fetch tool; the PR texts came from the GitHub API). The library models (`~/loom030/library/model/Qwen/Qwen3.8-27B/program`) use `command.concurrent` for independent projections (q, k, v; query and kv prepare; ffn gate and up) and have no replay, graph or elastic options in the programs.
- `command.concurrent` experiments (both on the R9700, HAL timeline):
  1. C2: the two `attn_tile` launches of a visit (mode 0 prefill rows write `ab` rows, mode 1 decode splits write `partial`: disjoint) in one concurrent region. The gap between them fell from 0.55 to 0.34 us, but the second launch still starts after the end of the first. About 10 us per step. Rejected (under 2 percent).
  2. E1a: a cache prefetch kernel (4 MB of the q weight, 256 workgroups) beside `mix_states`. The prefetch dispatch starts 0.4 us AFTER `mix_states` ends: sibling dispatches are not overlapped. The group GEMV round 1 did get faster (1757 to 1603 us per step, 3.2 us per visit), but the prefetch kernel costs 32 us per visit (4 MB at 130 GB/s: one load per thread in flight), device wall 10,335 to 11,755 us. Rejected, not in the patch.
  So an independent branch in this runtime saves the dependency gap (0.6 to 0.34 us) and nothing else. Real overlap of independent kernels is not available through `command.concurrent` on one queue in 0.3.0 (proposal 3).
- Merging hooks: the gap table says every boundary costs 0.6 us, so fusion pays 0.6 us per removed launch plus the removed launch latency (D2 realised 0.75 us per launch). Left candidates (each about 0.1 to 0.2 ms, below the 2 percent rule alone): router logits plus select (last-block pattern, needs a global atomic counter), `experts_combine` into the next `conv_res`, `attn_prep` into the tile.

## The kept change (C1)

Candidate record: `bench/decisions_CMDPROG.jsonl` (C1). Mechanism: `inkling_dense_mv_xs_plan` caps the passes per workgroup at 32; with the shared `xs_wg_per_unit` of gfx1201 (2) the group launch had 269 workgroups on about 128 resident slots, 2.1 rounds, so the third round runs 10 percent full (measured 81 percent of the bandwidth ceiling). A new template `inkling.dense.group.wg_per_unit` (default 2, gfx11 4 as before, gfx12 7) feeds the group launch and the group kernel; 7 gives 434 workgroups of 20 passes.

Files: `lib/inkling/dense_vec.loom` (template and three providers), `kernel/dense/gemv.loom` (declaration, kernel body uses the group template), `scripts/gen_hooks.py` (`attn_group` uses the group template, one declaration), `dispatch/hooks_dense.loom` (generated), `test/dense/test_gemv_group.py` (`PER_UNIT` gfx1201 7).

Sweep (HAL profile of the group launch, gfx1201, per step): 2 (base) 2017 us, 5 1888, 6 1806, 7 1757, 8 2397 (17 passes, odd counts and more workgroup starts cost more than the tail they save). Value 7.

ABBA, R9700, preset `chat-r9700` (prefix cache 16, prefill capacity 2048), fresh server per arm, 3 requests per metric, order A B B A (`scripts_CMD/abba_cmd.sh`, log `ab_log.jsonl` on the box):

| Row | A.0 (base) | B.0 (C1) | B.1 (C1) | A.2 (base) | delta (mean B over mean A) |
|---|---|---|---|---|---|
| decode short | 94.94 | 97.64 | 97.20 | 94.91 | +2.63 % |
| decode 8k | 84.35 | 87.03 | 86.64 | 84.93 | +2.60 % |
| decode 32k | 74.96 | 77.14 | 76.74 | 75.52 | +2.26 % |
| prefill 2k | 3711 | 3701 | 3700 | 3701 | -0.15 % (same kernels for n above 8) |
| 4 streams | 213.5 | 212.7 | 214.2 | 213.3 | +0.03 % |

Arms in time order. Spread inside an arm under 0.2 percent. 8060S: the template value stays 4, the generated code is the same; one A and one B arm (`c1s`): decode short 42.66 / 42.72, 8k 37.76 / 37.68, prefill 1267 / 1258 (noise; the ABBA second pair was not run).

Numerics: `test_gemv_group.py` 6 of 6 PASS on gfx1201 and gfx1151 (NMSE 1e-14, same arithmetic; the pass count only moves rows between workgroups). Greedy A, B, C (64 tokens) and the 8k and 32k documents (9,071 and 36,995 prompt tokens): IDENTICAL to the base build on both GPUs (`res/gl_*_r9700.json`, `gl_*_strix.json`, `scripts_CMD/m30_greedy.sh`, `m30_cmp.py`). Chunk independence was not touched (no state or layout change). Capture-route layer check: blocked on 0.3.0 as documented in `OPT_M30.md`.

## Where the remaining gap to the ceiling sits

R9700, decode short, ms per step (profiled period 10.76 ms with the kept change, ceiling 6.47 ms; unprofiled the step is 10.27 ms: the profiler adds about 0.5 ms spread over the rows):

| Cause | ms |
|---|---|
| Weight-streaming kernels above their byte floor (gate_up 0.28, down 0.20, group 0.13, o_proj 0.08, lm_head 0) | 0.7 |
| Latency-bound small kernels (mix_states 1.0, attention tile/prep/post 0.75, controllers above floor 0.26, router 0.30, conv_res 0.15, norms 0.14, combine 0.04, rest 0.02) | 2.65 |
| Barrier gaps between dependent dispatches (880 x 0.63 us) | 0.55 |
| Host turnaround between steps (token readback, next packet) | 0.4 |
| total over the ceiling | 4.3 |

8060S, decode short (period 23.99 ms, ceiling 17.44 ms): streaming kernels above floor 1.8 ms (gate_up 0.8, down 0.5, group 0.25, o_proj 0.22), controllers and router logits above floor 0.7, small latency-bound kernels 3.4, gaps 0.53, host 0.09: total 6.5 ms.

The next levers, by size: (1) 2.6 ms of small dependent kernels (mix_states alone is 1.0 ms: 99 launches of 10 us, a latency chain of loads and reductions), (2) 0.55 ms of barrier gaps, (3) 0.4 ms host turnaround, (4) 0.7 ms of streaming loss. None of them is a package-level quick win beyond what was kept.

## Proposals (runtime; not part of this lane, no runtime patch)

1. Hardware counters on gfx11 and gfx12 in the HAL (`profile.c:633`): TCC (L2) read requests and EA (memory) read requests per dispatch, through the counter and counter-range families. With them the table in Part 2 gets the bytes column and ratio. Ask Ben whether the gfx12 counter set (RDNA4) is planned.
2. Hide the step turnaround (0.37 ms per step on the R9700, 3.5 percent): the host waits for the step, reads the sampled token, builds the next packet and submits. Options: keep two steps in flight with the sampled token fed to the next packet on the device (the `feedback` library already writes the token to storage), or build and enqueue the next packet while the step runs, with a device-side dependency on the sampler output. Expected gain up to 3.5 percent (R9700) and 0.4 percent (8060S).
3. Barrier cost and overlap of command programs: (a) every dependent dispatch costs 0.6 us between the end of one kernel and the start of the next (880 per step); measure which part is the end-of-kernel release and the next acquire fence scope (agent scope instead of system scope between dispatches of one command buffer), and the barrier packet. (b) Siblings of `command.concurrent` do not overlap on one queue (the second starts 0.34 to 0.4 us after the end of the first, both experiments); allowing true overlap (no barrier bit, enough free CUs) would let a prefetch or a small branch run beside a latency-bound kernel. A third approach that needs no runtime change: prefetch loads inside an extra workgroup range of the producer kernel (the pattern of the `conv_commit` workgroups of D1), not tried here for lack of time: expected 0.15 to 0.3 ms (R9700) from priming 4 MB of the q weight in `mix_states` and 5 MB of the o weight in the attention tile.
4. Stage compile and `loom-serve` flag help: `--device_profiling_output` needs a decoder; a built `iree-profile` binary in the wheel would remove the need for `irpf.py` (building it from the SDK source was blocked by the permission system; see below).

## Denied or not done

- Building `iree-profile` from `~/hrx-sdk/src` into my own build directory (cmake configure of the SDK source) was denied by the permission system once. The bundle was decoded with `scripts_CMD/irpf.py` instead (format 2.0 from the header files). Not retried.
- Hardware counters: unavailable (above); `rocprofv3` not used; owner decision needed if a counter run through ROCm tools is wanted.
- Not done: ABBA second pair on the 8060S (no code change there), kernel-level cold versus hot ring check with `loom-benchmark` (the counter-less proxy for cache residency), the producer-kernel prefetch design, the full dev gates (run once at integration).

## Rebuild

```
# laptop
bash scripts_CMD/lsync.sh gen            # copy package_CMD and scripts_CMD to ~/inkling/optCMD, generate the dispatch files, copy them back
# box
python3 scripts/mk_mat.py <name> ~/inkling/optCMD/package <chat-r9700|chat>      # hard-link tree of the base mat with the changed program files
bash scripts/abba_cmd.sh r9700 <tag> dshort,d8k,d32k,pf2k,c4 3 ABBA A=<base mat> B=<mat>
DEC_TOKENS=150 flock -x ~/hrx-pr1/gpu.lock bash scripts/prof_run.sh r9700 dshort_d <mat> <out dir>
python3 -I scripts/step_report.py <out dir>/dshort_d.irpf <out>.json; python3 -I scripts/bytes_table.py <out>.json 620
GREEDY_LONG=1 flock -s ~/hrx-pr1/gpu.lock bash scripts/m30_greedy.sh r9700 <mat> res/<name>.json
```

Leftovers on the box (mine): `~/inkling/optCMD` (`package`, `pk_g5 pk_g6 pk_g7 pk_g8 pk_g10`, hard-linked `mat-*` trees, `prof/` bundles of 12 to 34 MB each, `res/`, `ablogs/`). Laptop: `scratch_CMD/` (text outputs and PR texts), `package_CMD`, `scripts_CMD`.
