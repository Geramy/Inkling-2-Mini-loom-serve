# Lane W64: wave64 decode kernels (bead llama-cpp-tg3.35)

Question (owner, 2026-11-22): does wave64 make the memory-bound decode kernels faster? The claim is about another chip ("Gorgon Halo", +30 percent). This lane tests our two GPUs: gfx1151 (8060S) and gfx1201 (R9700).

## Result

The claim is not reproduced on our chips. One family wins: the portable expert down kernel at decode on gfx1151, about 11 percent faster in isolation and 9 to 10 percent faster per launch in a serve profile. That is 0.4 ms of the 26.4 ms decode step on the 8060S in the profile and +1.0 percent (decode short), +2.2 percent (32k), +2.8 percent (4 streams) in ABBA. The ceiling gap on the 8060S (57.3 tok/s, 17.5 ms) is 9 ms; this lane closes about 0.3 ms of it. Every other family is neutral or slower at wave64 on gfx1151. On gfx1201 no family reaches 5 percent in a serve profile. The change is kept for gfx1151 only, through the per-arch template route (`rdna3` row 64, portable default 32). Greedy tokens and kernel outputs are bit-identical to wave32.

## How the subgroup size is chosen

- A kernel gets the subgroup size of its compile target. Without `target(...)` the device default applies (32 on gfx11 and gfx12).
- `amdgpu.target<gfx11-generic> @inkling_wave64 {subgroup_size = 64}` plus `kernel.def target(@inkling_wave64)` compiles one kernel at wave64. The same device then runs wave32 and wave64 kernels. The record is a requirement: the alias is tied to one family. A gfx11 alias on gfx1201 gives `TARGET/052 incompatible with specialization target`, so a gfx12 kernel needs its own alias. Two aliases cannot share a name and `amdgpu.target<...>` accepts only exact or generic profiles (no wildcard).
- Launching a kernel whose target does not match the device fails at compile time. The launch must be statically dead on other devices: the hook asks the template `@inkling.tune.experts.down_wave` (providers per arch, default 32, `rdna3` 64) and branches on the result. Planning folds the branch; a dead kernel is not compiled. Tested: the variant starts and runs on the R9700 (90.4 tok/s, equal to the base 90.2) with a gfx11-only wave64 kernel linked.
- The PR1 JIT of ggml-hrx defines `@ggml_wave32` and `@ggml_wave64` the same way (`loom-jit.cpp`). The Inkling attention tile kernels use matrix operations with the wave32 layout (`KERNELS_attn.md`: "tested with wave32 only"), so they are out of scope.

## Families tested

All numbers are device time p50. Isolated = loom-benchmark on real weights (experts: layer 0, n = 1, NVFP4; dense: input ring of 200 MB, `cols = 1` build, 3 repetitions). Serve = HAL dispatch statistics of a decode-short session (`profile_run.sh`). The 8060S has a 32 MB cache and the R9700 a larger one: the experts working set (24 MB gate_up) can sit in cache in the isolated runs, so serve numbers decide.

| Family | gfx1151 isolated (w32 to w64) | gfx1151 serve per launch | gfx1201 isolated | gfx1201 serve per launch | Verdict |
|---|---|---|---|---|---|
| experts down, NVFP4 n=1 | 64.1 to 56.3 us (-12 percent; workgroups 128, 192, 256 thread: 55 to 57) | 85.8 to 77.4 us (-10 percent; ratio to gate_up 0.604 to 0.551 and 0.537) | 17.0 to 17.2 us (0) | not run | keep on gfx1151 |
| experts gate_up, NVFP4 n=1, same algorithm | 122.5 to 136.5 us (+11 percent: 56 VGPR, occupancy 75 percent) | not run | 37.5 to 35.4 us (-5.6 percent) | 50.1 to 48.2 us (-3.8 percent; the unchanged down kernel moved -2.4 percent in the same pair: noise) | reject |
| experts gate_up, gate and up in separate loops (48 VGPR, occupancy 100 percent), 64 to 256 threads | 121.5 to 117.3 us (-3.3 percent; 384 threads: 133) | 129.1 to 128.1 us (-0.8 percent) | 37.5 to 35.4 us | not run | reject (below 5 percent) |
| dense GEMV Q8_0 attn_q 3072x2304 | 37.7 to 37.4 us | not run | 15.6 to 15.0 us (-4 percent) | not run | reject |
| dense GEMV BF16 attn_q | 72.8 to 77.4 us (+6) | | 26.8 to 26.7 | | reject |
| dense GEMV Q8_0 attn_k 1024x2304 | 20.2 to 19.3 us (-4; 1 us) | | 6.9 to 6.9 | | reject |
| dense GEMV BF16 attn_k | 27.0 to 27.1 | | 10.3 to 10.4 | | reject |
| dense GEMV Q8_0 attn_o 2304x3072 | 42.8 to 42.2 | | 14.6 to 14.5 | | reject |
| dense GEMV BF16 attn_o | 76.0 to 91.9 us (+20) | | 26.3 to 26.5 | | reject |
| lm_head Q8_0 200058x2304 | 2168 to 2157 us (-0.5; floor 2140) | | 797 to 785 us (-1.5; floor 794) | | reject (at the byte floor already) |
| lm_head BF16 | 4192 to 4367 us (+4; floor 4028) | | 1490 to 1460 us (-2.0; floor 1494) | | reject |
| router GEMV F32 128x2304 | 15.9 to 16.7 us | | 4.2 to 4.2 | | reject |
| attn_rel GEMV BF16 384x2304 | 17.3 (12.7 to 21.9, bimodal) to 12.9 us | | 5.5 to 5.5 | | reject (noise) |
| mix_states T=1 nb=3 nq=1,3; nb=7; nb=1 nq=4 | 9.9, 10.9, 11.6, 10.6 to 9.3, 10.0, 11.4, 10.2 us (-6, -9, -2, -4 percent) | not run | not run | not run | not kept: under 1 us per launch, one sample, D1 changes the kernel (see below) |
| ctrl down, z, apply, norm (T=1) | down 6.9 to 7.3, z 1.23 to 1.26, apply 3.3 to 2.7 (nc=3 f32), norm 1.52 to 1.48 us; totals 13.0 to 12.7, 10.5 to 9.9 (bf16), 10.2 to 10.1 us | | | | reject (within 5 percent in total) |
| rms_norm 2304 n=1 | 2.8 to 2.9 us | | | | reject |

The dense GEMV `xs` kernel needed no code change for wave64: its body uses workgroup, workitem and LDS indices only, so one more `kernel.def target(...)` around the same inline function gives the wave64 build. The expert kernels keep the 8-lane group design; a wave64 workgroup of 128 threads has 2 waves of 8 groups (16 rows), the same rows per workgroup as 4 waves of 4 groups at wave32.

Compiler (gfx1151, `loom-compile --compile-report`): `down_w64` 40 VGPR, no spills, occupancy 100 percent (same as wave32). `gate_up_w64` 56 VGPR gives occupancy 75 percent at wave64 (768 VGPR lanes per SIMD, granularity 12); the split-loop variant has 48 VGPR and occupancy 100 percent. Instruction counts differ by under 5 percent.

Why down wins and gate_up does not (hypothesis, not measured): gate_up carries two weight streams and two accumulator sets for each lane; at wave64 it drops to 75 percent occupancy. The down kernel is a single stream with 40 VGPR. Isolated, down is 1.2x over the byte floor at wave32 and 1.07x at wave64.

## Kept change

- `kernel/experts/down_w64.loom`: `inkling_experts_down_w64`, `target(@inkling_wave64)`, body copied from `down.loom` with 8 rows for each wave (one changed line). No change to `down.loom`.
- `lib/inkling/targets_w64.loom`: `amdgpu.target<gfx11-generic> @inkling_wave64 {subgroup_size = 64}`.
- `scripts/gen_hooks.py`, `dispatch/hooks_experts.loom`: template `@inkling.tune.experts.down_wave` (default 32, `target(@inkling_rdna3)` 64); the decode branch (n below 48) launches `inkling_experts_down_w64` when the value is 64.
- `scripts/gen_kernel_decls.py`: the regex accepts `target(@alias)` before `export(...)`; `dispatch/kernel_decls.loom` regenerated (one new decl).
- `card.json`: both library lists get the two new files.
- `test/experts/bench_experts.py`: `W64=1` times the wave64 down kernel.

Per-arch row (`docs/KERNELS_experts.md` should get this line): experts decode down subgroup size, gfx1201 32, gfx1151 64, portable default 32. The `rdna3` family covers all gfx11 parts (also gfx1100); only gfx1151 was measured. gfx1201 has no gfx12 wave64 kernel (no win), so the gfx12 alias is not needed.

## Numerics

- Kernel level (gfx1151, real layer-0 weights, router to combine): `act`, `dout` and `y` are bit-identical between wave32 and wave64 at n = 1 and n = 8 (also for the split-loop gate_up and for every workgroup size tried). The reductions keep the same 8-lane order, so no tolerance is needed.
- Greedy 64 tokens, oracle prompts A, B, C through the chat route: base against variant identical on both GPUs (8060S runs the wave64 kernel; the R9700 runs the base kernel through the template default, so its check proves that the changed hook and the dead gfx11-only kernel link and run there). Not run: the area kernel tests of `test/experts` (the bit-identical check replaces them for this kernel), the capture-route layer check (outputs are bit-identical), chunk independence (no state change).
- R9700 ABBA was not run (lock wait of about 30 minutes for each arm; the launched kernels are the base kernels there).

## End to end

8060S, fresh server per arm, order B A (orphan arm) then A B B A, 3 repetitions each, median of a session. A = base (T10b card), B = base plus `inkling_experts_down_w64` through the template route. Log: `~/inkling/optW64/ab_log.jsonl` (tags `dw64t.*`).

| Row | A sessions | B sessions | Change |
|---|---|---|---|
| decode short, tok/s | 39.61, 39.66 | 39.97, 40.00, 40.10 | +1.0 percent (25.23 to 24.97 ms per token) |
| decode 8k | 35.04, 34.96 | 35.27, 35.21, 35.52 | +1.1 percent |
| decode 32k | 29.88, 29.17 | 30.08, 30.03, 30.25 | +2.2 percent (A spread 2.4 percent) |
| 4 streams, aggregate tok/s | 82.24, 82.40 | 84.24, 84.83, 84.28 | +2.8 percent |
| prefill 2k tok/s (pf2k) | 1115.8, 1151.9 | 1148.9, 1159.3, 1153.3 | no change (the kernel is not used above n = 48; spread 3 percent) |

The single-token gain (0.26 ms) is below the 0.4 ms of the serve profile and near the session spread of the 8060S (about 1 percent in one arm, more across sessions). Prefill 1k and 8k and the R9700 rows were not run: prefill uses the matrix kernels, and on the R9700 the template returns 32 so the launched kernels are the base kernels (the card starts and runs: decode short 90.4 tok/s in a single run against 90.2 in PERF_BASELINE).

## Notes for the integrator

- Conflicts: `scripts/gen_hooks.py` experts section only (`experts_file`, one block of header text and the `chain` helper); `dispatch/hooks_experts.loom` (generated). No other lane edits those lines. D2, D3 and D6 edit kernel bodies that this patch does not touch.
- D1 (mix and norm chain): mix_states at wave64 was 2 to 9 percent faster in one isolated sample on gfx1151 (under 1 us). The kernel computes `wave = tid / 32` and uses `cluster_size = 32` reductions, so wave64 needs a re-check of the 288-thread workgroup (4.5 waves). Retest after D1's fusion, with the `kernel.def target(@inkling_wave64)` trick on the same body.
- Gate measure for any later wave64 kernel: serve profile per launch, not isolated runs (cache effects; the R9700 gate_up shows -5.6 percent isolated and -3.8 percent in serve).
- The 8060S timing noise in this session was large (decode short 36 to 39 tok/s between sessions of the same arm) because other lanes ran CPU work. Per-kernel serve profiles are the better evidence than whole-token ABBA for a 1.5 percent change.

## How to rebuild and repeat

1. `python3 scripts/gen_kernel_decls.py; python3 scripts/gen_hooks.py` (writes `dispatch/kernel_decls.loom`, `dispatch/hooks_experts.loom`).
2. Isolated: `W64=1 python3 test/experts/bench_experts.py gfx1151 1` against `python3 test/experts/bench_experts.py gfx1151 1` (`down` rows).
3. Serve profile: `profile_run.sh strix dshort_d` on the base card and the variant card; compare `inkling_experts_down` and `inkling_experts_down_w64` in the kernel table of `serve.log`.
4. Patch: `lanes/W64.patch` (diff -ruN of `package` and `scripts`, a/ b/ prefixes) against `package_T10b` and `scripts_T10b`.

Experiment sources that are not part of the patch (box, `~/inkling/optW64`): `package/kernel/experts/{gate_up,down}_{w64,sp,sp64,sp192,sp256,sp384,t64,t192,t256}.loom` (variants), `package/kernel/dense/gemv.loom` (wave64 GEMV kernels appended), `pkgN` (wave64 norm and mix kernels), `pkg12` (gfx12 alias), `ab/` (A/B and profile scripts with the preset `chat`), `prof/`, `work/cache`.

## Leftovers on the box

`~/inkling/optW64` (scratch, no weight copies: `mat-base` and the `var-*` trees are hard links, 17 GB of weights are shared). `/tmp/ce8_*.npz`, `/tmp/w64*`. Orphan wait loops `until ... eb_sp.out` of this lane may still run: they exit when the file has enough lines.
