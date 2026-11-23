# Lane D6: experts decode (bead llama-cpp-tg3.33)

Result: a software-pipelined fast path for the decode kernels `gate_up` and `down` (NVFP4 weights, one live route per record) and a cheaper NVFP4 code builder. The outputs are bit-equal to the base. Gain in the real graph (ABBA, fresh server per arm): decode short R9700 90.5 to 91.75 tok/s (+1.4 percent), 8060S 39.75 to 40.36 tok/s (+1.5 percent). Both are at or just under the old stop threshold of the lane (1.5 percent); the owner then moved the target to the roofline ceiling. The lane stops because the remaining candidates are each worth under 1 percent and a session takes one to two hours of lock wait (list at the end).

## What the kernels did in the real graph

Method: `bench_experts.py` with `RING=24` (24 copies of the weights, about 1 GB touched per dispatch, far above the 64 MB last-level cache) reproduces the in-situ time of the kernels. With the old ring of 4 the weights are partly cache resident and the kernels look faster (R9700 gate_up 37.7 us, down 17.2 us).

| R9700, one NVFP4 visit (n = 1, 8 experts) | bytes | floor at 620 GB/s | HAL profile in the graph (average of all 48 visits) | cold bench | streaming probe, same grid, 16 B loads, no compute |
|---|---|---|---|---|---|
| gate_up | 23.9 MB | 38.5 us | 50.0 us | 48.2 us | 38.6 us |
| down | 11.9 MB | 19.3 us | 26.5 us | 25.0 us | 19.8 us |

The probe reaches the floor, so the loss is in the kernel and not in the memory system (cold weights, workgroup count, tail effects are not the cause; see the shape test below).

Cause found in the low IR of the base kernels (`loom-compile --dump-ir-after=low-dce`, final IR is near ISA): inside the unit loop every live route has the chain `load route id` (VMEM), wait, `load act or x row` (two 16 B loads), wait, then the 8 dot2 and 7 adds. The weight loads of unit i+1 are issued only after the dots of unit i. With one live route (every record at n = 1) the loop runs 18 (gate_up) or 9 (down) of these latency chains per lane, and the memory pipe is idle during the decode and dot work of the whole wave.

## Change

Files: `kernel/experts/gate_up.loom`, `kernel/experts/down.loom` (fast branch added, generic code unchanged), `lib/inkling/experts_common.loom` (new public functions appended, the existing ones are not touched), `test/experts/bench_experts.py` (env `RING`, `ITERS`, `BENCH_CACHE`).

1. Fast branch, taken when `wtype == 0` and the record has one live route (`wtype * 16 + live == 1`; the weight type is a launch constant, so the dense arm and the generic code keep their own paths). The route id and token are read once before the loop. The unit loop is software pipelined: the raw words (scale word and two code words, `@inkling_nvfp4_unit_raw`) and the x or act row of the next group are loaded before the current group is decoded (`@inkling_nvfp4_unit_decode`) and multiplied. The raw words, not the decoded codes, travel through the loop, so the decode of group i overlaps the loads of group i+1. gate_up uses 2 units per iteration (9 iterations), down 1 unit (9 iterations). The last iteration reloads its own group (clamped index); this costs one repeated cache hit per lane.
2. `@inkling_nvfp4_codes_fast`: same 8 f16x2 code words as `@inkling_nvfp4_codes`, 4 operations per word pair (two shifts, two masks, one or; the top pair needs one shift less) instead of 7. A host check on 200,000 random words confirms the same bits. VALU operations per iteration of the `down` fast loop (one unit): 93 with the old builder, 79 with the new one.
3. The matrix path is untouched: `gate_up_wmma`, `down_wmma`, `routes`, `combine`, `router` give the same HSACO as the base (md5 of the compiled kernels, gfx1201 and gfx1151).

Compiler (`loom-compile --compile-report`, standalone, all kernel arms linked): no spills. gate_up VGPR 56 to 85 (gfx1201) and 96 (gfx1151), instructions 1828 to 2266 and 1809 to 2342; down VGPR 40 to 40, instructions 1069 to 1278 and 1053 to 1246 (the generic code stays; the fast branch is added); occupancy 100 percent in all cases.

## Numerics

Bit-equal to the base on the bench fixture (layer 0, real weights and routing): `act` and `dout` for n = 1 and n = 4 are identical (`np.array_equal`) because the sums keep the order of the generic loop.

Kernel tests (`test/experts/run_all.sh`: `test_gate_up`, `test_oracle` with the n = 1 row slices that take the fast path, `test_prefill`): gfx1201 266 PASS and gfx1151 266 PASS, 0 FAIL, rc 0 on both. `y` against the oracle `conv_in` is at the base level (1.0e-2 for NVFP4, the oracle quantizes activations; the same values as in `KERNELS_experts.md`).

Greedy 64 tokens, oracle prompts A, B, C through the chat route (`route_check.py --diff`), D6 build against the base build of the same session type: R9700 IDENTICAL, 8060S IDENTICAL. 256 tokens: IDENTICAL on both GPUs.

No state, ring, page, chunk or capacity change: the chunk independence checks do not apply.

## Evidence

HAL profile, decode short (`profile_run.sh` sessions `dshort_p` and `dshort_d`, 19,488 launches per kernel, average over the 48 visits of a step, 4 of them Q8_0 and not changed):

| us per launch | R9700 base | R9700 D6 | 8060S base | 8060S D6 |
|---|---|---|---|---|
| gate_up | 50.01 | 46.92 | 141.91 | 129.42 |
| down | 26.48 | 25.43 | 86.37 | 68.62 |
| combine (not changed) | 0.87 | 0.82 | 2.10 | 1.62 |

The 8060S sessions differ by clock state between sessions (the unchanged `down` of the intermediate build `v1` measured 74.3 us against 86.4 us in the base session): use the ABBA below for the 8060S, not this table.

Cold bench (device time p50, `RING=24`, n = 1, layer 0, R9700):

| build | gate_up | down |
|---|---|---|
| base | 48.2 | 25.0 |
| fast path, 1 unit per iteration | 43.9 | 25.0 |
| fast path, 2 units (kept) | 43.0 | 25.0 |
| + cheaper decode (kept) | 42.3 | 24.5 |
| 3 / 6 units per iteration | 43.7 / 43.2 | 25.6 (3) |
| dot2 chain instead of 8 dots and 7 adds | 42.2 | 23.6 (1 ulp change of `dout`, rejected) |
| header loads before the branch | 42.7 | 23.6 (no change) |
| workgroups of 64 / 256 instead of 128 threads | 42.4 / 42.1 | 24.5 / 24.4 |
| 8 B load for the two code words | 50.2 | 25.3 (slower, rejected) |
| n = 4 (generic path mostly) base / D6 | 146.9 / 147.9 | 77.9 / 77.3 |

All rows are in `bench/decisions_D6.jsonl`.

End to end ABBA (`ab_abba2.sh`, fresh server per arm, order A B B A, 3 repetitions per session, medians per session; A = base, B = D6; published preset values, prefill_capacity 2048):

R9700:

| Row | A | B | Change |
|---|---|---|---|
| decode short, tok/s | 90.57, 90.44 | 91.88, 91.62 | +1.37 percent |
| decode 8k | 81.05, 80.77 | 82.13, 81.92 | +1.4 percent |
| 4 streams, aggregate | 206.47, 207.86 | 205.75, 205.12 | -0.8 percent (noise about 2.5 percent in earlier lanes) |
| prefill 2k | 3551.6, 3563.0 | 3539.1, 3537.2 | -0.6 percent (prefill kernels are byte identical; session spread) |

8060S (same method):

| Row | A | B | Change |
|---|---|---|---|
| decode short, tok/s | 39.73, 39.77 | 40.34, 40.38 | +1.5 percent |
| decode 8k | 35.09, 35.07 | 35.57, 35.61 | +1.5 percent |
| 4 streams, aggregate | 83.35, 83.96 | 85.45, 85.38 | +2.1 percent |
| prefill 2k | 1138.2, 1132.7 | 1152.7, 1161.9 | +1.9 percent (the prefill kernels are byte identical: this is session spread) |

## Per-arch rows

None. The same code runs on both GPUs; units per iteration (2 and 1) were flat from 1 to 6 on the R9700 and are fixed in the source. No arch string, no tune row. The 8060S was only measured end to end (the cold bench on the 8060S is not stable: identical kernels measured 154 and 265 us in one session pair).

## Stop reason

Against the roofline target (R9700 154.6, 8060S 57.3 tok/s) the experts bucket is still at 3.47 ms (R9700, in-situ sum of gate_up and down per step) against 3.0 ms by the streaming probe (2.98 ms by the traffic floor). What is left against the streaming probe is 3.7 us (gate_up) and 4.7 us (down) per visit, about 0.37 ms per token. It is VALU work (79 operations per unit and row), two scattered 16 B loads of x per unit, and three 4 B loads per unit and row; wider weight loads were slower (4 B aligned 8 B access), and rows are not 16 B aligned for down (648 B per row). Not tried because the expected gain is under the noise of one session: 2 rows per lane group in `down` (shares the act loads), a chained dot2 in `down` (-0.9 us but changes `dout` by 1 ulp), layers 20 to 23 Q8_0 and BF16 (4 of 48 visits, gate_up 82 us against 72.8 us floor: at most 0.04 ms), fusion of gate_up, SiTU and down in one launch (needs a grid barrier or float atomics), combine into down (0.04 ms per token). Candidate for another lane or a later round, not built: for n = 1 the `routes` launch can be skipped (the 8 experts of one token are distinct, so every record has one live route and `ids` can be read directly by the fast paths): 1.4 us kernel time plus one launch gap per visit, about 0.09 ms per token (0.8 percent); it needs a new kernel arm and a hook change in `gen_hooks.py`, plus the same numerics gates.

## How to rebuild

1. Package: apply `lanes/D6.patch` (diff -ruN of `package_T10b` against `package_D6`, `-p1` from the repo root of the package, or copy the four files). `scripts/gen_hooks.py` and the hooks are not changed.
2. Kernel tests, both GPUs: `cd package/test/experts; ./run_all.sh gfx1201; ./run_all.sh gfx1151` (python from the inkD venv; the scripts write fixtures below `~/inkling/optP1/work`, set the path in the three `work =` lines for another box).
3. Cold bench: `RING=24 BENCH_CACHE=<dir> python bench_experts.py gfx1201 1`.
4. Profile and ABBA: `scripts_D6/prof_job.sh`-style calls of `profile_run.sh` with `INKV` and `MAT` set, and `scripts_D6/ab_abba2.sh <gpu> <variant> dshort,d8k,c4,pf2k 3`.
5. Development aids (not part of the package): `scripts_D6/gen_fast.py` and `apply_fast.py` print and apply the fast block for a given units-per-iteration; `mkprobe.py` turns the fast path of `down.loom` into a load pattern probe; `dumpir.sh` writes the low IR of a kernel.

## Notes for the integrator

- Files touched: `gate_up.loom`, `down.loom`, `experts_common.loom` (append only), `bench_experts.py`. Wave 64 lane: the fast block loads 8 lanes per row and 4 rows per wave (wave32); a wave64 variant must keep its own copy of the fast branch.
- The kernels declare `@inkling_nvfp4_unit_raw` and `@inkling_nvfp4_unit_decode` with `func.decl`; both are `func.def public inline` in `experts_common.loom` (needed for the serve link).
- Scratch on the box (mine): `~/inkling/optD6` (variants `v/`, `var-*`, `mat-*` hard-link trees, `prof/`, `work/`, about 40 GB of test fixtures in `work/`). Left over from other lanes: none.
