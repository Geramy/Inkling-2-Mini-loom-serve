# OPT_P3: prefill attention, shared K/V tile for the 3 heads of a KV group (lane P3)

Result: rejected, nothing is wired into the serving route. The kernel works and is bitwise equal to the base for prefill, but it is not faster: equal on 512-token rows, 3 percent slower (global) and 19 percent slower (local) on 2048-token rows (R9700, the published chunk size). Stop rule hit (below +3 percent at 8k): the end-to-end ABBA was not run because the kernel gate failed. The 8060S was not measured (lock contention, see Process notes).

## What was built

`package_P3/kernel/attn/attn_prefill.loom` (new, export `inkling_attn_prefill`, same launch ABI as `inkling_attn_tile`, grid `(tile_bound, 8, 1)`). One workgroup is 16 query rows of the 3 heads of one KV head (48 rows). The K and V tiles are staged once per block for the 3 heads. The K fragment of the QK step is loaded once for the 3 heads, the V fragments of the PV step likewise. Relative bias, sink, window, paged and ring reads and the causal block skip are as in mode 0. Decode (mode 1) and `attn_tile.loom` are unchanged. Not in `hooks_attn.loom` or `card.json`. The file also holds the V register prefetch (candidate 2).

Test harness (`package_P3/test/attn`): `attn_test.py` and `attn_perf.py` read `ATTN_ROOT`, link the new file and launch `inkling_attn_prefill` for mode 0. `attn_perf.py` has the rows `pre_*_2048` and `PRE_OLD=1` (launches the old mode 0). `scripts_P3/sync_p3.sh` copies the tree to `~/inkling/optP3`.

## Evidence (R9700, `attn_perf.py`, microseconds per launch)

The clock varies during a run (sclk 1033 to 2350 MHz seen): spread 0 to 3 percent, and the local row sometimes reads 1.8x on a cold clock (those values are dropped).

| Row | Base mode 0 | Shared 3-head |
|---|---|---|
| pre_g_512_at4096 | 2048, 2102 | 2095, 2102, 2107 |
| pre_l_512_at1024 | 518, 538 | 544, 538, 550 |
| pre_g_2048_at4096 | 8673 | 8964 (+3 percent) |
| pre_l_2048_at1024 | 1817 | 2162 (+19 percent) |

Compile report (gfx1201): no spills; `resident_subgroups_per_simd` 2 (workgroup memory 55,040 B: Q 13,056, K/V 17,408, scores 12,288, probs 3 x 2048, r 3072, scale 3072) against 4 for the base (32,000 B). Candidate 3 (K of the next block also in registers) overflows the register file (private_memory 574) and runs 2.8x slower.

## Findings

- Base mode 0, dead-code ablation, global row 2048 us: K/V staging 60 percent (stage removed: 822 us), relative-bias gather about 10 percent (1832 us), exp none (2046 us). Local row 518 us: staging 27 percent.
- With the stage shared by 3 heads the stage cost fell only from 1226 to 860 us (stage removed: 1242 us) and the bias share rose to 21 percent (1651 us). The total did not move. So the time is not the K/V byte volume: it is the serial phases inside one workgroup (stage, barrier, QK, barrier, softmax, barrier, PV) with few waves to overlap them. 3 heads gives 256 workgroups per 512 tokens at 2 resident per CU, against 768 at 4 resident: the same wave count, worse latency hiding.
- Causal skip: yes (`b_end = p_hi / 64 + 1`), and the window start `b_lo = kmin / 64` (local visits run at most 17 blocks).
- Softmax stage: 2619 vector ALU instructions per block per wave (about 100 per score element; the bias is 16 FMA and 15 adds per element): about 25 percent of the kernel time if nothing overlapped.
- V register prefetch: no effect. K prefetch: register overflow.

## Numerics

`attn_test.py run --dev gfx1201 --only "A slot 0|L slot 0"` with ATTN_LONG: all PASS, NMSE equal to the base table (out 1.3e-8 for L slot 0, 2.5e-8 for A slot 0). 3 chunks and 512+rest are bitwise equal to 1 chunk (max abs 0); decode tail 8.8e-4 as in the base. The summation order per row equals the base, so the prefill result is bitwise equal to the base. Not run (kernel not adopted): other slots, gfx1151, layer checks, greedy, chunk independence at 2048.

## Ideas left (not tried)

- 3 heads with 4 resident workgroups needs about 32 KB (now 55 KB): scores or probabilities in registers (softmax on the accumulator layout) would remove 18 KB. Large rewrite.
- Relative bias with WMMA (bias[row][dist] = r . table[dist], 5 tiles of 16x16x16 per block, hi/lo F16 split for F32 accuracy) instead of 16 FMA per element.
- Split over key blocks for global visits needs a merge pass and loses bitwise chunk invariance: not recommended.

## Process notes

The GPU lock was held by other lanes (greedy and serve jobs) for most of the time: each timing waited 5 to 25 minutes, so only the runs above were made. No per-arch rows added (no change adopted). My box leftovers: `~/inkling/optP3/{abl_*,build,fixtures,out,p3_k2048.out}`; one queued `flock ... loom-benchmark` for `abl_nostage` (PID 4005134) that I was not allowed to kill; it ends by itself after one short run. Files: `lanes/P3.patch` (diff against package_T10b), `bench/decisions_P3.jsonl`, `scripts_P3/sync_p3.sh`.
