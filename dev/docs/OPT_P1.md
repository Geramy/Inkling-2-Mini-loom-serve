# Lane P1: larger-n experts pass and prefill_capacity (optP1, 2026-11-21, bead llama-cpp-tg3.24)

Result: KEPT. Prefill with `prefill_capacity` 1024 and 2048 works on both GPUs, greedy output is token-identical to the base build, decode rows and the request floor do not change. Gains are larger than the plan model (+20 / +34 percent on the R9700) because the experts matrix kernels now use 64-route tiles.

## What changed (patch `lanes/P1.patch`, base `package_base_T10`, scripts hunk for `scripts/gen_hooks.py`)

| File | Change |
|---|---|
| `kernel/experts/routes.loom` | The workgroup rank store (4096 words, the `n <= 512` limit) is gone. The count pass runs twice: the second pass resets the counters and re-issues the atomics, and the returned rank places the route. Any `n` is valid (tested 2048; LDS use 1 KB). Record stride is `max(tile, 32) + 2` words (34 for tile 8 and 32, 66 for tile 64). |
| `kernel/experts/gate_up_wmma.loom`, `down_wmma.loom` | Route tile 32 to 64: two column blocks of the 2x2 provider per wave (eight accumulators), the weight rows are decoded once per 64 routes. Record stride 66. `down` stores through a non-inline function (`inkling_experts_down_store`): the 64 unrolled store regions in one kernel were rejected by the target (`masked_region_exits_by_cfg`). |
| `dispatch/hooks_experts.loom`, `scripts/gen_hooks.py` | Matrix path calls `routes` with tile 64; tasks buffer 264 B per record. |
| `test/experts/{test_prefill,test_oracle,test_gate_up,bench_experts}.py` | Tile 64 and the 66 word stride in the fixtures. |

The arithmetic per route is unchanged (same K order and f32 accumulation per output), so experts results are equal to the tile-32 build: `test_prefill` numbers are identical to the old ones (dout 6.9e-7, y 9.9e-7 on gfx1201).

Compile report (loom-compile, `build.sh`): gfx1201 gate_up and down 144 VGPR, gfx1151 168 VGPR, 0 spills, 27,648 B LDS (occupancy limited by LDS as before; tile 32 used 96 VGPR). Compile time of the package grows with the number of shape buckets: stage compile sum (`/v1/status`, summed over the workers) 41.4 s (prefill 512), 56.1 s (1024), 75.1 s (2048) on the R9700 and 74.6, 97.9, 135.3 s on the 8060S.

No change was needed in prepare.loom, config.loom, the attention kernels, the dense GEMMs or any other dispatch file: they already scale with `inkling.tokens`. The 256 span limit is untouched. Nothing hangs at 1024 or 2048 (the old hang came from the routes rank store only).

## Numerics and chunking evidence

- `test_prefill.py` (n = 37, 308, 512, 1024, 2048; portable and matrix paths; NVFP4, Q8_0, BF16) PASS on gfx1201 and gfx1151. `test_gate_up.py` 12 of 12 and `test_oracle.py` 140 PASS, 0 FAIL on both GPUs (expert output y vs oracle `conv_in` 1.0e-2 NVFP4, as before).
- Greedy 64 tokens through the chat route, prompts A, B, C of the oracle and two long prompts (L3k 2434 tokens: 5 chunks at 512, 3 at 1024, 2 at 2048; L9k 9071 tokens): reply (reasoning and content) identical to the base build for every prompt on both GPUs, for the same package at prefill_capacity 512 (new kernels), 1024 and 2048. This is the chunk-independence check (1 chunk vs 3 vs 512+rest; prefill then decode). Files: `~/inkling/optP1/g/*.json` (box), compare with `scripts/route_check.py --diff`.
- The layer-NMSE capture route (capture_all) was not re-run: the experts numbers above and the token-identical replies cover the changed kernels. The integrator's full gates (golden, compare, consistency) still run once per GPU.

## End-to-end prefill (ABBA, fresh server per arm, 3 or 5 requests per metric, median; mean of the two arms of each side)

Base = `programs_sha256 51c67d50` tree (inkV `mat-*`), prefill_capacity 512. Metrics as in `ab_session.py` (max_tokens 1, total time, nonce per request): pf1k 1020 tokens, pf2k 2040, pf8k 9082, pf32k 37006. Tok/s.

| GPU | Row | Base | capacity 1024 | | capacity 2048 | |
|---|---|---|---|---|---|---|
| R9700 | prefill 1k | 2054 | 2504 | +21.9 % | 2476 | +21.0 % |
| R9700 | prefill 2k | 2226 | 2998 | +34.7 % | 3447 | +55.3 % |
| R9700 | prefill 8k | 1940 | 2577 | +32.8 % | 2944 | +52.0 % |
| R9700 | prefill 32k | 1406 | not run | | 1990 | +41.5 % |
| 8060S | prefill 1k | 797.5 | 876.9 | +10.0 % | 836.6 | +4.7 % |
| 8060S | prefill 2k | 861.1 | 1038.7 | +20.6 % | 1131.9 | +31.4 % |
| 8060S | prefill 8k | 772.0 | 926.3 | +20.0 % | 994.2 | +28.8 % |
| 8060S | prefill 32k | 586.6 | not run | | 727.2 | +24.0 % |

Arm-to-arm spread inside one side is below 1 percent (rows in `bench/decisions_P1.jsonl`, raw lines in `bench/ab_log_P1.jsonl`). Targets of plan section 4: R9700 2k 3300 reached at capacity 2048 (3447); 8060S 2k 1250 not reached (1132).

The 8060S 1k row is lower at capacity 2048 than at 1024 (837 vs 877): a 1020 token prompt runs in a larger shape bucket (padding: experts and all GEMMs run on the bucket capacity). On the R9700 the same effect is 1 percent.

## Guard rows (decode, request floor, short prompts, memory)

Decode, ABBA of base against capacity 1024 (3 requests per metric, mean of the two arms per side), tok/s:

| GPU | short | 8k | 32k | 4 streams |
|---|---|---|---|---|
| R9700 base | 86.06 | 76.94 | 67.95 | 200.3 |
| R9700 capacity 1024 | 85.93 (-0.15 %) | 76.70 (-0.3 %) | 67.35 (-0.9 %) | 200.6 (+0.2 %) |
| 8060S base | 38.59 | 33.53 | 27.84 | 81.9 |
| 8060S capacity 1024 | 38.42 (-0.4 %) | 33.22 (-0.9 %) | 27.47 (-1.3 %) | 80.7 (-1.5 %) |

R9700 differences are inside the arm-to-arm noise (the two A arms differ by 0.7 % at 32k). On the 8060S the loss is small but consistent in both B arms; part of it (about 0.2 %) is the second counting pass of the routes kernel on every decode visit, the rest is not explained (unified memory placement of the larger workspace is a guess). Capacity 2048 decode was not measured (decode uses the capacity-1 and capacity-4 stages; only the shared workspace differs).

Request floor (`Say hi`, 1 token, ms): R9700 52.9 base, 53.0 at 1024, 52.9 at 2048; 8060S 125.0 base, 124.8 at 1024, 124.0 at 2048. No effect.

Short prompts, R9700, base against capacity 2048 (ABBA, 5 requests per metric): prefill 150 tokens 1288 to 1270 (-1.4 %), 300 tokens 1714 to 1774 (+3.5 %), 600 tokens 1969 to 2405 (+22.1 %), 1500 tokens 2211 to 3094 (+39.9 %). Short prompts lose at most 1.4 % (bucket padding); nothing else loses.

Memory (R9700 preset, 4 rows, 131072 context, `memv` after the first requests): 25.09 GB base, 25.52 GB at capacity 2048 (+0.43 GB). 8060S gtt: 25.15 GB, 25.58 GB. The R9700 has 34.2 GB: 8.7 GB free at capacity 2048. Pool workspace slab (8060S status): 1.007 GB base, 1.141 GB at 1024, 1.409 GB at 2048; storage committed 24.98, 25.12, 25.40 GB.

## For the integrator (do not edit `scripts/publish_build.py` here)

- Preset change: `execution.prefill_capacity` 512 to 2048 for `chat-r9700` and `chat` (Q4). Capacity 1024 is the conservative choice for the 8060S 1k row (+10 % instead of +4.7 %), at the cost of 2k, 8k and 32k gains (+20.6 / +20.0 vs +31.4 / +28.8). Chat presets for Q8: the same code, not measured; Q8 transient memory estimate 3.6 GB plus 0.43 GB.
- `memory.transient_bytes`: 7,550,000,000 stays valid (measured non-weight use at 131072 is 7.9 GB at capacity 2048 on the R9700 (25.52 - 17.62); raise to 8,000,000,000 to keep the estimate honest). `workspace_gib` of llm_config 2.14 to about 2.6.
- `docs/CONTEXT_TABLE.md`: replace "prefill_capacity stays 512 (experts routing kernel limit)": the limit is lifted; the build assert that presets equal `card.json` except context and memory fields must also allow `prefill_capacity`, or `package/card.json` must carry the new value.
- Stage compile (cold kernel cache) grows from 75 s to 135 s of summed worker time at 2048; the kernel cache is keyed by content, so only the first start after a program change pays it.
- Conflicts with other lanes: this lane edits `routes.loom`, `gate_up_wmma.loom`, `down_wmma.loom`, `hooks_experts.loom` and two lines of `gen_hooks.py` (the `o264` record size and `chain(64, ...)` in the experts hook). The decode path of the router and routes kernels (lane D3, not in this wave) must keep the stride rule of `routes.loom`.
- Test hygiene: `test/experts/test_prefill.py` and `test_oracle.py` write about 34 GB of fixtures under the work directory (`~/inkling/inkD/work` in the sources; I redirected it to my lane directory); delete it after a run (the box disk had 9 GB free during the lane).

## Not done

- Layer-NMSE capture check (see above).
- Tile 128 and an LDS double buffer (more weight reuse at n = 2048): not tried; the stop rule was far exceeded and the integrator time box applies. Experts at n = 2048 are probably still far from the 90 percent memory floor; plan lane P2 continues from here.
- Kernel-level benchmark rows (`bench_experts.py`) per n: the runs failed twice (my own deadlock: the script takes its own GPU lock, I wrapped it in an outer `flock -x`; fixed with `runbench2.sh`, then not repeated for lack of lock time).

## Rebuild and reproduce

```
bash scripts/sync_p1.sh                                   # package_P1 to ~/inkling/optP1/package (laptop)
bash scripts/p1_mkvar.sh r9700 p1c2048 ~/inkling/optP1/package 2048    # variant model dir (hard links, no weight copy)
bash scripts/p1_ab.sh r9700 ~/inkling/inkV/mat-r9700 ~/inkling/optP1/var-r9700-p1c2048 floor,pf1k,pf2k,pf8k,pf32k,memv 3 1
bash scripts/p1_greedy.sh r9700 <model dir> out.json      # A, B, C and two long prompts; compare two files with route_check.py --diff
```

`scripts/ab_session.py` has new metrics (`pf1k`, `pf150`, `pf300`, `pf600`, `pf1500`, `pf32k`, `floor`, `memv`, `memg`). Tests: `. <venv>; cd package/test/experts; python test_prefill.py gfx1201 512 1024 2048; python test_oracle.py gfx1201; python test_gate_up.py gfx1201` (shared GPU lock inside).
