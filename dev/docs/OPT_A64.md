# Lane A64: row alignment (row64) experiment, phase 1 only

Result: negative, stop rule hit at phase 1. Padding rows to 32, 64, 128 or 256 B does not raise the streaming bandwidth on the R9700 (gfx1201) or the 8060S (gfx1151) above the natural GGUF strides. Phase 2 (layout design, repack, kernels) was not started. No package change: `package_A64` was not created, `package/` and `package_M30/` are untouched. The patch (`lanes/A64.patch`) holds only the probe scripts.

## Method

`scripts_A64/a64_probe.py` generates Loom streaming kernels shaped like our GEMV kernels: 128 threads, wave32, groups of G lanes read one row each, loads of 4, 8 or 16 B per lane, loads fully unrolled in the source, a lane sum only (no compute), one store per wave. One dispatch reads 1 GiB to 2 GiB of random i8 data (the buffer is 2 GiB, far above the last-level cache), so every dispatch is cold. `loom-benchmark --measure=dispatch_complete --batch-size=1 --iterations=25`: 60 timed samples per point, median reported. A config is a list of segments (stride S, contiguous bytes C per row), a start offset O inside the buffer, a load width W and a group size G. Rows: Q8_0 K=2304 is 72 x 34 = 2448 B; NVFP4 gate_up row is 1296 B (36 blocks of 36 B); NVFP4 down row is 648 B; the "split" layout is a scale plane (144 B per row) and a quant plane (2304 B per row), each padded as named. Results: `bench/A64/results_<group>_<gpu>.jsonl` (all timing fields). Rebuild: `a64_probe.py build <layout|offset|contig|gran>`, then `flock -x ~/hrx-pr1/gpu.lock python3 a64_probe.py run <group> <gfx1201|gfx1151> 25`; `a64_sum.py` prints a table.

The lock wait did not dominate (the owner's perf session ran between my groups). Each config costs about 20 s because the benchmark tool generates 2 GiB of random data per process.

## 1. Alignment of row starts: natural against padded (decision table)

"useful" counts only the real row bytes (what a kernel needs). "span" counts the bytes of the padded layout (what the memory system moved). Padding overhead is (S - C) / C. Medians, GB/s, 16 B loads unless named.

R9700 (gfx1201), useful / span:

| layout | natural | pad 32 | pad 64 | pad 128 | pad 256 |
|---|---|---|---|---|---|
| Q8_0 row 2448 B, stride | 2448 | 2464 | 2496 | 2560 | 2560 |
| overhead | 0 | 0.7 % | 2.0 % | 4.6 % | 4.6 % |
| GB/s | 598.1 / 598.1 | 593.4 / 597.3 | 583.5 / 594.9 | 573.5 / 599.7 | 573.6 / 599.8 |
| NVFP4 gate_up 1296 B, stride | 1296 | 1312 | 1344 | 1408 | 1536 |
| overhead | 0 | 1.2 % | 3.7 % | 8.6 % | 18.5 % |
| GB/s | 600.1 / 600.1 | 590.8 / 598.1 | 579.0 / 600.4 | 553.1 / 600.9 | 507.5 / 601.4 |
| NVFP4 down 648 B (8 B loads), stride | 648 | 672 | 704 | 768 | 768 |
| overhead | 0 | 3.7 % | 8.6 % | 18.5 % | 18.5 % |
| GB/s | 595.8 / 595.8 | 575.0 / 596.3 | 550.8 / 598.3 | 505.3 / 598.8 | 503.8 / 597.0 |
| Q8_0 split planes (scale 144 B + quant 2304 B), GB/s | 599.1 / 599.1 | 595.5 / 599.4 | 589.6 / 601.2 | 572.1 / 598.3 | 575.4 / 601.7 |

8060S (gfx1151), useful / span:

| layout | natural | pad 32 | pad 64 | pad 128 | pad 256 |
|---|---|---|---|---|---|
| Q8_0 2448 B | 225.6 / 225.6 | 222.8 / 224.2 | 221.2 / 225.5 | 215.4 / 225.3 | 215.4 / 225.3 |
| NVFP4 1296 B (16 B loads) | 226.0 / 226.0 | 220.0 / 222.7 | 213.1 / 221.0 | 207.2 / 225.1 | 194.4 / 230.4 |
| NVFP4 648 B (8 B loads) | 234.5 / 234.5 | 225.0 / 233.4 | 214.9 / 233.4 | 195.8 / 232.0 | 195.8 / 232.0 |
| Q8_0 split planes | 225.4 / 225.4 | 223.0 / 224.4 | 220.4 / 224.7 | 219.9 / 230.0 | 219.1 / 229.2 |

Reading:
- The span bandwidth (bytes moved per second) is the same for every alignment: R9700 594 to 602 GB/s, 8060S 220 to 234 GB/s (the 8060S scatters by about 4 percent between configs, the R9700 by about 1 percent). The useful bandwidth falls by the padding overhead. Natural strides stream as fast as aligned ones.
- 4, 8 and 16 B load widths make no difference (R9700 within 0.5 percent).
- Reading the padding too with 16 B loads ("readpad" points) gives the same GB/s but moves more bytes per row, so it is slower per row.
- Start offset sweep (stream of 2048 B rows, offsets 0, 4, 8, 16, 32, 48, 64, 96, 128 B, 4 B and 16 B loads): R9700 593.7 to 601.4 GB/s, 8060S 223.6 to 224.9 GB/s, no trend.
- Best aligned gain over natural: below 1 percent on the R9700 (+0.2 to +0.4 percent in span), and inside the scatter on the 8060S (-0.5 to +1.9 percent). In useful bytes every padded layout loses 0.7 to 18.5 percent. The stop rule (aligned gain under 2 percent on both GPUs) holds.

## 2. Minimum fetch (owner addition): sparse row sweeps

Rows of S bytes (512, 768, 1024), only the first C bytes of each row read. If the memory system fetched only what is asked, the time would fall with C. It does not.

R9700, S = 512 (4 M rows, 2 GiB span), 16 B loads:

| C (B) | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---|---|---|---|---|---|
| time (us) | 1860 | 1866 | 1890 | 1922 | 1917 | 3672 |
| useful GB/s | 34.9 | 70.8 | 140.9 | 278.3 | 558.9 | 584.2 |

The time is flat from 16 to 256 B and doubles at 512 B (two units per row). S = 768: 1314 to 1360 us for C = 16 to 256, 2530 us at 512. The cost is one fetch per row touched: 256 B x 4 M rows = 1.07 GB in 1.9 ms = about 560 GB/s, the full streaming rate. Minimum fetch on gfx1201: 256 B.

The 8060S (S = 512, 16 B loads) gives 4654 us (C = 16) to 4739 us (C = 128), 8920 us at C = 256 (offset 0) and 8979 us at C = 512: the same flat curve up to one unit.

Position inside the unit (S = 1024, G = C/16, offsets 0, 32, 64, 96, 128, 192, 224 B; time in us):

| C | R9700 in one 256 B unit | R9700 straddling a 256 B boundary | 8060S in one unit | 8060S straddling |
|---|---|---|---|---|
| 64 | 1878 (offsets 0 to 192) | 3517 (offset 224) | 4654 to 4692 (offsets 0 to 64, 128 to 224) | 8978 (offset 96) |
| 128 | 1892 to 1919 (offsets 0 to 128) | 3563 to 3565 (offsets 192, 224) | 4760 to 4798 (offsets 0, 128, 192, 224) | 8994 to 9389 (offsets 32, 64, 96) |
| 256 | 1950 (offset 0) | 3586 to 3606 (offsets 32 to 224) | 4702 (offset 128) | 9063 to 9079 (offsets 0 to 96, 192, 224) |

R9700: the fetch unit is 256 B, aligned to the buffer start. A request group that touches two units costs twice, and 64 to 256 B inside one unit cost the same.

8060S: a touched unit also costs a full 256 B transfer (C = 64, 128 and an aligned 256 B all take about 4.7 ms, 228 GB/s of moved bytes), so the minimum efficient fetch is 256 B in this test, not 64 B. Crossing a 128 B boundary at an odd multiple (128, 384) doubles the time, crossing 256 does not. So the 256 B units sit 128 B away from the start of this buffer (allocation placement or address swizzle: not known to us). A layout aligned to 256 B inside the buffer is therefore not guaranteed to be aligned in the memory system on the 8060S; the buffer base would have to be known. The owner's 64 B for gfx1151 is not seen at this level: if the DRAM burst is 64 B, the fetch unit above it is larger.

Why this does not help our kernels: the units matter only when rows are sparse. Our weight rows are read whole and contiguous: every 256 B unit is used completely by the same row or by its neighbour, so no fetch is wasted except the two edge units of one contiguous region (an expert matrix is 0.8 MB to 1.5 MB: about 0.03 percent). Section 1 shows it: natural layout equals aligned layout.

## Decision

Stop. Row alignment (32 to 256 B) gives under 1 percent on both GPUs and costs 0.7 to 18.5 percent more bytes with padding. A load-time repack would need a second copy of the weights: the R9700 has about 9 GB free (34 GB, 24.9 GB used at 131072 context) and the NVFP4 experts alone are about 17 GB, so a full copy does not fit; repacking in place at load needs a change in loom-serve or HRX (not allowed). Cost if built with 64 B padding: Q8_0 +2.0 percent, NVFP4 gate_up +3.7 percent, down +8.6 percent of the row bytes. No phase 2 numbers, no kernel change. Promote: no.

Where the 256 B unit could still matter (not tested, not in our weight streams): sparse or strided per-row access, for example small per-row arrays or routing records at a stride. If the graph has such an access, grouping it into whole 256 B units is the useful form of the idea.

## Files

`scripts_A64/a64_probe.py`, `scripts_A64/a64_sum.py`, `bench/A64/results_*.jsonl`, `bench/decisions_A64.jsonl`, `lanes/A64.patch`. Scratch on the box (mine): `~/inkling/optA64` (probe sources and merged modules, no weights); delete with `rm -r ~/inkling/optA64/probe ~/inkling/optA64/scripts`.
