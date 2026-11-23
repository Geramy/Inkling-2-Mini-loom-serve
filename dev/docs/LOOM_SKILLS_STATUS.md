# Final status of the loom-skills adaptation (wave 2, 2026-11-22)

## Published

- Tree `publish/unsloth-Inkling-2-Mini-loom/` (and the GitHub repository Geramy/Inkling-2-Mini-loom-serve, root = the tree, `dev/` = sources and evidence). Built by `scripts/publish_build.py`, `publish_lock.sh` and `make_card.sh` with hrx-loom 0.3.0 tools. `info.json`, `README.md` and `REPORT.md` are tool output, never edited by hand.
- Programs: wave 1 (T10b) plus D2, D3, D6 and W64 (merged by M30). Q4 presets: context 131072, prefill 2048. Q8 presets: context 32768, prefill 512 (not measured).
- SDK pin in the card: hrx-loom 0.3.0 (the hrx-system commit is not recorded by the 0.3.0 install file: "unknown").
- License MIT for the programs, owner decision; weights are not in the tree.

## Measured (merged build, one session per GPU, medians of 3 requests, tok/s)

| | decode short | decode 8k | decode 32k | prefill 2k | 4 streams | decode short vs ceiling |
|---|---|---|---|---|---|---|
| R9700 | 94.86 | 84.77 | 75.38 | 3704 | 213.2 | 0.61 of 154.6 |
| 8060S | 42.80 | 37.81 | 32.44 | 1221 | 88.06 | 0.75 of 57.3 |

Wave 1 published: 90.2 and 39.7 decode short. Details and the wave 1 tables: `PERF_BASELINE.md`.

## Gates

- Kernel tests: 33 of 33 on both GPUs.
- Greedy text of the merged build equals wave 1 (A, B, C, 8k and 32k documents, both GPUs).
- Dev gates (M30, `bench/M30/`): consistency, agent, golden and compare pass on both GPUs. Sampling fails by design (the skills' check expects a greedy default; this server samples at 1.0). Context admission 130,991 tokens of 131072 completes on both GPUs.
- Card tools: schema 0 errors, `render_card.py --check` OK, scrub PASS, `loom-fetch` of all four presets OK, clean-room `publish_check.sh` (result in `bench/prompt_c/publish_check.log`).

## Not verified

- Capture-route layer check on hrx-loom 0.3.0: the `sequence` deployment does not register ("invocation argument count mismatch") because the 0.3.0 library moved to `text_epoch_v2`. `dispatch/sequence.loom` and the vendored epoch need a port. The text route (all serving and all gates) is not affected. The last layer check ran on 0.2.0 with wave 1.
- No 0.2.0 A arm of the wave 1 build in the same session as the 0.3.0 measurement: the R9700 decode of the wave 1 build on 0.3.0 is 3 percent below the published 0.2.0 number, cause not separated.
- Wave 2 gains were measured per lane at kernel level (0.2.0) and as one merged session per GPU (no ABBA, no per-lane end-to-end run on 0.3.0).
- The card holds only decode short, decode 32k and 4 streams for wave 2. Prefill rows 1k to 32k, decode 2k and cold start were not re-measured with `perf.py`; ceilings and efficiency ratios in the card rows come from `roofline.json` of wave 1 (ceilings unchanged).
- The `offset` to `index` rename is NOT applied: the 0.3.0 library still uses `offset` and a mixed program fails to register.
- Q8 presets (both GPUs) were not run for wave 2. The full-level gate audit was not run.
- Peak memory was not re-measured after wave 2; the budget stays at the wave 1 value (24.9 GB at 131072 context).
- Owner to confirm the license before any further publication.

## Known issue of the generated card on hrx-loom 0.3.0

- The generated R9700 quick start in `README.md` ends with `--parameter_mode=preload` (rule of the skills' card tool, made for 0.2.0). `loom-serve` 0.3.0 rejects it: registration fails with `unknown --parameter_mode='preload' (expected 'file' or 'direct')`. The 8060S quick start has no such flag and works. `PORTING.md` section 9 says to leave the flag out; `README.md` is tool output and was not edited (the skills tree is read-only). Without the flag the R9700 serve works (`publish_check.sh` serves `chat-r9700` on the R9700 without it; `QS_DROP_PARAMETER_MODE=1` makes the quick start step drop it too).
