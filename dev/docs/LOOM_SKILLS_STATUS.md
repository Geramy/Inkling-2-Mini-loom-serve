# Final status of the loom-skills adaptation (wave 2, chat fixes, projection grid; 2026-11-24)

## Published

- Tree `publish/unsloth-Inkling-2-Mini-loom/` (and the GitHub repository Geramy/Inkling-2-Mini-loom-serve, root = the tree, `dev/` = sources and evidence). Built by `scripts/publish_build.py`, `publish_lock.sh` and `make_card.sh` with hrx-loom 0.3.0 tools. `info.json`, `README.md` and `REPORT.md` are tool output, never edited by hand.
- Programs: wave 1 (T10b) plus D2, D3, D6 and W64 (merged by M30) plus the grouped projection grid of lane CMDPROG (`docs/OPT_CMDPROG.md`) plus the chat fixes of lane CHATFIX (prefix cache 16 in all presets, tolerant tool-call policy, fork-state fix; `docs/OPT_CHATFIX.md`). Q4 presets: context 131072, prefill 2048. Q8 presets: context 32768, prefill 512 (not measured).
- SDK pin in the card: hrx-loom 0.3.0 (hrx-system commit "unknown": the 0.3.0 install file does not record it).
- License MIT for the programs, owner decision; weights are not in the tree.

## Measured (3 sessions per GPU, medians, tok/s; wave 1 in brackets)

| | decode short | 2k | 8k | 32k | prefill 1k | 2k | 8k | 16k | 32k | 4 streams |
|---|---|---|---|---|---|---|---|---|---|---|
| R9700 | 97.8 (90.2) | 90.0 (83.2) | 86.7 (80.3) | 76.3 (71.3) | 2670 (2576) | 3132 (2996) | 3239 (3092) | 2932 (2801) | 2342 (2242) | 226.8 (218.9) |
| 8060S | 42.6 (39.7) | 39.4 (36.7) | 37.6 (34.9) | 32.3 (29.7) | 937 (894) | 1095 (1033) | 1123 (1051) | 1040 (973) | 867 (812) | 95.7 (89.0) |

Decode short is 0.63 (R9700, ceiling 154.3) and 0.74 (8060S, ceiling 57.2) of the roofline. The card shows every row except decode 8k (an extra row, not a card metric) and has no cold-start or TTFT rows (the card schema has none; they are in `PERF_BASELINE.md`: ready in 2.1 to 2.6 s, first request 3.2 s / 0.8 s, TTFT at 9k tokens 2.8 s / 8.1 s).

The R9700 row is the final build (3 sessions, one GPU at a time). The 8060S row was measured before the chat fixes and the grid change; its code path is unchanged (tune value 4) and one session of the final build gave decode 42.73 / 39.3 / 32.27 tok/s. Against the previous commit the R9700 decode is +2.0 to +2.8 percent; prefill reads about 2 percent lower in the median (session 1 equals the old table, sessions 2 and 3 are lower: session drift, not isolated). Wave 2 per-lane numbers and the first M30 single-session figures are in `PERF_BASELINE.md`.

## Gates and checks

- Kernel tests: 33 of 33 on both GPUs (final build). Greedy text (A, B, C, 8k, 32k) equals the previous published build on both GPUs.
- Dev gates on the final build, both GPUs: consistency, agent, golden and compare pass; sampling fails by design. Context admission 130,991 tokens completes (175 s R9700, 619 s 8060S).
- Card tools: schema 0 errors, scrub PASS, `loom-fetch` of all four presets OK, `publish_check.sh` PASS on both GPUs (`bench/prompt_c/publish_check.log`).
- Published route (chat-r9700, R9700), final tree: `chat_tool_probe.py` 55 of 55; `chat_numerics.py` identical 24 of 24, repeats (cache hits) equal their first run 48 of 48; peak VRAM 27.6 GB (`bench/integ4/`).

## Not verified

- Capture-route layer check on hrx-loom 0.3.0: the `sequence` deployment does not register ("invocation argument count mismatch"; the 0.3.0 library moved to `text_epoch_v2`). The text route is not affected.
- No 0.2.0 A arm in the same session; the wave 1 build on 0.3.0 read about 3 percent lower on the R9700 decode earlier.
- Wave 2 per-lane gains were measured at kernel level (0.2.0); on 0.3.0 only the merged build was measured.
- The 8060S perf table was not re-measured on the final build (one sanity session only). The chat fixes were checked on the R9700 (Q4) on the published tree; the 8060S and Q8 presets on the candidate only (`docs/OPT_CHATFIX.md`).
- The offset to index rename is NOT applied: the 0.3.0 library still uses `offset`.
- Memory hints: the Q4 presets now carry transient 10.8 GB (prefix cache), Q8 6.4 GB.
- Full-level gate audit not run.
- loom-serve gaps (issue texts in `OPT_CHATFIX.md`): `function.arguments` as an object and a missing tool-call `type` are rejected by the runtime; a tool message cannot get its call name; the prefix cache keeps only chunk frontiers; no rewind of generated tokens.
- Owner to confirm the license before further publication.

## Known issue of the generated card on hrx-loom 0.3.0

- The generated R9700 quick start in `README.md` ends with `--parameter_mode=preload` (rule of the skills' card tool, made for 0.2.0). `loom-serve` 0.3.0 rejects it (`expected 'file' or 'direct'`). The card tool offers no input to change it (only a fleet `serve_args` entry, outside our build route, and the skills tree is read-only), so `README.md` stays as generated and `PORTING.md` section 9 says to leave the flag out. `publish_check.sh` with `QS_DROP_PARAMETER_MODE=1` verifies the quick start without it.
