# Final status of the loom-skills adaptation (wave 2 and chat fixes, 2026-11-24)

## Published

- Tree `publish/unsloth-Inkling-2-Mini-loom/` (and the GitHub repository Geramy/Inkling-2-Mini-loom-serve, root = the tree, `dev/` = sources and evidence). Built by `scripts/publish_build.py`, `publish_lock.sh` and `make_card.sh` with hrx-loom 0.3.0 tools. `info.json`, `README.md` and `REPORT.md` are tool output, never edited by hand.
- Programs: wave 1 (T10b) plus D2, D3, D6 and W64 (merged by M30) plus the chat fixes of lane CHATFIX (prefix cache 16 in all presets, tolerant tool-call policy, fork-state fix; `docs/OPT_CHATFIX.md`). Q4 presets: context 131072, prefill 2048. Q8 presets: context 32768, prefill 512 (not measured).
- SDK pin in the card: hrx-loom 0.3.0 (hrx-system commit "unknown": the 0.3.0 install file does not record it).
- License MIT for the programs, owner decision; weights are not in the tree.

## Measured (3 sessions per GPU, medians, tok/s; wave 1 in brackets)

| | decode short | 2k | 8k | 32k | prefill 1k | 2k | 8k | 16k | 32k | 4 streams |
|---|---|---|---|---|---|---|---|---|---|---|
| R9700 | 95.1 (90.2) | 87.7 (83.2) | 84.5 (80.3) | 74.8 (71.3) | 2725 (2576) | 3197 (2996) | 3297 (3092) | 2972 (2801) | 2339 (2242) | 225.3 (218.9) |
| 8060S | 42.6 (39.7) | 39.4 (36.7) | 37.6 (34.9) | 32.3 (29.7) | 937 (894) | 1095 (1033) | 1123 (1051) | 1040 (973) | 867 (812) | 95.7 (89.0) |

Decode short is 0.62 (R9700, ceiling 154.3) and 0.74 (8060S, ceiling 57.2) of the roofline. The card shows every row except decode 8k (an extra row, not a card metric) and has no cold-start or TTFT rows (the card schema has none; they are in `PERF_BASELINE.md`: ready in 2.1 to 2.6 s, first request 3.2 s / 0.8 s, TTFT at 9k tokens 2.8 s / 8.1 s).

These numbers differ from the M30 single session (decode within 0.6 percent; 4 streams 213.2 and 88.06 there, measured with another tool). They were measured on the M30 build; the CHATFIX perf guard showed decode unchanged and prefill within the session drift, and the full table was not re-measured on the published build.

## Gates and checks

- Kernel tests: 33 of 33 on both GPUs (M30 build). Greedy text equals wave 1.
- Dev gates (M30 build and CHATFIX build): consistency, agent, golden and compare pass; sampling fails by design. The agent gate also passes without `name` in tool messages.
- Card tools: schema 0 errors, scrub PASS, `loom-fetch` of all four presets OK, `publish_check.sh` PASS on both GPUs (`bench/prompt_c/publish_check.log`).
- Published route (chat-r9700, R9700): `chat_tool_probe.py` 55 of 55; `chat_numerics.py` prefix-cache hit equals cold on 24 of 24 prompts and repeats equal their first run (48 of 48); peak VRAM 27.6 GB (`bench/chat_published/`).

## Not verified

- Capture-route layer check on hrx-loom 0.3.0: the `sequence` deployment does not register ("invocation argument count mismatch"; the 0.3.0 library moved to `text_epoch_v2`). The text route is not affected.
- No 0.2.0 A arm in the same session; the wave 1 build on 0.3.0 read about 3 percent lower on the R9700 decode earlier.
- Wave 2 per-lane gains were measured at kernel level (0.2.0); on 0.3.0 only the merged build was measured.
- The perf table was measured before the chat fixes (see above). Prefix cache and tool policy were checked on the R9700 (Q4) on the published tree; the 8060S and Q8 presets were checked on the candidate only (`docs/OPT_CHATFIX.md`).
- The offset to index rename is NOT applied: the 0.3.0 library still uses `offset`.
- Memory hints: the Q4 presets now carry transient 10.8 GB (prefix cache), Q8 6.4 GB.
- Full-level gate audit not run.
- loom-serve gaps (issue texts in `OPT_CHATFIX.md`): `function.arguments` as an object and a missing tool-call `type` are rejected by the runtime; a tool message cannot get its call name; the prefix cache keeps only chunk frontiers; no rewind of generated tokens.
- Owner to confirm the license before further publication.

## Known issue of the generated card on hrx-loom 0.3.0

- The generated R9700 quick start in `README.md` ends with `--parameter_mode=preload` (rule of the skills' card tool, made for 0.2.0). `loom-serve` 0.3.0 rejects it (`expected 'file' or 'direct'`). The card tool offers no input to change it (only a fleet `serve_args` entry, outside our build route, and the skills tree is read-only), so `README.md` stays as generated and `PORTING.md` section 9 says to leave the flag out. `publish_check.sh` with `QS_DROP_PARAMETER_MODE=1` verifies the quick start without it.
