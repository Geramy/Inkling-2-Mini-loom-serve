# dev/: development side of the Inkling 2 Mini Loom package

The repository root is the published package (`model.json`, `program/`, `weights.loom`, `tokenizer.json`, `llm_config.json`, `info.json`, `README.md`, `REPORT.md`, `PORTING.md`). It is complete: `loom-fetch` and `loom-serve` read it as it is. Nothing in `dev/` is needed to run the model.

`dev/` holds the source and the evidence that the root tree was built from.

| Path | Content |
|---|---|
| `dev/package/` | The working source of the Loom programs (kernels, dispatch, control, libraries, `card.json`) and the kernel tests in `package/test/`. The root `program/` is a pruned copy of it. `tokenizer.json` is not repeated here: it is the root file. |
| `dev/scripts/` | The tools: `publish_build.py` (builds the root tree from `dev/package/`), `publish_lock.sh`, `make_llm_config.py`, `make_card.sh` (writes `info.json`, `README.md`, `REPORT.md` with the loom-skills card tools), `scrub_check.sh`, `publish_check.sh` (clean-room fetch and serve), `run_gates.sh`, `ktest_adapter.py` (kernel tests), `perf_session.sh` and `perf_run_all.sh`, `roofline_inkling.py`, the hook generators `gen_hooks.py` and `gen_kernel_decls.py` (they write `package/dispatch/hooks_*.loom` and `kernel_decls.loom`; do not edit those files by hand). |
| `dev/kit/` | The chat kit: `chat_block.json` (the chat dialect block of `llm_config.json`) and the golden chat cases. |
| `dev/docs/` | Design notes and results: architecture, kernels per area, dispatch, tune route, chat policy, the optimization plan and lane reports (`OPT_*.md`), `PERF_BASELINE.md`, `ROOFLINE.md`, `CONTEXT_TABLE.md`, `SKILLS_LOG.md` (a dated work log; it names the agents that did each step, as history). |
| `dev/lanes/` | The optimization patches: wave 1 (`D1`, `D4`, `D5`, `P1`, `wave1_merged`), rejected or unpublished lanes (`P3`, `QV`), and wave 2 (`D2`, `D3`, `D6`, `W64`, `M30` = the merged result on hrx-loom 0.3.0, `CHATFIX` = prefix cache and tool-call policy, `CMDPROG` = grouped attention projection grid). They are already applied in `dev/package/`. |
| `dev/oracle/` | The CPU llama.cpp dump tool and the three oracle prompts. The tensor dumps are not included (several hundred MB). |
| `dev/bench/` | Small evidence: per GPU (`radeon-r9700`, `strix-halo`) the evidence JSON that the card is built from (`ev_gates.json`, `ev_perf.json`, `ev_roofline_*.json`), the gate results, the perf logs, the kernel test logs, the context table and the optimization decision logs. `bench/M30/` holds the wave 2 evidence (kernel tests, greedy checks, gates, the merged A/B log). Files with the suffix `_wave1` are the earlier evidence; per-target kernel test logs under `kernel_tests/` are from wave 1 (the wave 2 summary is `kernel_tests.log`). Large raw profiles are not included. |

## How the root relates to `dev/`

1. `dev/package/` is edited (kernels by hand; `hooks_*.loom` and `kernel_decls.loom` come from the generators).
2. `scripts/publish_build.py` copies the files that `model.json` names into the root layout and writes `model.json`, `NOTICE`, `LICENSE`, `.gitattributes` and `llm_config.json`. `publish_lock.sh` writes `model.json.lock`.
3. `scripts/make_card.sh` writes `info.json`, `README.md` and `REPORT.md` from the evidence in `dev/bench/`. These three files are tool output and are never edited by hand. `PORTING.md` is hand-written.
4. `scripts/scrub_check.sh` checks that the root tree names no host, user or local path.

The scripts were written for one GPU box (two GPUs: Radeon AI PRO R9700, gfx1201, and Radeon 8060S, gfx1151) and use box paths such as `~/inkling/...`, `~/loom` or `~/loom030` (SDK venv; wave 2 uses hrx-loom 0.3.0, set `LOOM_ROOT`) and a GPU lock file. Adapt the paths before you run them. Host names, user names and home paths in this tree are placeholders (`user@gpu-box`, `/home/user`).

License: MIT for the programs (see the root `LICENSE` and `NOTICE`). The weights are not in this repository.
