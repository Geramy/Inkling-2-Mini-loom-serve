# Plan: adapt the Inkling 2 Mini loom-serve package to the official loom-skills workflow

Bead llama-cpp-tg3.12 (agent inkL). Status: reading and planning only. The package was not restructured, nothing was committed or uploaded.

Sources read: `loom-fun/loom-skills` at 618f89e (box `~/inkling/loom-skills`, read-only; below "skills/..." means that tree), and the published example `loom-fun/unsloth-Qwen3.5-4B-loom` (box `~/inkling/example-qwen35`). Statements marked (unverified) were not run.

Things run for this plan (all in my own dirs, nothing under ~/.claude or ~/.codex, skills not installed):
- `~/loom/.venv` (hrx-loom 0.2.0 from PyPI, `loom-install.json`), `~/loom/.tools` and `~/loom/library` (catalog v0.2.0) were created with `skills/loom-setup/scripts/install_loom.sh ~/loom`.
- `~/inkling/inkL/cfg.json` from `inspect_gguf.py`; `chat_probe.py`; `kb.py route`; `roofline.py`; HF repo listing (anonymous, `HF_HOME` in inkL, no token).
- `~/inkling/inkL/probe/`: scratch copy of the package in the published layout (see 1.9) plus `loom-fetch --update-lock` against local symlinks. This worked (details below). `~/inkling/inkL/snap/` holds the symlinks.

---------------------------------------------------------------------------------------------------------------

## 0. One-screen summary

1. The skills treat a published Loom LLM as a repo of: `model.json` (card), `model.json.lock`, `llm_config.json`, `weights.loom`, `program/*.loom`, `tokenizer.json`, plus three generated files `README.md`, `REPORT.md`, `info.json`. Weights are never in the repo: `model.json` points at the unmodified GGUF in an HF repo by pinned revision, `loom-fetch` pins sha256 in the lock.
2. Our package is a hand-written new architecture. The skills have an exact route for that: `catalog-ports` (`kb topic catalog-ports`, `starter/catalog.py`): a kit holds `catalog/{catalog.json, model.json, program/**}` and `port.sh catalog` turns it into a package with `llm_config.json`, so qualify/gates/perf/card all work unchanged. The `qwen4exp-example` kit is the model for it (nested `program/**`, hand-written, `catalog.json` with memory, sdk, source).
3. Verified today: a card in the published layout with origins `@checkpoint/Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file` at revision `ea0c79b2ecd73669e774c8519b79bfabfddbc027` is accepted by `loom-fetch`, and `--update-lock --all-variants` wrote a correct `model.json.lock` in 10 s. Lock sha256 equals the HF LFS sha256 (Q4 `e64276...`, Q8 `2da517...`). So restructuring is cheap.
4. The tools run on our GGUF only in part:
   - `inspect_gguf.py` runs, warns (arch unknown, rope guess, nvfp4 has no starter kernels, 3 q-heads per kv-head) and writes `chat: null`.
   - `kb.py route` refuses `inkling2_loop` ("unknown arch"). It needs `--arch <known>` as a stand-in (I used `qwen3moe --facets moe,sliding-window`); there is no inkling node, no `publish` topic exists.
   - `chat_probe.py` crashes in `derive()`: `ProbeError: reasoning instructions: unexpected head '<|message_system|><|content_text|>Thinking effort level: 0.1<|end_message|><|message_user|><|content_text|>'`. So the chat block must be hand-written, with a kit `chat_probe.py` (`simulate`, `render_calls`, `raw_from_chat`) pointed at by `LOOM_KIT`. This is blocker B1 for every gate that renders prompts (golden, compare, agent_golden, sampling).
   - `roofline.py` runs but is wrong for us: it counts 2.315 GB/token and 264.8 tok/s (R9700) because it does not know the 48 visits / looped layers / NVFP4 experts. Our own floor (ARCHITECTURE.md section 12) is 4011 MB/token = 154.6 tok/s (R9700), 57.3 tok/s (8060S). We need our own `roofline.json` (do not edit the skills).
5. Owner rules force documented deviations: no ROCm and no Vulkan llama.cpp, so the "llama.cpp" side is our CPU oracle (`~/inkling/inkB`, mainline + PR 32704). Correctness gates (golden/compare/agent/sampling) can run against it through `LLAMA_URL`. Performance margins vs llama.cpp ROCm/Vulkan cannot be provided; only Loom rows, ceilings and a CPU llama.cpp row labelled as CPU, which is not a GPU baseline.
6. Missing today: `llm_config.json` equivalent, chat kit, gate evidence in the skills' format, `ctx_admit` at the preset context (native context is 1,048,576; ours is served at 32,768), roofline table, `info.json`/README/REPORT, scrubbed repo tree, perf in the skills' method (3 sessions, median, active wait), HAL profile tables.
7. Work plan: 12 tasks (section 6). Critical path: T1 layout -> T2 llm_config/work dir -> T3 chat kit -> T4 CPU reference server -> T5 gates dev -> T6 perf -> T7 roofline -> T8 card. Optimize loop (T10) only after the owner agrees; it is not needed to publish a first local folder.

---------------------------------------------------------------------------------------------------------------

## A. What the skills require of a published Loom LLM package

### A.1 Layout (example: `example-qwen35`; `skills/loom-kb/topic/benchmark.md:101-186`; `skills/loom-kb/topic/porting.md` "Package card")

```
model.json            card (schema_version 1), see A.2
model.json.lock       schema_version 2, scope "assets": pinned origins + size + sha256 of every non-program file
llm_config.json       build facts + chat dialect, see A.4 (rides as program data of prepare.loom so loom-fetch delivers it)
weights.loom          tiny: func prepare_weight(%key: buffer) -> (i64) returning 0 (the weights are read by name from the GGUF)
tokenizer.json        HF tokenizers JSON (origin "tokenizer.json" in the lock)
program/*.loom        prepare.loom, control.loom (chat glue), storage.loom, service.loom, llm.loom (libs) ...
README.md REPORT.md info.json   generated by the card scripts (A.5-A.6), never hand-written
.gitattributes
```
Nested program trees are normal for hand-written packages: the qwen4exp catalog model uses `"programs.source": {"directory": "program", "origin": "./", "inputs": [...], "libraries": [...]}` with `program/kernel/*.loom`, `program/dispatch/*.loom`, `program/shared/...` (`reference/qwen4exp-example/catalog/model.json`).
Not in the repo: weights, `model_target.json`, `model_d*.json`, `parameter/`, tests (`publish_package.py:SKIP`, `catalog.py:KEEP_OUT = catalog.json, BUILD.bazel, test, __pycache__, model.json.lock`).

### A.2 `model.json` (example and `starter/catalog.py card_for`)
- `schema_version: 1`, optional `$schema` (hrx-system model-card.v1), `model {id, name, description?, origin?, license?}`.
- `services.chat {kind:"text", version:1, components?}`.
- `programs.<set> {directory?, origin?, inputs:[path | {path, data:[...]}], libraries:[...]}`. `prepare.loom` carries `data: ["weights.loom", "llm_config.json"]`.
- `variants.<v>.services.chat {programs, config{...}, parameters{<name>{files:[{path:"parameter/X.gguf", format:"gguf", origin:"@checkpoint/<repo path>"}]}}, tokenizers{text:"tokenizer.json"}}`. Parameter names used by the skills: `model`, `draft`, `vision`.
- `presets.<p> {service, variant, programs, execution{rows, context_capacity, prefill_capacity, max_tokens, draft_depth?, pool_capacity?}, options.sampling{temperature, top_p, top_k}, memory?}`. A catalog package has no `pool_capacity` (loom-serve rejects the field; `catalog.py` comment).
- Preset names per GPU: first preset is tuned on Strix Halo; `<first>-r9700` is the R9700 preset; `-16gb`, `-gfx1150`, `-gfx1152` exist for other classes (`device/rdna4/radeon-r9700/README.md`: "Preset: `<first>-r9700`").
- `origins.<name> {kind:"hf", repository, revision}`: one entry per source repo (checkpoint, drafter, ...). The revision is a commit sha.
- Example weight file origins have the repo-relative path after `@checkpoint/`. Ours, verified: `@checkpoint/Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file` (the repo's files carry the intentional `.gguf_file` extension, see Mainline_Support/README.md; the local copy is renamed `.gguf`; the local `path` is `parameter/Inkling-2-Mini-UD-Q4_K_XL.gguf`). The revision we downloaded is in `~/models/inkling2-mini/.cache/huggingface/download/Mainline_Support/*.metadata`: `ea0c79b2ecd73669e774c8519b79bfabfddbc027`. The repo HEAD today is `ae9327c5fbc7333116ba32eee78bb5d2bbefae4f` (updated 2026-11-21 08:35). Pin the revision we validated (`ea0c79b`), re-check later.
- Repo license: `apache-2.0` (front matter of unsloth/Inkling-2-Mini-GGUF; base model `thinkingmachines/Inkling-2-Mini`). Not yet seen in `card.json`; `model.license` field is available.

### A.3 `model.json.lock`
Written by `loom-fetch ./model.json --update-lock --all-variants --no-input --origin checkpoint=<local snapshot dir>` (`publish_package.py prepare` does exactly that; `--lock-scope assets` default leaves programs editable). Content: `origins.<n>.{requested, resolved.revision}`, `files[] {origin, size, sha256}` including `tokenizer.json`. `publish_package.py prepare` reads the revision from an HF cache path (`models--org--repo/snapshots/REV/file`); our GGUFs are in a `--local-dir` tree, so use `loom-fetch` directly with the symlink snapshot (done in the probe) or add the `.metadata` revision by hand.

### A.4 `llm_config.json` (what it is for in our hand-written case)
It is the contract between the package and the skills' scripts: `bench_common.Package` reads it (`context_length`, `chat`, `model_id`, `vision`, `serve`, `loom_thinking_default`, `draft`), `gates.sh` reads `vision.mmproj`, `source.kind`; `card_lib` reads `generator` and `sdk`; `roofline.py --cfg` reads `sliding_layers`, `kv_layers`, `head_dim_swa`; `find_reference`/fleet `add_model.py` read it (drops `tensors, generator, draft, resolved, weights`). In the catalog route it is written by `catalog.py build()` as: `cfg.json` (inspect output minus `tensors`) + `model_id`, `draft`, `catalog{variant, preset, draft_kinds, memory}`, `sdk`, `generator`, `resolved` (an "exempt" block: "catalog package: hand-written programs, no generator runs"), and `chat` (the dialect). `generator` = loom-skills commit + SDK (`kit_source.generator_stamp`); `sdk` = `{hrx_loom, source, wheel_sha256, hrx_system_sha}` from `loom-install.json`.
For us: not an input to our programs (our `control` is hand-written Loom); it is the machine-readable description for tools and the card. Fields from our `cfg.json` that matter: `arch inkling2_loop`, `layers 24`, `hidden 2304`, `heads 24/8`, `head_dim 128`, `vocab 201024`, `context_length 1048576`, `sliding_window 1024`, `sliding_layers` (20 of 24), `eos_id 200006`, `extra_eos_ids [199999]`, `moe{128 experts, 8 used, ffn 1152, gating 1, scale 8.0, probs_bias}`; `inspect_gguf.py` leaves `chat` null and reports `formats ['bf16','f32','q8_0']` (it does not list nvfp4).

### A.5 `info.json` (`skills/loom-benchmark/scripts/card/info_schema.json`, `topic/benchmark.md:101-186`)
Schema `loom-card/3`; required top keys `schema, repo, card, versions, sources, devices`; also `report: "generated"`. Written by `card/make_info.py build DIR --repo ORG/NAME --memory-bytes N --output OUT --report generated --evidence ev_*.json -o DIR/info.json`. Holds only what no package file records:
- `card.features`: `api, audio, concurrency{rows}, context{native, tokens}, image_input, max_output_tokens, reasoning_effort, sampling, speculative_decoding, thinking{mode, default}, tool_calling` each `{status yes|n/a|..., value}`; `card.license`, `card.memory_bytes`.
- `devices.<gpu>` for `strix-halo` and `radeon-r9700` (and others): `gates{<gate>{status, level, date, programs_sha256, source, detail}}`, `performance{baseline{name, version, args}, baselines{vulkan{...}}, sessions...}`, `roofline{checkpoint, device{name, bandwidth, peak_flops}, drafting, metrics{decode, decode_2k, decode_32k, prefill_1k..32k, streams4: ceiling, efficiency}}`, `quick_start{preset, device, args}`, `serve_args`, `served_context`, `memory_bytes`.
- `sources`: one per measurement run (`kind devlab|factory|machine|xsj|local`, device, hrx_loom stamp, run id/url). `versions.card_generator`.
Evidence comes from `make_info.py perf|gates|roofline` (`loom-card-evidence/2`), made on the machine that measured. `perf` refuses `--source-kind local`; `machine` (a named machine) is accepted (`card_lib.PERF_SOURCE_KINDS = devlab, factory, machine, xsj`, `make_info.py:1010-1021`). So our box is a `machine` source.

### A.6 README.md and REPORT.md (rendered by `card/render_card.py DIR`, never by hand)
README sections in this order: front matter (base_model, license, tags), title, quick start per GPU (`pipx install "hrx-loom[hf]==0.2.0"`, `loom-fetch huggingface:ORG/REPO:PRESET --materialize=copy --output=NAME`, `loom-serve --model=NAME '--device=amdgpu://?completion_spin_ns=4294967295' --pool_backing=fixed --parameter_mode=preload`), "Send a request", "Performance vs llama.cpp" per GPU (table: llama.cpp, Loom, margin vs best llama.cpp, roofline, Loom % of roofline, llama.cpp Vulkan; metrics decode short/2k/32k, prefill 1k/2k/8k/16k/32k, 4 streams, drafted accept), methodology bullets, "Features", "Correctness" (gate table with footnotes), "Files". REPORT: Setup, Method, per GPU: Baseline line, Session N (device, SDK, programs sha256, runs and values), baseline sessions, Roofline (device constants, tool sha), Gate evidence (per run). "No prose of your own on a card" (`benchmark.md:106`); a value without evidence renders "not measured / no evidence".
Our deviations (no ROCm/Vulkan) must therefore be expressed through data, not prose: `baseline.differs` ("not supported" cell with reason, `loom-benchmark/SKILL.md:50-52`), plus a separate `docs/` file outside the repo.

### A.7 Tokenizer
HF `tokenizer.json` of the same model (GGUF repos usually lack it); `chat.nonspecial_tokens` lists structure tokens that must be non-special so loom-serve does not strip them from decoded text (`topic/porting.md` "Tokenizer gotchas"). Ours already did this (INTEGRATION_LOG M6: 12 structure tokens non-special, built from the GGUF tokenizer by `scripts/gen_tokenizer.py` + `chat_tokenizer.py`). Open: the lock entry for `tokenizer.json` is our own file (12,244,587 B, sha256 `0c3b63...`) because it is generated, not copied from `thinkingmachines/Inkling-2-Mini`.

### A.8 How a package is built
- Starter path (`port.sh ORG/REPO`): `inspect_gguf.py` -> `cfg.json` -> `build.py` generates programs (`gen_llm.py`, `gen_moe.py`, `gen_flash2.py`, ...) -> `serve.sh` -> kernel tests (`make_kernel_tests.py` + `run_kernel_tests.sh`) -> smoke. Covers dense, Qwen3.5 hybrid, softmax-router MoE with shared experts.
- New architecture (`topic/new-arch.md`): trace llama.cpp (`llama_trace.py`), diff vs starter, copy the starter as a kit (`apply_example.sh`), add cfg flag + generator op + kernel test per feature, `chat_probe.py` kit for unmodelable chat formats, node-by-node numpy check, resume with `FROM=build KIT=W/kit port.sh`. Example kits: gemma4, lfm2moe, flint, gpt-oss (harmony channels + reasoning effort: the closest to Inkling's format), glimmer (channel format), qwen4exp.
- Hand-written package (`topic/catalog-ports.md`, `starter/catalog.py`): `CATALOG=DIR port.sh catalog W PORT auto MODEL_ID`. We already have the hand-written programs, so this is our route. `catalog.json` keys: `source{repo, revision, file, extra}, variant, preset, context, loom_thinking_default, prefill_capacity, serve{args, committed_bytes}, memory{resident_weights_bytes, kv_bytes_per_position, recurrent_bytes_per_row, workspace_gib, over_cap, note}, sdk, tokenizer`. Kernel tests are the package's own `loom_test` targets, run by `loom-benchmark/scripts/kernel_tests.py` (no Bazel); a catalog kit without them gets "no evidence" on `kernel_tests`.
- Reference start (`topic/reference.md`): only a category A package counts; none exists for `inkling2_loop`, so no reference.

### A.9 Probe of the layout (done)
`~/inkling/inkL/probe.py` copies `~/inkling/inkK/package` into `probe/{model.json, tokenizer.json, program/**}` (all `.loom` below `program/`, paths prefixed, `capture` program/variants/preset and `test`, `devices` dropped), adds `origins.checkpoint` and per-file `origin`. Result: `loom-fetch ./model.json --variant ud-q4-k-xl --preset interactive-q4 --dry-run` parses the card (fails only for the missing lock), `--update-lock --all-variants --origin checkpoint=snap` writes the lock in 10 s. Not yet done: link/compile of the moved tree and a served request from a `loom-fetch --materialize=copy` of it (task T1).

---------------------------------------------------------------------------------------------------------------

## B. Required gates, how they run, thresholds

Order for a new port: kernel tests -> `chat_probe --verify` -> smoke -> `qualify.sh` (gates LEVEL=quick) -> optimize (LEVEL=dev after each kept change) -> `finalize.sh` (LEVEL=full + audit + report) (`loom/SKILL.md:80-137`). A port is done when: kernel tests pass, golden shows only near-tie divergences, `agent_golden.py` passes, `ctx_admit.py` shows the full context capacity admitted (`loom-port-llm/SKILL.md:120-122`).

| Gate | Tool | What / pass criterion |
|---|---|---|
| kernel tests | `starter/run_kernel_tests.sh` (starter) or `loom-benchmark/scripts/kernel_tests.py PKG TESTDIR` (catalog `loom_test` targets, BUILD.bazel evaluated without Bazel) | each kernel vs numpy dequant/reference; statuses pass/fail/skipped; "skipped" is never "pass"; evidence `kernel_tests pass (N/N)` |
| chat glue | `chat_probe.py MODEL.gguf --verify cfg.json` | every conversation OK: chat, multi-turn, tools, system+tools, agent loop, parallel calls, thinking on/off; run before gates when the glue changes |
| golden + compare | `golden.py PKG golden.json --llama URL --long 1000,2000,16000,32000` twice (2nd recording as `--alt`), `compare.py PKG golden.json --alt golden2.json --url LOOM` | token ids sent to llama.cpp `/completion` greedy with `n_probs 5`; 20 short prompts x 64 tokens + one prompt per long document. PASS: every first divergence is a near-tie (gap logp(golden)-logp(loom) <= 1.0 and token in golden top-5, or Loom took llama's 2nd choice where p1 < 0.5), extended by llama.cpp's own noise (`llama_views`: batch vs micro-batch of 8, capped 0.5). Healthy: max gap 0.1-0.6 and a quarter full matches. FAIL: large gap or token outside every view's top list (`topic/benchmark.md:8-99`). Before declaring FAIL on one prompt, re-record it. |
| consistency | `consistency.py URL --model ID`, `determinism.py` | concurrent == alone, run-to-run identical; difference passes only if near-tie by the compare rule |
| agent | `agent_golden.py PKG URL --llama URL` (thinking on and off if toggle) | per assistant turn: `render_ok` (`/v1/chat/completions/input_tokens` equals `chat_probe.simulate` token count), `reuse_ok` (`cached_tokens` >= 80 % of previous prompt+completion), `calls_ok`, golden per turn near-tie |
| sampling | `sampling_check.py PKG URL --llama URL` | T=0 and top_k 1 equal greedy; seed reproduces, different seeds vary; 200 first-token draws vs llama.cpp probabilities within sampling noise (TV distance; 0.011 on Qwen3-0.6B) |
| spec | `spec_check.py` | only packages that draft; n/a for us |
| vision | `vision_golden.py` | only with mmproj; n/a for us |
| ctx admit (check, no gate row) | `ctx_admit.py PKG URL [--tokens N] [--complete]` | one request of ~`context_capacity` tokens is admitted (not 4xx/503) then cancelled; failure fails run and audit. Use `--complete` once for a request that really finishes (`loom-optimize` step 7) |
| level matrix | `gates.sh` (`LEVEL=dev|fast|quick|full`) | dev: consistency + agent + golden/compare on 12 short + 1k/2k docs + sampling light; quick/full add the 20 short + 1k-32k, 200-draw sampling, ctx admit (`loom-benchmark/SKILL.md:78-83`). A gate that passed on the same package hash is not re-run (`gates.sh` hash of model.json + program/*.loom, `gates_runs.jsonl`). `NO_LLAMA=1`: reference-free checks only, correctness "unverified". |

Performance method (`loom-benchmark/SKILL.md:224-288`, `reference/methodology.md`): `perf.py URL label --model ID --log perf_log.jsonl [--long]`: decode 300-token essay, decode_2k/32k, prefill 1k/2k, 8k/16k/32k documents, median of 5; `c4_steady.py URL 4` aggregate (rows >= 4); warm up after every restart; nonce per prompt; a decode counts only if >= half max_tokens; sessions: 3 sessions median (fast models, decode > ~100 tok/s: 6 sessions with `--repeat-rows 3`); noise +-2-3 % decode, +-5-10 % long prefill, margin within +-3 % is a tie; Loom served with the active wait `--device=amdgpu://?completion_spin_ns=4294967295` (`serve.sh` default); `--pool_backing=fixed`; `--parameter_mode=preload` on stock hosts (memlock, ROCm/hrx-system#3332); idle machine. Local numbers are for development; card numbers come from `devlab|factory|machine|xsj` sources.

Roofline (`loom-roofline/SKILL.md`, `roofline.py`): per metric ceiling = decode: bandwidth / bytes read per token (weights + active experts + KV); prefill: peak FLOP/s / flops per token; 4 streams read weights once. Device presets from `loom-kb/device/*/device.json`: strix-halo 228 GB/s measured streaming (spec 256), 59 TFLOP/s spec; radeon-r9700 613 GB/s, 191.4 TFLOP/s spec (R9700S measured 214). TARGET comes from efficiencies Loom ports already reached (`EFF`, `STRETCH`), measured against the ceiling; `qualify.sh` writes `bench/roofline.json` and `bench/targets.jsonl`; `perf.py` prints CONTINUE / REVIEW / BUDGET / DONE; `audit.py` must print `AUDIT PASS`; Loom must not exceed ~0.97 of a ceiling.

Profiles (`topic/optimize-loop.md:43-66`): `loom-serve --device_profiling_mode=dispatch-events,executable-metadata --device_profiling_flush_interval_ms=5 --print_device_statistics`; rows `dispatch NAME count=N total=T avg=A` printed at graceful shutdown (SIGTERM); flush interval 5 ms to avoid `dropped_records`; one session per scenario (pf2k, pf8k, drafted decode); `profile_requests.py` + `profile_stats.py` group by `profile_families.json`; check `busy` against request wall. Hardware counters are UNIMPLEMENTED on gfx11/gfx12 (`profile_data=counters` returns UNIMPLEMENTED).

Optimize loop (`loom-optimize/SKILL.md`): targets from leader and roofline per row -> HAL profile -> bucket budgets (judge the bucket, not the kernel) -> design (fuse by default, price first, several designs, old path stays selectable, generated not hand-pasted) -> kernel level (`loom-opt` over variants, ISA read, `ab_kernel.py`) -> end-to-end `ab_perf.py` ABBA interleaved against a base sha -> keep only if it wins its row and regresses none by more than 2-3 % -> `LEVEL=dev gates.sh` once on the winner -> record (`perf.py` labels, `--reverted`, `bench/decisions.jsonl`). Lane ends only with a stop reason (at the roof >= 90 % of bandwidth; ceiling below 1 %; diminishing returns after 2+ designs; correctness; costs another row more).

Card-level gates the skills also apply (we mostly do not need): `logits`, `decision`, `vision`, `spec`, `agent_think_off`.

---------------------------------------------------------------------------------------------------------------

## C. Deviations we must document (owner rules)

Owner rules: no ROCm builds, no Vulkan measurements, no changes to HRX/Loom (SDK, runtime, compiler), no uploads. The skills assume ROCm and Vulkan llama.cpp baselines (`loom/SKILL.md:154`, `loom-benchmark/SKILL.md:29-52`: "Reference: llama.cpp ROCm (not Vulkan, not CPU)").

| Skill requirement | What we can do | Deviation to document |
|---|---|---|
| Golden reference = llama.cpp ROCm, GPU (`loom/SKILL.md:214`: external reference, never own numpy) | CPU llama.cpp from `~/inkling/inkB` (master 418fe97 + PR 32704 cherry-pick, f32 KV; docs/ORACLE.md) is an external llama.cpp run of the same GGUF. `golden.py/compare.py/agent_golden.py/sampling_check.py` take `--llama URL`; `gates.sh` takes `LLAMA_URL` (skips `llamacpp_server.sh`, which refuses a build without ROCm0). | Reference numerics are CPU f32-KV, not ROCm f16-KV: near-tie noise differs (in our favour: more exact). The near-tie extension `llama_views` restarts llama-server with other flags through `LLAMA_START` (not set with `LLAMA_URL`): (unverified) it may be unavailable; if so only the plain 1.0-nat rule applies and a FAIL is judged by hand with `-ub` variants (`topic/benchmark.md:54-62`). |
| `pick_llamacpp.sh`, `qualify.sh` step 1-2 (fastest ROCm build, baseline perf) | Skip: do not call `qualify.sh` whole; run its steps by hand or with `NO_LLAMACPP` semantics (`finalize.sh` exports `NO_LLAMA=1` when no build is found). | `llamacpp_pick.json` absent: `perf.py` and `audit.py` want it ("Order is enforced", `--no-campaign`; audit Baseline check will FAIL/WARN). Write a pick file naming the CPU build and say so; or run audit with that check waived in our own report. (unverified which.) |
| llama.cpp ROCm baseline column and margins; Vulkan baseline | Not possible. | Card/README shows no llama.cpp GPU numbers. A CPU llama.cpp row (`llama-bench`/`perf.py --kind llama` on the CPU server, exact args, `-ngl 0`) can be shown only as "llama.cpp CPU (mainline + PR 32704)". The Vulkan column is a "not supported" cell with the reason (`baseline.differs`). Margin vs best llama.cpp "not measured". |
| Perf source `devlab`/`xsj` | Use `--source-kind machine` (named machine, our box). | Card sources say named-machine run; the R9700 is a Thunderbolt 5 eGPU (not the DevLab R9700S), the 8060S shares the box; GPU shared with other agents (flock). |
| Roofline from `roofline.py` | Wrong for looped arch (2.315 GB vs our floor 4.011 GB per token; 264.8 vs 154.6 tok/s). Skills are read-only and `roofline.py` is a protected core file (`kb/SKILL.md` rule 4). | Provide our own `roofline.json` in the tool's schema, computed from ARCHITECTURE.md section 12 and the bandwidth/FLOP constants of the device presets (or measured on this box, docs/DEVICES.md); say that `roofline.py` does not model the loop. |
| `kb.py route --config` | Fails ("unknown arch"). | Route with a stand-in arch (`qwen3moe`, facets `moe,sliding-window`); no skill change. |
| Starter kernels / generated programs | Not used: hand-written, our own tune table and motifs (owner rule). | `llm_config.json` `resolved` block = "hand-written programs, no generator runs"; `generator` stamp records loom-skills sha. |
| Native context (`ctx_admit` at `context_capacity`, `loom-port-llm/SKILL.md:20`) | Native is 1,048,576. KV is 32 KiB per position per pool entry (INTEGRATION_LOG); 1 M positions would not fit the R9700 (weights 17.6 GB + 6.7 GB transients already ~24 GB of 34 GB). | Serve a reduced `context_capacity` per GPU (the skills allow it: "a package that does not fit at its native context is served at a reduced `served_context`", `radeon-r9700/README.md`; example preset `chat-gfx1150` 63,488). Needs a measured per-GPU context table (T5). Also open: `prefill_capacity <= 512` limit of our router kernel (RUNNING.md). |
| Speculative decoding (default in skills) | None. The repo has `dspark-Inkling-2-Mini-*.gguf` drafter, not supported by us. | `spec` gate n/a, `speculative_decoding` feature "no". |
| Vision (`mmproj-*.gguf` in repo) | Text only (`chat_image_limits` zero). | `image_input` n/a (feature row) and `vision` gate n/a. |
| Thinking switch (`enable_thinking`) | Inkling uses an effort level (`Thinking effort level: 0.1..0.99`, names none/minimal/low/medium/high/xhigh/max); CHAT_POLICY section 2. | Feature row `reasoning_effort` yes; `thinking` toggle semantics differ; the `agent_think_off` gate maps to effort `none` (still may emit thinking, model card). `loom_thinking_default` needs review. |
| Tool-call grammar | Template defines it; model output grammar unverified (CHAT_POLICY section 7). | `agent_golden` will be the first real test; may expose a format mismatch. |
| Hardware | The R9700 is an eGPU over TB5; both GPUs shared with other agents (flock). | Record in `sources`; timed runs under `flock -x ~/hrx-pr1/gpu.lock`, correctness under `-s`. |
| `loom-fetch` revision pin | `ea0c79b` (validated) not HEAD `ae9327c`. | None needed beyond the pin; recheck repo drift before any future publish. |

No change to HRX / Loom / skills / SDK is required by any of this. The two places where a skill script may need adaptation (the pick file, `llama_views` with `LLAMA_URL`) are worked around in our own scripts/env, not edited in `~/inkling/loom-skills`.

---------------------------------------------------------------------------------------------------------------

## D. Mandate / scope rules and this project's scope

Rules (`skills/loom-kb/SKILL.md:43-58`, applied by `loom-port-llm`, `loom-optimize`, `loom-fleet`): (1) declare the scope first (a `loom-scope` block in the PR body: kind, scopes, kernels); (2) opt in by scope: a new optimization applies to the measured scope (family, facet, device) only; (3) run `fleet/scope_lint.py` before opening a PR; (4) stop on out-of-mandate needs: a change to a protected core file (generator core, `dual_target.py`, `build.py`, `kb.py`, `choices.py`, methodology, registries, card schema, workflows) is an `architecture` PR: report and stop, do not work around it. Also hard rules in `loom/SKILL.md:160-215`: memory (weights + KV within 60 % of RAM), /tmp is RAM, run long jobs in their own scope, serve as the card serves (`--pool_backing=fixed`), no unapproved binaries, golden references external, stop servers by PID.

What an agent may do under them: read the skills, run their tools from the box against our package in our own work dir, write our own scripts and docs, produce card files into the local `publish/` tree. May not: install skills into agent configs, edit the skills tree, change card schema/methodology/roofline.py/kb nodes, open PRs, upload to HF, run DevLab/fleet jobs, loosen a gate.

Scope for this project: local folder `C:\Development\inkling-loom-serve\` only (laptop + `~/inkling/<NAME>` on the box). No PRs, no HF upload (`publish_package.py upload` is never run), no fleet jobs, no nightly-list entry, no scope_lint (nothing to lint: no PR). Our implicit `loom-scope`: `kind: card` for model `unsloth-Inkling-2-Mini-loom`, scopes `arch inkling2_loop`, devices `strix-halo` (gfx1151) and `radeon-r9700` (gfx1201). If the owner later wants a PR to loom-skills (a kb node for `inkling2_loop`, a loop-aware roofline), that is an `architecture`/`tooling` PR and needs the owner.

---------------------------------------------------------------------------------------------------------------

## E. Gap analysis of the current package

Current state (INTEGRATION_LOG, RUNNING, DISPATCH, `package/card.json`): hand-written Loom package for `inkling2_loop` runs on both GPUs with Q4_K_XL and Q8_K_XL; M5/M6 oracle checks done (l_out NMSE 1.4e-4 to 2.2e-3, head NMSE 7.0e-4 vs oracle at f32 KV, top-5 5/5); chat route with template-exact rendering (44 count cases pass on both GPUs, scripted outputs 28, options 34, sampler 35). First (untuned) numbers: decode ctx16 R9700 86.6 tok/s Q4 (floor 154.6), 8060S 38.9 tok/s (floor 57.3); prefill about 2200 tok/s R9700, 900 tok/s 8060S at 512-token chunks.

| Requirement | Status |
|---|---|
| Card with services/variants/presets/programs | Satisfied in content: `package/card.json` has the same grammar. Differences: file is `card.json` (and a different old `package/model.json` definition exists), not `model.json` (root card); all .loom outside `program/`; extra `capture` program/service/variant/preset (debug route); weight `path` is a bare file name, no `origin`, no top-level `origins`; presets named `interactive-q4/q8` (skills: `<first>`, `<first>-r9700`); `memory` blocks in presets are ours; no `model.license`/`origin`. |
| `model.json.lock` | Missing. Verified recipe works (A.9). |
| `llm_config.json` | Missing. Source facts exist (`cfg.json` from `inspect_gguf.py`); chat block, `model_id`, `catalog`, `sdk`, `generator`, `resolved` to be added (T2). Needs the chat block derivation (B1). |
| `weights.loom`, `tokenizer.json`, `program/prepare.loom` | `weights.loom`, `prepare.loom`, `tokenizer.json` exist; `prepare.loom` must list `llm_config.json` as data (catalog.py does it). |
| Chat/tool/reasoning glue | Satisfied functionally (CHAT_POLICY, 44 template counts, Jinja reference); not expressed as `chat` dialect + kit `chat_probe.py` (`simulate`, `raw_from_chat`, `render_calls`) the gates import. `chat_probe.py` cannot derive the template (crash above). |
| README.md / REPORT.md / info.json | Missing. `docs/model_card_README.md.txt` is hand-written prose (a different thing: skills forbid hand-written card text). |
| Kernel tests vs numpy | We have `package/test/{attn,chat,dense,experts,mix}` (py scripts, oracle compares). Not in `loom_test` target form, so `kernel_tests.py` cannot run them; evidence row `kernel_tests` has none (T5 adapter or custom evidence JSON). |
| Golden vs llama.cpp (log-prob near-tie) | We compared against the CPU oracle at tensor level (`compare_to_oracle.py`, `head_check.py`, teacher-forced greedy) but not with the skills' golden/compare (20 prompts + 1k-32k docs, `--alt` rerecord, near-tie rule). Missing. |
| agent_golden, sampling_check, consistency | Missing (sampler tests exist locally: 35 PASS; consistency across 4 rows untested in the skills' sense). |
| ctx_admit | Missing; preset context is 32,768 vs native 1,048,576; the per-GPU served context table is missing. |
| qualify/finalize evidence | Missing: `gates.json`, `perf_log.jsonl` in the skills' method (3 sessions, median, active wait?), `targets.jsonl`, `audit.json`. Our numbers (INTEGRATION_LOG) use our own scripts (`bench_chat.py`: min/median of 4-5 streamed requests; locked, warmed). |
| Roofline tables | We have floors (ARCHITECTURE.md section 12, `cost_model_output.txt`, `floor_bench.py`): need to be re-expressed in `roofline.json` form (T7). |
| Profiles with HAL dispatch timestamps | `scripts/jsplit.sh` measures stage times by hook counts (INTEGRATION_LOG); no `--device_profiling_mode` family table yet (T10). |
| Gate on `/v1/chat/completions/input_tokens`, `cached_tokens`, `/metrics` | Unknown (unverified): `test_template.py` counts input tokens, reuse of retained state is in `complete_text`. Check the endpoint paths in loom-serve (T3). |
| Hygiene for publishing | Package contains `test/`, `devices/`, capture route, 14 MB of tests; scripts and docs name host `gpu-box`, user `user`, `~/inkling/inkK` etc. Publish tree must be scrubbed (T1, T9). |
| License / notices | Programs partly vendored (`lib/vendor/*` from hrx-loom-library, MIT, `lib/VENDORED.md`) and PR1 motifs (`lib/pr1`): need LICENSE/NOTICE in the publish tree (the skills repo itself is MIT, Third-party notices list adapted library code). Decide the license of our programs (owner). |

Already satisfying the skills: the loom-fetch card grammar, weights read in place from the GGUF by tensor name, weights outside the repo, fixed pool, an external CPU llama.cpp oracle with exact version, tensor-level numerics at the drift floor, template-exact chat with a verified Jinja reference, tokenizer with non-special structure tokens, per-arch tune data (matches the skills' opt-in-by-scope idea).

---------------------------------------------------------------------------------------------------------------

## F. Open questions for the owner

1. Is publishing local-only (a tree under `publish/`, `loom-fetch` from a local dir) enough, i.e. no HF upload ever in this task? (This plan assumes yes.)
2. The llama.cpp side: confirm the CPU oracle is the only reference and that a CPU llama.cpp row on the card (labelled CPU) is acceptable, or show no llama.cpp column at all.
3. Context: which `context_capacity` per GPU is wanted (the memory table will give the maximum; 32,768 now; native 1,048,576 is not reachable on R9700)?
4. Which quant is the headline package: Q4_K_XL only (as `chat`), or both Q4_K_XL and Q8_K_XL as two variants/presets? (Q4 is the skills' default tier; Q8_K_XL is the near-lossless reference.)
5. Are we allowed to add a stand-in kb node or a loop-aware roofline to a local copy of the skills (not the read-only checkout, no PR)? Otherwise we keep our own scripts under `scripts/`.
6. Program license and attribution for vendored/library-derived files.
7. Optimize loop: go or stop for now (memory says tuning stopped 2026-11-19 for the PR1 llama.cpp work; this is the loom-serve package, separate)? Time budget per `loom-roofline` section 0: 30 min for an existing port unless the owner sets one.
8. Pin policy for the checkpoint revision: stay on `ea0c79b` while repo HEAD moved to `ae9327c`?

---------------------------------------------------------------------------------------------------------------

## G. Risks

- B1: chat kit is mandatory for most gates; our policy/prompt is custom (effort header state, tool-call grammar unverified). Biggest schedule risk.
- CPU reference speed: 26 B params, Q4/Q8 on 32 threads (shared box, 119 GiB free). Long goldens (16k/32k) prefill on CPU can take hours; use `--long 1000,2000` (+ 8000) and `LEVEL=dev` equivalent; record `llama_cache` once and reuse it for every later Loom build (cache is keyed on GGUF + build + request, `topic/benchmark.md`). Unknown real speed: read `oracle/logs/` bench.
- `llama_views` with `LLAMA_URL` (above), `llama_fill.py` deferral machinery assumes `LLAMA_START`.
- Skill tools may assume things our server lacks (`/metrics` acceptance counters, `/v1/status` fields used by `ctx_admit`, `/v1/chat/completions/input_tokens`): spike early (T3/T5).
- The `prefill_capacity <= 512` router limit and 256-span limit: fine for gates, bad for prefill speed vs `prefill` rows; ctx_admit `--complete` at 32k takes ~15 s at 2200 tok/s; at larger contexts it grows.
- Moving all `.loom` under `program/` changes relative include/library paths and `config` keys; must re-link everything and rerun the capture-route oracle checks once to prove no numeric change (hash of programs changes the `pkg_hash`).
- GPU sharing: timed runs under `flock -x`; every gate/perf run is long enough that other agents wait.
- Q8_K_XL on the R9700 uses 22.9 GB + 6.7 GB transient: one server at a time; the 8060S has 128 GB shared with the 119 GB box and the oracle server.
- Repo drift: `ea0c79b` vs HEAD `ae9327c` (file contents of the Mainline_Support GGUFs unchanged? the Q4/Q8 sha256 equal the sizes and hashes the API reports today: yes for both, so the pinned files are the current ones).

---------------------------------------------------------------------------------------------------------------

## H. Work plan (ordered, one agent per task)

Notation: laptop root `C:\Development\inkling-loom-serve` (LR); box work dir `~/inkling/<NAME>`. Skills env: `export LOOM_ROOT=~/loom`; run skills by absolute path `~/inkling/loom-skills/skills/...`; tools venv `~/loom/.tools/bin/python -I`. Never edit `~/inkling/loom-skills`, `~/hrx-sdk`, `~/hrx-pr1`, other agents' dirs.

**T0 Decisions** (owner): answers to section F (1, 3, 4, 6, 8 are needed before T1/T8).

**T1 Layout build script (restructure into the published layout)**
- Inputs: `package/` (card.json, all .loom, tokenizer.json, weights.loom, prepare.loom), probe `~/inkling/inkL/probe.py` as a draft.
- Output: `scripts/publish_build.py` (reproducible, no hand edits) producing `publish/unsloth-Inkling-2-Mini-loom/{model.json, program/**, weights.loom, tokenizer.json, .gitattributes, LICENSE/NOTICE}` from `package/`; `capture` variants/presets, `test/`, `devices/` dropped; presets renamed per skills (`chat` = Q4_K_XL tuned for Strix Halo, `chat-r9700` = R9700; Q8 variant if owner wants it); `origins.checkpoint` -> `unsloth/Inkling-2-Mini-GGUF@ea0c79b...`; per-file `origin` `@checkpoint/Mainline_Support/<name>.gguf_file`, `path` `parameter/<name>.gguf`; `prepare.loom` data lists `llm_config.json`; `model.license apache-2.0`; scrub (no hostnames, users, `~/inkling` paths) with a grep check script.
- Files owned: `scripts/publish_build.py`, `publish/**` (generated, not hand-edited), `docs/PUBLISH_LAYOUT.md` only if needed.
- Acceptance: (a) `loom-fetch ./model.json --update-lock --all-variants --origin checkpoint=<snap>` writes a lock with the two known sha256; (b) `loom-fetch publish/... --materialize=copy --output X` into a clean dir (parameter files from the local snapshot via `--origin`), then `loom-serve --model=X` (flock -s) registers on both GPUs and answers the 44-case template echo and one chat request, token-identical (greedy 64 tokens) to the old `package/` route on prompts A/B/C; (c) the old tests still run against `package/` (unchanged).

**T2 Work dir in the skills' shape + llm_config.json + catalog.json**
- Inputs: `~/inkling/inkL/cfg.json`, `catalog.py` doc, ARCHITECTURE.md memory numbers, `loom-install.json`.
- Output: `scripts/make_llm_config.py` (writes `publish/.../llm_config.json` and a work dir `~/inkling/<NAME>/W/` = `pkg/` (materialized publish tree with `parameter/` symlinks), `cfg.json`, `gguf/`, `bench/`, `catalog/catalog.json`). `catalog.json` values: source repo/revision/file, `variant`, `preset`, `context` (T5 result), `prefill_capacity 512`, `memory{resident_weights_bytes 17617780736 (Q4) / 22855473152 (Q8), kv_bytes_per_position 32768 (check), workspace_gib, over_cap false, note}`, `sdk` from `loom-install.json`, `tokenizer thinkingmachines/Inkling-2-Mini`. Fills `generator` (`kit_source.generator_stamp`), `sdk`, `resolved` (exempt block), `model_id`, `chat` (from T3).
- Acceptance: `python3 -I starter/catalog.py source catalog` prints the line; `CATALOG=<dir> port.sh catalog ...` (or just `catalog.py build`, if port.sh needs the HF download we skip: use `FROM=build KIT=...` as in new-arch step 6) produces `pkg/` that `serve.sh` starts; `cfg` key list matches the example (`name, arch, gguf, hidden, ... chat, model_id, draft, sdk, generator, resolved`).

**T3 Chat kit (`chat_probe.py` for Inkling) and chat block**
- Inputs: `docs/chat_template.jinja`, CHAT_POLICY.md, `scripts/chat_golden.py`, `scripts/chat_cases.py`, gpt-oss and glimmer examples (`reference/gpt-oss-example/chat/harmony.loom`, `glimmer-example/chat/atem.loom`) as format precedent, `starter/chat_probe.py` interface (`simulate(chat, messages, tools, gen, enable_thinking)`, `render_calls`, `raw_from_chat(chat, message, finished, thinking, prompt)`, `derive/verify`).
- Output: `kit/chat_probe.py` (starter copy extended; `LOOM_KIT=kit`), `cfg.json["chat"]` hand block incl. `nonspecial_tokens`, effort mapping, `terminal`, `think` markers; `chat_probe.py GGUF --verify cfg.json` all lines OK (chat, multi-turn, tools, system+tools, agent loop, parallel calls, effort variants).
- Spike in this task: verify `/v1/chat/completions/input_tokens` and `prompt_tokens_details.cached_tokens` exist in our loom-serve route; list what else the scripts call (`/v1/status`, `/metrics`).
- Acceptance: every `--verify` line OK on CPU; `agent_golden.py --help` imports resolve with `LOOM_KIT`; kit `simulate` equals our Loom `input_tokens` for the 44 cases.

**T4 CPU reference server and golden recordings**
- Inputs: `~/inkling/inkB/build/bin/llama-server`, docs/ORACLE.md, T3 kit.
- Output: `scripts/oracle_server.sh` (start/stop by PID; flags `-ngl 0 -c <ctx+1024> -np 1 -t N -ctk f32 -ctv f32 --jinja`; Q4_K_XL and Q8_K_XL; port in the 18xxx range), `bench/llamacpp_args.txt`, `bench/llamacpp_pick.json` naming the CPU build (commit, flags); first `golden.py`/`golden2.json` run (12-20 short prompts + `--long 1000,2000`, 64 tokens) cached in `bench/llama_cache`; timing of CPU prefill/decode for the plan.
- Acceptance: `/completion` returns tokens + `completion_probabilities` (top 5) on the oracle build; golden.json and golden2.json written; rerun is served from cache; record seconds per prompt to size the long goldens.

**T5 Correctness gates on both GPUs (dev level), kernel-test adapter, served-context table**
- Inputs: T1 package served on gfx1151 and gfx1201 (Q4_K_XL, then Q8 if wanted), T3 kit, T4 server (`LLAMA_URL`), skills scripts.
- Output: `bench/<gpu>/{gates.json, gates_*.log, compare.json, consistency, agent, sampling, context_admit.json}`; `scripts/run_gates.sh` (sets `LOOM_ROOT, LOOM_KIT, LLAMA_URL, LEVEL=dev`, `flock -s`); `scripts/ktest_adapter.py` turning `package/test/*` runs into `kernel_tests` evidence (pass/total) or `loom_test` targets; `docs/CONTEXT_TABLE.md` (largest `context_capacity` per GPU/quant that registers and where `ctx_admit --complete` finishes; `ctx_admit.py --tokens N`).
- Acceptance: compare PASS (all divergences near-ties, max gap reported, number of full matches), consistency PASS, agent `render_ok/reuse_ok/calls_ok` PASS with thinking on and effort none, sampling PASS, ctx admit ADMITTED at the preset context. A FAIL is investigated per `topic/benchmark.md` (re-record, `-ub 128/2048`), never loosened. Quick level (`20 short + 1k-32k` docs) only if T4 shows long CPU prefill is affordable.

**T6 Performance in the skills' method**
- Inputs: T1 package, `perf.py`, `c4_steady.py`, `docs/DEVICES.md`.
- Output: `bench/<gpu>/perf_log.jsonl` (labels `loom-baseline`; Loom with `completion_spin_ns=4294967295`, `--pool_backing=fixed`, `--parameter_mode=preload` if needed), 3 sessions each (6 with `--repeat-rows 3` for decode > 100 tok/s: ours is below, so 3), `--long` rows within the served context, `c4_steady 4` (rows 4); CPU llama.cpp row via `perf.py --kind llama` against the oracle server (labelled CPU); `ev_perf.json` from `make_info.py perf --source-kind machine --baseline-version "llama.cpp-cpu/418fe97+PR32704"` (unverified that free text is accepted) per GPU.
- All timed runs under `flock -x ~/hrx-pr1/gpu.lock`, warmed, idle box checked.
- Acceptance: median over sessions with raw runs listed, noise noted, decode rows with generated tokens >= half max_tokens, numbers consistent with INTEGRATION_LOG within noise (R9700 Q4 decode ~86 tok/s, 8060S ~39).

**T7 Roofline for the looped architecture**
- Inputs: ARCHITECTURE.md section 12, `scripts/cost_model.py`, `scripts/floor_bench.py`, device presets in `loom-kb/device/*/device.json`, docs/DEVICES.md.
- Output: `scripts/roofline_inkling.py` writing `bench/<gpu>/roofline.json` in the schema `make_info.py roofline` reads (keys per metric: ceiling, target/stretch, efficiency; device constants; no `roofline.py` edit), plus `ev_roofline.json` via `make_info.py roofline` if it accepts a precomputed table, else hand-assembled evidence in the same schema (unverified). Include per metric: decode short/2k/32k (KV with sliding window: 20 of 24 layers read 1024 positions), prefill 1k..32k, streams4.
- Acceptance: ceilings agree with the documented floors (154.6 / 57.3 tok/s Q4 decode; Q8 107.3 / 39.8); cross-check against `roofline.py --device ... --bw ... ` with the loop bytes substituted; Loom efficiency < 0.97 everywhere.

**T8 Card generation (info.json, README.md, REPORT.md)**
- Inputs: T1-T7 outputs.
- Output: `make_info.py gates ... -o ev_gates.json`, `make_info.py build publish/... --repo local/unsloth-Inkling-2-Mini-loom --memory-bytes N --output <dir> --report generated --gpus strix-halo,radeon-r9700 --evidence ...`, `render_card.py publish/...` writing README.md, REPORT.md, info.json. No hand edits. Records the deviation fields (`baseline.differs`, "not supported" Vulkan cell).
- Acceptance: schema validation (`make_info.py` own check, `info_schema.json`); README quick start commands resolve locally; no hostnames/usernames in the three files; every cell is either evidence or "no evidence/not measured"; spike result documented if `make_info.py` rejects a hand-written package (then report the exact error to the owner; do not patch skills).

**T9 Local "publish" check (scrub + clean-room serve)**
- Inputs: T8 tree.
- Output: `scripts/publish_check.sh`: grep for `user|gpu-box|/home/|C:\\|inkK|~/inkling`; `loom-fetch <dir>/model.json --output clean --materialize=copy` into an empty dir with `--origin checkpoint=snap`; `loom-serve` on each GPU, one request, greedy equal to T1. Checks that `.gitattributes`, LICENSE, NOTICE are present. No upload.
- Acceptance: all checks pass; `publish/` is the deliverable folder (size without weights).

**T10 Optimize loop for gfx1201 and gfx1151 (only with owner go)**
- Inputs: T5 gates, T6 baseline rows, T7 roofline, `loom-optimize/SKILL.md`, `topic/optimize-loop.md`, `topic/decode.md`, `topic/prefill.md`, `device/rdna4/*.md`, `device/rdna3/strix-halo/*.md`, INTEGRATION_LOG M8 list (merge small mix/norm/conv launches, head, lm_head launch count; gap vs floor: decode 1.9x on R9700, 1.5x on 8060S; ~480 launches of 3-8 us per token).
- Method: (1) HAL profile per scenario (`--device_profiling_mode=dispatch-events,executable-metadata --device_profiling_flush_interval_ms=5 --print_device_statistics`, graceful stop, `profile_requests.py`/`profile_stats.py` with our own families file mapping our kernel names, `busy` vs wall, no `dropped_records`); (2) bucket budgets: decode step = floor bytes at measured bandwidth + overhead buckets (launch gaps, mix/ctrl, attention, router, head); prefill = GEMM flops at best GEMM rate; (3) design with per-arch tune rows (owner rule: data, not arch strings; Loom template families); (4) kernel level: `loom-opt`, compile report, `ab_kernel.py`; (5) `ab_perf.py` ABBA end to end against a base sha; keep only wins (> noise, no row loses > 2-3 %); (6) `LEVEL=dev` gates once on each winner (numerics-moving kernels gated before their speed counts); (7) record in `bench/decisions.jsonl` and `perf.py` labels. Stop reasons as in `loom-optimize` "Keep going". One agent per lane (decode launch merge; head; experts; attention proj GEMV; prefill GEMM), each on its own package copy and GPU time slot.
- Acceptance per lane: target row met or a stop reason with numbers; gates dev PASS; profile table before/after in `bench/`; `perf.py` verdict printed (DONE/REVIEW/BUDGET).

**T11 (optional) Q8_K_XL variant parity**
- Repeat T5 (dev) and T6 for `ud-q8-k-xl`; second preset; memory 22.9 GB + 6.7 GB on R9700 (one server at a time).

**T12 Finalize local report**
- Output: `docs/LOOM_SKILLS_STATUS.md` (what ran, deviations, numbers, open items), update `docs/RUNNING.md` for the publish layout; owner-facing summary. `finalize.sh`-equivalent steps by hand: final perf, `LEVEL=full` where the CPU cost allows, `audit.py` (a documented waived Baseline check), `report.py` page if wanted.

Dependency order: T0 -> T1 -> T2 -> (T3 || T4) -> T5 -> (T6 || T7) -> T8 -> T9 -> T12; T10 and T11 after T5 and T7, T11 may run in parallel with T6; T10 results re-enter T6/T8 (re-measure, rebuild the card).

Reproduce today's probes:
```
ssh user@gpu-box
export K=~/inkling/loom-skills/skills; T=~/loom/.tools/bin/python; G=~/models/inkling2-mini/Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf
cd ~/inkling/inkL
$T -I $K/loom-port-llm/starter/inspect_gguf.py $G cfg.json
$T -I $K/loom-port-llm/starter/chat_probe.py $G                      # ProbeError, see 0.4
python3 -I $K/loom-kb/kb.py route --arch qwen3moe --facets moe,sliding-window --device radeon-r9700 --topic porting,new-arch,benchmark
$T -I $K/loom-roofline/roofline.py $G --cfg cfg.json --device radeon-r9700   # 264.8 tok/s decode ceiling (loop not modelled)
cd probe && ~/loom/.venv/bin/loom-fetch ./model.json --update-lock --all-variants --no-input --origin checkpoint=$HOME/inkling/inkL/snap
```
