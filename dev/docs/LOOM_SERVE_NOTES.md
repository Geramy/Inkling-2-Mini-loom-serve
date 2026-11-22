# loom-serve notes (hrx-loom 0.2.0, written for the Inkling 2 Mini package)

Sources read: SDK docs `serving/*.md`, `serving/workflows/*.md`, the SDK tiny-text fixture (`loom/src/loom/serve/fixture/text`), the text service sources (`loom/src/loom/serve/service/text/*.h`, `README.md`) and the AMD library `ROCm/hrx-loom-library` (MIT, cloned to `C:\Development\hrx-loom-library`, read only). The venv used on the box is private: `~/loom-serve/.venv-inkC` (`hrx-loom[hf]` 0.2.0 plus `gguf`). The shared `~/loom-serve/.venv` was not touched.

## 0. The important discovery: a production reference exists

`ROCm/hrx-loom-library` already contains complete retained-text packages that load Unsloth GGUF files: `Qwen3.8-27B` (dense, UD-Q5_K_XL, MTP/DFlash) and `Qwen3.8-Flash-Next` (48 layers, 36 GDN recurrent + 12 sparse-attention layers, 512 experts, UD-Q4_K_XL, PLE, MTP, vision). They are run with `loom-serve --device=amdgpu://0` and they use paged KV, recurrent slots, forks, checkpoints and tool calls. This is the closest example of what we need and it answers most questions below with working code:

| Need | Where in the library |
|---|---|
| Cold `text_prepare_v1`, stage catalog per shape, storage geometry | `model/Qwen/Qwen3.8-Flash-Next/program/prepare.loom` |
| Epoch encode / publish / feedback packets | `program/text/qwen3_8/{encode,epoch,publish.cxx,storage}.loom`, `motif/sequence/qwen3_8/*.loom` |
| Chat policy (ChatML, reasoning, tools, streaming classification) | `program/text/qwen3_8/chat.loom` (87 KB), `chat_history.loom` |
| GGUF tensor binding by name, `"blk.{}.x"[%layer]` templated keys | `model/Qwen/Qwen3.8-Flash-Next/program/dispatch/*.loom` |
| GGML Q8_0 / Q4_K / Q5_K / Q6_K block readers over raw bytes | `motif/format/ggml/*.loom`, `motif/linear/ggml/*.loom`, `kernel/ggml/linear/*.loom` |
| MoE (routing, top-k, expert buckets, grouped matrix tiles) | `model/Qwen/Qwen3.8-Flash-Next/program/kernel/expert_*.loom`, `routing*.loom` |
| In-place weight re-layout at load (`prepare_weight` policy) | `.../program/weights.loom` |
| Card with variants, presets, memory estimates | `.../Flash-Next/model.json` |

The library is a Bazel repo, but its cards work from a plain checkout: a card lists `.loom` source files with `"origin": "../../../"`. We can vendor the shared files we need into `package/` and drop Bazel.

## 1. Package contract (retained text)

A package is `prepare` (cold VM) + `control` (warm VM) + numerical providers (`.loom` sources with `kernel.def` / `command.def`) + tokenizer + checkpoint. Direct definition (used by the skeleton) names exports without prefix (`prepare`, `epoch`, `chat_message` ...). A portable card names versioned exports (`text_prepare_v1`, `text_epoch_v1` ...). The source is JITted for the device by the server; there is no compiler subprocess in the warm path.

Cold `prepare(prefill, context, pool, rows, pins, draft_depth, shapes, shape_count, weights, options)` returns nine values:
`(effective_prefill, shared_stage_count, control_state, terminal_token_text, allocations, views, origins, draft_origins, geometry)`.
- It declares stages with `prepare.stage(root, tag)`, binds configuration with `config_i64(stage, key, value)`, declares checkpoint sources with `source(dir, path)` or `source_scope(name)` and binds parameters with `parameter(stage, binding, input, source, policy)`.
- `shared_stage_count` must be at least 1 and at most the declared stage count. The Flash-Next package returns its shape count (one stage per `{token capacity, span capacity}` shape; every bucket uses the same parameters). Error seen when wrong: `source bootstrap exceeds token runner capacities`.
- `terminal_token_text` is resolved through the tokenizer (we return `<|content_model_end_sampling|>`; the server then passes its id as `eos` to `encode_epoch`).
- Storage records: `allocations` = 11 x `{bytes, alignment, clear_bytes}` i64 triples (slot 0 residual, 1 arena = pooled KV pages then private row slots, 2 metadata packet, 3 tables = per-row origin records and page maps, 4 token inputs, 5 outputs, 6-10 draft/MTP slots). `views` = rows x 4 x `{offset, length}`. `origins` = rows x 5 i64 `{read anchor, write slot, base, fork frontier, page-map offset}`. `geometry` = header `{page tokens, map origin, carry stride, feedback split, recurrent origin, slot stride, slot length, slot count}` followed by regions `{allocation, origin, plane count, plane stride, page bytes}`. Flash-Next documents it in comments at `prepare.loom:141` and `:271`.
- `pool` is the number of pooled KV tokens; `rows` resident sequences; `pins` checkpoint endpoints. Growth: `resize_state(control_state, rows)` recomputes the same five records; existing byte offsets must not move.
- `chat_image_limits()` must return five zeros or five positive values. (Error seen: `source image limits must be all zero or positive bounded count, compressed bytes, width, height and pixels`.)

Warm `control` exports: `model_name`, `request_defaults(options)`, `request_options(defaults, request, nonce)`, `bind_request(state,row,options)`, `chat_reasoning() -> (supported, effort bitmap, default ordinal)`, `chat_image_limits`, `chat_begin(request, tools) -> (prefix, state)`, `chat_message(raw, role, content, parts, frames, reasoning, calls, state, ordinal, fields, part_count) -> (fragment, state, plan, count)` (`fields` bit0 reasoning, bit1 calls, bit2 tool id), `chat_end(state) -> (suffix, output_state)`, `render_tool`, `parse_tools`, `prepare_input(text, markers, marker_count, format, boundary, capacity) -> (tokens, count, spans)` (boundary 0 fresh, 1 open, 2 terminated), `output_next(response, cursor, state, phase) -> (kind, begin, end, state)` (kind 0 hidden, 1 content, 2 reasoning; bytes in `[cursor, begin)` are hidden framing), `complete_text(prompt, response, content, reasoning, tools, initial_state, final_state) -> (canonical transcript, reusable)`, `encode_state(bindings, count, origins)`, `encode_epoch(...) -> (output_count, transfer_plan, byte_offset, byte_length)`, `publish_epoch(...)`, `epoch(...) -> work completion`, `step` (isolated step, unused in pooled mode), `resize_state`.

Per-epoch data flow (one stage per shape bucket, mixed prompt and decode in the same packed epoch):
- Host plan record per span (9 u32): `{length, position, source position (2 x position for text), row, input offset, input length, flags (bit0 = select an output token), ...}`. A prefill span has length N, a decode span length 1. `publish_epoch` writes 21-i32 progress records `{consumed, ..., selected count, ..., next source position, selected token}`.
- `encode_epoch` writes a 16-byte header `{span count, eos id, first packet capacity, next packet capacity}` plus the plan into the metadata buffer and the token ids into the token buffer, and returns a transfer plan (5 u64 per record) so that only live bytes are uploaded.
- `epoch` calls `runner.execute_N(stage, buffers...)` then `runner.feedback(slot, buffer, offset, length)`. Argument counts are fixed per arity (`execute_6` in the skeleton, the six buffers are residual, metadata, origins, state arena, tokens, outputs).
- Kernels reach KV through the host-owned page map: physical page id from `origins[row].map[position / page_tokens]`, then `base + page_id * page_bytes + offset`. A fork reader selects `read anchor` when `position == fork frontier`, otherwise the private writer slot. The hostside never moves bytes; copy-on-write is native.

## 2. Model document and card

Direct deployment (used by `package/model.json`): `{"name", "device", "pool", "definition": {"kind":"text","prepare","control","sources":[...],"weights":"x.gguf","tokenizer":"tokenizer.json","rows","context_capacity","prefill_capacity","max_tokens", ...}}`. `weights` is one checkpoint path; `source(directory, path)` receives it as a buffer. A policy file (`weights.loom`, `prepare_weight(key) -> bytes`) is named by the cold program. Returning 0 loads the canonical tensor bytes unchanged. Returning a size with a stage root re-lays out the tensor in place at activation (Flash-Next does this for GDN Q8 projections).

Card (`model.json` with `programs`, `services`, `variants`, `presets`) supports several GGUF shards in one scope (`parameters.model.files` with `"format":"gguf"`), named parameter scopes (`source_scope("model")`, `"draft"`, `"vision"`), tokenizers (`"format":"huggingface-json"`), program sets, memory estimates and `fetch`. Prefer a card for publication. The skeleton uses the direct form because it is simpler to iterate on.

Capacities (`definition` or preset `execution`): `rows` (resident sequences), `max_sessions`, `context_capacity` (per row), `prefill_capacity` (largest packed token shape), `checkpoint_capacity`, `session_cache_capacity`, `prefix_cache_capacity`, `cache_target_bytes`, `max_tokens`, `draft_depth` (0..7), `continuation_epochs`. Pool backing is elastic by default: slabs are committed on demand within `memory_bytes`.

## 3. HTTP profile (observed)

Endpoints used: `/v1/models`, `/v1/status`, `/v1/chat/completions` (stream and not), `/v1/chat/completions/input_tokens`, `/tokenize`, `/detokenize`, `/v1/sessions`, headers `X-Loom-Session`, `X-Loom-Checkpoint`. `/v1/status` reports `registration` (`ready` or `failed` with `error.code/message`), `preparation_duration_ns`, `compilation` (commands, kernels, diagnostics), `parameters.primary` (`phase`, `active`, `reserved_bytes`, `committed_bytes`), `status.phase`, and the pool accounting.

### Tiny-text fixture (SDK, copied to `~/inkling/inkC/tiny-text`, weights file made as in the fixture README, `checkpoint_capacity` 2)

```sh
. ~/loom-serve/.venv-inkC/bin/activate
loom-serve --device=amdgpu://id/000000e2000000808e270f2b0bbb4e00 --model=tiny-text/model.json --listen=127.0.0.1:18081 --workers=2   # R9700: amdgpu://GPU-e0880c05003c324d
curl -s localhost:18081/v1/models
{"object":"list","data":[{"object":"model","id":"fixture-text","created":1795232964,"owned_by":"local","input_modalities":["text"],"output_modalities":["text"],"reasoning":false,"context_length":96,"limits":{"max_output_tokens":96},"loom":{"kind":"text","endpoint":"/v1/chat/completions","tokenize":true,"input_tokens":true,"variants":{"selected":1,"count":1},"incarnation":1,"registration":"ready","device":"device","pool":"pool"}}]}
curl -s localhost:18081/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"fixture-text","max_tokens":16,"messages":[{"role":"user","content":"abxa"}]}'
{"id":"loom-1","object":"chat.completion","created":1795232987,"model":"fixture-text","choices":[{"index":0,"message":{"role":"assistant","content":"bxaaaxb"},"finish_reason":"stop","logprobs":null}],"usage":{"prompt_tokens":4,"completion_tokens":8,"total_tokens":12,"prompt_tokens_details":{"cached_tokens":0}}}
curl -sN localhost:18081/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"fixture-text","stream":true,"max_tokens":4,"messages":[{"role":"user","content":"abxa"}]}'
data: {"id":"loom-2","object":"chat.completion.chunk",...,"choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null,"logprobs":null}]}
data: {... "delta":{"content":"b"} ...}   (then "x", "a", "a")
data: {... "delta":{},"finish_reason":"length" ...}
data: [DONE]
```

Both GPUs gave the documented outputs (`bxaaaxb`, `bx`+`length`). `/v1/status` for the fixture shows preparation 190-310 ms, `commands:2, kernels:3`, parameters `reserved_bytes 4096`, `backing elastic`, activation on the first request.

## 4. Framework floor measured with the SDK tiny fixture

`scripts/bench_serve.py` (60 warmed requests per cell, sequential, one HTTP connection per request, `flock -x ~/hrx-pr1/gpu.lock` around the client; the server itself ran unlocked and idle between runs). Prompt `abxa`.

| GPU | wait_mode | request, 1 token | per extra token | stream first chunk | stream inter-chunk | conc. 4 | conc. 8 |
|---|---|---|---|---|---|---|---|
| gfx1151 | active | 22.6 ms | 1.37 ms | 23.1 ms | 1.29 ms | 108 req/s | 126 req/s (1006 tok/s) |
| gfx1151 | blocking | 24.8 ms | 1.45 ms | 25.1 ms | 1.36 ms | 102 req/s | 101 req/s |
| gfx1201 | active | 23.0 ms | 1.61 ms | 24.7 ms | 1.52 ms | 100 req/s | 102 req/s (814 tok/s) |
| gfx1201 | blocking | 25.2 ms | 1.69 ms | 25.8 ms | 1.63 ms | 95 req/s | 112 req/s |

These numbers describe the fixture's configuration (`context_capacity 96`, `prefill_capacity 32`), not a fixed server cost. The same server with my placeholder package (trivial kernels) on gfx1151 shows that the floor depends strongly on the capacities. Request time for `max_tokens=1` (`scripts/bench_serve.py`, prompt `Hello`, 16 prompt tokens):

| context_capacity | prefill_capacity | 1-token request | 8-token request |
|---|---|---|---|
| 96 | 32 | 22.8 ms | 31.6 ms |
| 96 | 512 | 9.1 ms | 12.4 ms |
| 512 | 512 | 8.0 ms | 10.7 ms |
| 4096 | 32 | 4.4 ms | 6.3 ms |
| 4096 | 512 | 3.6 ms | 5.6 ms |
| 4096 | 4096 | 0.95 ms | 1.64 ms |

With `4096/512` the decode step (slope, `scripts/bench_decode.py`) is 0.27-0.29 ms/token on both GPUs and the request floor is 3.6 ms (gfx1151) and 2.9 ms (gfx1201). I did not find the cause of the capacity dependence (a pooling or admission cost that scales with page count is my guess; the page here is 16 bytes, a real KV page will be far larger). Conclusion: the serving floor is at most about 1-4 ms per request and 0.3 ms per decode step with a sane configuration, so it is small against the 6.5 ms (R9700) / 17 ms (8060S) weight-read floors in `ARCHITECTURE.md` section 12. Re-measure with the real page geometry before quoting a floor.

Kernel launch cost inside one command: a command with 744 dependent one-thread launches (experiment B) adds about 0.42 ms per decode step on gfx1151 (about 0.6 us per launch) and was not measurable on gfx1201 (0.263 ms/token against 0.288 for the 2-launch placeholder). Launch count per epoch is therefore not a constraint at the 1000-launch scale (48 visits x about 20 launches). JIT of that 744-launch command: 7 ms of compile stage time (4 distinct kernels).

## 5. Experiments (all on both GPUs, scripts in `scripts/`)

A. Placeholder package registers and answers (`package/`, `scripts/serve.sh`). `/v1/status` shows `registration: ready`, `reserved_bytes 874,323,968` (three bound tensors), `tokenize: true`. `scripts/placeholder_oracle.py` replays the kernel on the host from the GGUF bytes and the tokenizer output and compares: `jklcss` equals `jklcss` on both GPUs. So the whole chain tokenizer -> chat template -> paged state -> kernel reading GGUF tensors -> detokenizer works.

B. Whole-checkpoint bank. `scripts/gen_bank_probe.py` rewrites the kernel so that one `fold` launch per tensor xors that tensor's last byte into an accumulator that seeds the hash. All 744 tensors are bound as `view<Nxi8>` in one command. Result: `reserved_bytes 17,617,780,736` (the full file), activation in about 10 s (gfx1151) and 12 s (gfx1201), the oracle matches on both (`orjkng`). So a multi-GB bank with tensor offsets up to 17.6 GB, and a single tensor of 492 MB, bind and read correctly. Single views of 28.8 GB are used by the library (`per_layer_token_embd` in Flash-Next), so there is no 4 GiB view limit.
- Unused `command.parameter` bindings are dropped (reserved bytes only counted the 16 used tensors when the other 728 were declared but not consumed). A tensor must be consumed by a launch to be resident.
- `buffer.length` is rejected in kernels: `target 'amdgpu-rdna3-5' export 'fold' ... has no target-low contract for 'buffer.length' in '@fold'`. Pass the byte length as an `index` launch argument.

C. Tokenizer. `scripts/gen_tokenizer.py` builds a HuggingFace `tokenizer.json` from the GGUF (200,058 tokens, 424,413 merges, 60 control tokens as `added_tokens`, `Split` pre-tokenizer with the o200k regex of `ARCHITECTURE.md` section 10, `ByteLevel`, BPE, ByteLevel decoder; 12 MB). The loom-serve tokenizer (IREE tokenizer, supports `\p{..}`, `(?i:...)`, lookahead `\s+(?!\S)`) loads it, and `/tokenize` equals `llama-tokenize` on the same GGUF for 23 prompts (English, contractions, code, whitespace runs, CRLF, digits, Greek/Cyrillic/CJK/Korean, emoji, URLs, 200 repeated chars, special tokens in text). `scripts/tokenizer_check.py`: 0 mismatches of 23, both GPUs. Special tokens are matched in text by default (`NO_SPECIAL_TOKEN_MATCHING` flag exists).

D. Loom types. NVFP4 block decode compiles and is exact on both GPUs (`scripts/experiments/nvfp4_block.loom`, `loom-run ... --kernel-input-buffer="36xi8=74"` gives `-5` and `10` as expected): UE4M3 scale by `scalar.bitcast i8 -> f8E4M3` then `scalar.extf`, E2M1 code by integer arithmetic. `f8E4M3`, `f8E5M2`, `bf16`, `f16` are Loom types; the library uses `view<..xbf16>` loads with `scalar.extf`. Matrix fragments (`vector.mma`) exist and are used by the library for gfx11 and gfx12.

E. Chat framing. The skeleton renders the Inkling template (effort header before the first non-system message, user/system/assistant envelopes, no generation prompt). Prompt of `Hello` = 16 tokens = `llama-tokenize` of the same rendered text (the oracle script checks this indirectly through the matching replay). `reasoning_effort` is read in `chat_begin`; `/v1/models` reports `supported_reasoning_levels` none..xhigh and default `high`. A history turn that does not start with the terminal token after a generated EOS is rejected by the skeleton on purpose (the placeholder does not emit Inkling framing tokens); see the plan for the real policy.

## 6. Errors hit and fixes (minimal repro lines)

| Error text | Cause | Fix |
|---|---|---|
| `source bootstrap exceeds token runner capacities` | `prepare` returned 2 as shared stage count with one declared stage | return the stage count |
| `source image limits must be all zero or positive ...` | `chat_image_limits` returned (0,1,1,1,1) | return five zeros |
| `no target-low contract for 'buffer.length'` | `buffer.length` inside a kernel | pass the length as an `index` |
| `tar: ... in the future` | laptop and box clocks differ by about 10 s | harmless, `tar xm` |
| `inkling skeleton accepts text messages ...` on turn 2 of a session | my `prepare_input` rejects a suffix that does not start with the terminal token after an EOS | placeholder artifact |

## 7. Limits and gaps found

- Capacities: `prefill_capacity` at most 32768 for the library packages (a `require` in `prepare`), context a multiple of 64 there; page IDs are 32-bit (`UINT32_MAX` pages).
- Views over 4 GiB, 744 parameters in one command, 17.6 GB bank: all work (experiment B).
- Sampling is source policy: the device program owns the RNG and the logits stay on the GPU. A temperature 1.0 / top-p 1.0 / top-k 0 sampler needs a device sampler kernel (library: `motif/tensor/sampling.loom`, `random.loom`). The skeleton has no sampler.
- Tool calls: `parse_tools` returns function records; the optional `tools.parse_xml` codec is for the Qwen XML grammar. Inkling uses `<|content_invoke_tool_json|>{"name":..,"args":..}` so `parse_tools` must be written for it.
- Reasoning: output kinds 0/1/2 (hidden/content/reasoning) in `output_next`; `chat_reasoning()` declares efforts. Inkling needs a small state machine over `<|message_model|><|content_thinking|>`/`<|content_text|>` ... `<|end_message|>`.
- Image input is a separate finite component; out of scope here.
- Not measured: JIT time of the big model program (the library's Flash-Next programs compile on first load; their cost is not published), and device sampling.

## 8. The skeleton in `package/` (run and status)

Files: `model.json` (deployment), `prepare.loom`, `control.loom`, `config.loom` (checkpoint facts, linked through `config.def` + a `config.decl` in the kernel file), `kernel.loom` (placeholder `consume`/`select` kernels and the `advance` command), `storage.loom`, `weights.loom`, `devices/{gfx1151,gfx1201}.json`. The kernels bind `output_norm.weight` (F32 [2304]), `token_embd.weight` (Q8_0, 492 MB) and `blk.0.ffn_gate_up_exps.weight` (NVFP4, 382 MB) as raw views and mix their bytes into the placeholder recurrence.

```sh
scripts/sync.sh                      # laptop: copy package/ and scripts/ to ~/inkling/inkC on the box
scripts/restart.sh gfx1151 18090 25  # sync, restart serve (pid in run-gfx1151/pid), print registration state
ssh user@gpu-box 'cd ~/inkling/inkC; ./scripts/serve.sh gfx1201 18092'   # foreground start; INKLING_GGUF selects the file
```

`serve.sh` builds `run-<arch>/` with symlinks (the checkpoint appears as `weights.gguf`), generates `tokenizer.json` from the GGUF once, and starts `loom-serve --devices=package/devices/<arch>.json --model=run-<arch>/model.json`. Status after start (gfx1151, identical on gfx1201 with the R9700 name):

```
registration: ready   preparation_duration_ns: 342,898,222   compilation: commands 1, kernels 2, errors 0
parameters.primary: phase ready, active (after first request), backing elastic, reserved_bytes 874,323,968, committed_bytes 874,323,968
devices[0].name: AMD Radeon 8060S Graphics
/v1/models: reasoning true, supported_reasoning_levels none..xhigh, default high, context_length 4096, tokenize true, input_tokens true
```

Checks: `scripts/placeholder_oracle.py <port> <gguf>` (device answer equals host replay; `lebwii` on both GPUs), `scripts/tokenizer_check.py`, `scripts/bench_decode.py`. `scripts/serve_tiny.sh` runs the SDK fixture. All servers were started by me and stopped by PID file.
