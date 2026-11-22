# HTTP endpoints and fields the skills scripts use (T3 spike, bead llama-cpp-tg3.14)

Tested on a live `serve_card.sh gfx1201 q4` server (package of inkK, R9700, port 18420, started under `flock -s`, stopped by PID). Scripts: `skills/loom-benchmark/scripts` and `loom-port-llm/port.sh` (loom-optimize scripts make no HTTP calls of their own; they run perf.py).

## Endpoints

| Call (script) | Our server | What the script does when it is missing or different |
|---|---|---|
| `POST /v1/chat/completions`, JSON (agent_golden, compare, consistency, determinism, sampling_check, ctx_admit, spec_check, port.sh) | yes. Message has `content`, `reasoning_content`, `tool_calls[].id/function.name/arguments` (compact JSON string); `finish_reason` stop, length, tool_calls | n/a |
| `POST /v1/chat/completions` stream with `stream_options.include_usage` (bench_common.stream: perf.py, c4_steady.py, ctx_admit.py) | yes. Deltas carry `content` and `reasoning_content`; a last chunk with `choices: []` and `usage` | n/a |
| `usage.prompt_tokens`, `completion_tokens`, `total_tokens` | yes | n/a |
| `usage.prompt_tokens_details.cached_tokens` (agent_golden reuse_ok, consistency, determinism) | yes. Measured in the agent loop: 288 of 314 prompt tokens cached after a tool turn, 685 of 707 later | defaults to 0; reuse_ok would fail |
| `POST /v1/chat/completions/input_tokens` (agent_golden render_ok, port.sh) | yes, answers `{"input_tokens": N}`. Equals the kit `simulate` token count for 39 of 44 golden cases; the other 5 are rejected as designed (3 invalid efforts and 1 seven-decimal effort: the kit raises too; 1 tool message without name: the kit follows the template and renders 82 tokens, the server returns 400) | `counted.get("input_tokens")` is None: render_ok false |
| `GET /v1/status` (ctx_admit `data[0].loom.status.active_requests`; _lib.sh and xsj_perf `data[].loom.registration`) | yes. `data[0].loom.status.active_requests`, `.queued_requests`, `loom.registration` ("ready") all present | ctx_admit crashes on KeyError; _lib.sh waits |
| `GET /v1/models` (wait_model, spec_check, port.sh, find_reference) | yes. Also lists `supported_reasoning_levels` (none..xhigh), `default_reasoning_level`, `context_length` 32768, `input_modalities` ["text"] | wait_model cannot start |
| `GET /metrics` (bench_common.server_draft_counters, perf.py, c4_steady.py, gates.sh) | yes (Prometheus text). Has `loom_serve_model_*` queue gauges. No `field="draft_proposed_tokens"` or `draft_accepted_inputs` lines | draft_counters returns `{}`: acceptance is None, which is right for a package without drafting |
| `POST /tokenize`, `POST /detokenize` (port.sh, sweep_compile, tune.py) | yes (POST only; GET is 404). `{"content","add_special","parse_special"}` returns `{"tokens": [...]}` | only used while building a starter port |
| `GET /health` (bench_common wait, xsj_perf for non-loom engines) | 404 | the scripts call it for llama.cpp, not for Loom (Loom uses `/v1/status`) |
| `GET /props`, `POST /completion`, `/apply-template`, `/slots` | 404 | llama-server only (golden.py, compare.py, agent_golden --llama, vllm_shim); they target the reference server, not ours |
| `POST /v1/completions`, `/v1/responses` | 404 | not used by the skills |

## Request fields the scripts send to Loom

| Field | Our server | Effect |
|---|---|---|
| `temperature`, `top_p`, `top_k`, `seed`, `max_tokens`, `max_completion_tokens`, `stream`, `tools`, `reasoning_effort` (names, numbers) | accepted | n/a |
| `cache_prompt`, `n_probs`, `user`, `metadata`, top-level `enable_thinking` | accepted and ignored | n/a |
| `chat_template_kwargs` | object with `reasoning_effort` and/or `enable_thinking` (false = effort none, true = default effort), or null: accepted (bead tg3.17). Other keys, non-objects, wrong value types and conflicts with the top-level `reasoning_effort`: HTTP 400 | rules in CHAT_POLICY section 3 |
| `logprobs: true`, `top_logprobs`, `stop` | HTTP 400 | text_match.py check mode (logprobs) fails; only for engine-to-engine text comparison |
| `reasoning_content: null` in a history message | accepted here | n/a |
| tool message with `tool_call_id` and no `name` | HTTP 400 `a tool message needs a name; looking the name up from tool_call_id is not supported` | see finding 1 |

## Findings that block or change a skills run

1. OPEN, blocked by loom-serve (tg3.17): agent_golden.py appends tool results as `{"role":"tool","tool_call_id":...,"content":...}` without `name` (line 382). Our policy rejects this (CHAT_POLICY section 2: `chat_message` keeps one message and one 64-bit state, so it cannot look up the earlier call). Run unmodified, the first tool step passes and every later request fails (`1 turns, 3 failing`). With a copy of the script that adds `"name": c["function"]["name"]` (one-line sed, `~/inkling/inkN/agent_golden_named.py`, skills untouched) the real model passes: 6 turns, 0 failing, render_ok on all 6, reuse_ok on all 6 (cached 288/314, 386/409, 541/592, 685/707, 784/799), calls_ok on all, parallel calls (two get_weather in one turn) included. Options for the lead: (a) keep our own copy of the script under scripts/ (rule 5 of the brief); (b) let the Loom policy accept a nameless tool message and print `<|message_tool|>` without a name, which no longer equals the template (kit and server would both have to follow); (c) ask loom-serve upstream for the tool-call name to reach `chat_message`.
2. FIXED (tg3.17): `chat_template_kwargs` was rejected, so every script that sends `{"enable_thinking": false}` fails on our server: `wait_model.py` (called by `_lib.sh` restart and xsj_perf, exits with `first request failed: HTTP 400`), `perf.py` and `c4_steady.py` with their default `--thinking off`, `spec_check.py`. Use `--thinking template` (sends nothing) for perf.py and c4_steady.py. `sampling_check.py` and `ctx_admit.py` send it only when the block says `thinking: "toggle"`; the kit block says `"always"`, so they send nothing. agent_golden `--thinking on|off` would also fail; use the default. Option: accept `chat_template_kwargs` with only `enable_thinking` in `request_options` and ignore it (the template has no such variable). Owner of chat.loom decides.
3. There is no draft counter in `/metrics` (no drafting): handled by the scripts.
4. `text_match.py` and any path that asks for `logprobs` on Loom fail with 400; they are reference comparison helpers.
5. Real generation confirmed the tool-call grammar of CHAT_POLICY section 7: the model writes `<|message_model|>NAME<|content_invoke_tool_json|>{"name":..,"args":..}<|end_message|>`; `parse_tools` produced valid calls in the agent run (open question closed for greedy get_weather and calculator calls).

## Update (inkQ, 2026-11-21, bead llama-cpp-tg3.17)

- Finding 2 is fixed. `chat_template_kwargs` takes `reasoning_effort` and `enable_thinking`; `wait_model.py` works unmodified against the live R9700 Q4 server ("model ready after 3 s"), so `_lib.sh` restart, xsj_perf and the default `--thinking off` of perf.py and c4_steady.py now send a request the server accepts. The model still writes a reasoning block with effort none (model card), so `enable_thinking: false` shortens, not removes, reasoning.
- Finding 1 cannot be fixed in the policy. The lead's design (function name inside the tool-call ID) needs the policy to set the ID. loom-serve assigns it natively: `serve/service/text/tool_calls.c` writes `call_<request id>_<index>` in `loom_serve_text_tool_calls_append`, after `parse_tools` returned only name and arguments (`chat.c` reads `type` and `function` of each record). A live response confirms `"id":"call_7_0"`. History IDs also never reach the policy (`render_tool` gets name, arguments, index; `chat_message` gets the message `raw` and one 64-bit state, and the earlier assistant message cannot hand over a name text through 40 free state bits). The jinja template does not print the ID, so no rendering question arises. Unmodified agent_golden.py still stops after step 0 (`1 turns, 3 failing`, HTTP 400, now with a message that says `name` is needed). Options for the lead: (a) the copy of agent_golden with `name` (6 turns, 0 failing; run again on this build: cached_tokens 0, 288/314, 386/409, 541/592, 685/707, 784/799; render_ok, reuse_ok, calls_ok on all 6); (b) render a nameless tool message as `<|message_tool|>` without a name (the policy and kit would then differ from the template for that input); (c) ask loom-serve to pass the earlier call names (or the id-to-name map) to `chat_message`.
