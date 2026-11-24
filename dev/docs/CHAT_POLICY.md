# Inkling 2 chat policy (bead llama-cpp-tg3.9)

Files (all written for hrx-loom 0.2.0, tested on both GPUs with the skeleton's placeholder model):

| File | Content |
|---|---|
| `package/control/chat.loom` | The text-service chat exports (`text_*_v1` names): `request_defaults`, `request_options`, `bind_request`, `chat_reasoning`, `chat_image_limits`, `chat_begin`, `chat_message`, `chat_end`, `render_tool`, `parse_tools`, `output_next`, `complete_text`, `prepare_input`. About 1450 lines. |
| `package/control/chat_json.loom` | Canonical JSON text: compact separators, `json.dumps(ensure_ascii=False)` string escapes, recursive key sorting. Public helpers `chat_put*`, `chat_decode`. |
| `package/control/chat_sampler.loom` | Kernel `inkling_sample_rows`: device sampler wired to the options record and a logits buffer. |
| `package/test/chat/` | `test_template.py`, `test_output.py`, `test_options.py`, `test_sampler.py`, `scenarios.py`, `golden.json`, `vendor/{sampling,random}.loom`. |
| `scripts/chat_*` | Golden generator, test package builders, kernel rewriters, runners, tokenizer patch. |

## 1. What loom-serve does with the response (measured)

The native session decodes every generated token with `IREE_TOKENIZER_DECODE_FLAG_SKIP_SPECIAL_TOKENS` (`serve/service/text/session.c:141`). With the tokenizer of the skeleton all 60 control tokens are `"special": true`, so `output_next` receives only the visible bytes: `<|message_model|><|content_thinking|>ab<|end_message|>` arrives as `ab`. A scripted kernel that emitted these tokens produced `abcdx{}` in `/v1/chat/completions` (JSON and SSE) on gfx1151; no marker survives and thinking, text and tool calls cannot be told apart.

Fix, no change to HRX: the structure tokens must be `"special": false` in `tokenizer.json`. `scripts/chat_tokenizer.py in.json out.json` does this for the 12 tokens `<|message_{user,model,system,tool}|>`, `<|content_{text,image,thinking,xml,tool_error,invoke_tool_json,invoke_tool_text}|>` and `<|end_message|>`. `<|content_model_end_sampling|>` (EOS) and `<|endoftext|>` stay special, so the terminal token is never written to the response. With the patched tokenizer the same script produced `<|message_model|><|content_thinking|>ab<|end_message|>...` in both modes, one SSE delta per token, and prompts tokenise identically (`/tokenize` of every reference prompt equals the llama.cpp token count; the echo test below compares IDs). `gen_tokenizer.py` (inkC) and the card's tokenizer must apply this patch. `/detokenize` always keeps special tokens.

The response never carries a token ID, so a literal `<|end_message|>` typed by the model as ordinary pieces is indistinguishable from the token. Accepted: it only ends the current message early.

## 2. Template rendering

Source of truth: `docs/chat_template.jinja` (identical to the GGUF `tokenizer.chat_template`). Reference prompts: Python jinja2 with the Hugging Face `tojson` (sorted keys, compact separators, `ensure_ascii=False`) and `generation` tags removed, cross-checked with llama.cpp minja (`llama-server /apply-template` of the oracle build, CPU only, started and stopped by `chat_golden.py`). 44 conversations in `scripts/chat_cases.py`, frozen in `package/test/chat/golden.json`.

Findings from the two engines:
- jinja2 and minja agree on every case except these, which are llama.cpp behaviours and not template behaviour: a final assistant message is treated as a prefill and loses `<|content_model_end_sampling|>`; numeric `reasoning_effort` and `"none"` are only honoured through `chat_template_kwargs` (a top-level `reasoning_effort: "none"` gives 0.9); floats print with 6 significant digits (`0.1234567` becomes `0.123457`); tools without `type`, or with `description: null`, are rejected by its schema check.
- The package follows the jinja2 result.

How the pieces map (`chat_begin`, `chat_message`, `chat_end`):
- Tool declaration (`<|message_system|>tool_declare<|content_xml|>[specs]<|end_message|>`) is the `chat_begin` prefix. Every spec is `{"description","name","parameters","type"}` with `description` default `""`, `parameters` default `{}`, `type` default `"function"`; `tool.function` or the tool itself; keys sorted at every depth; strings re-escaped.
- The effort header is written before the first non-system message (or at the end if there is none). State bits 0-19 hold the effort in millionths, bit 23 marks the header as written. Names: none 0, minimal 0.1, low 0.2, medium 0.7, high 0.9, xhigh and max 0.99, surrounding ASCII spaces trimmed. A number is accepted in [0, 0.99] with at most six decimals and prints as the shortest decimal (`0.25`, `0`). Null or missing means 0.9.
- Messages: system, user, assistant, tool as in the template; content is never trimmed; assistant `reasoning_content` becomes a `<|content_thinking|>` message first; an empty assistant content is skipped only when the assistant has reasoning or calls; a content array becomes one text message per part; assistant tool calls follow as `<|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{...}}<|end_message|>` (top-level argument keys sorted case-sensitively, values compact in source order); an assistant message ends with `<|content_model_end_sampling|>`; `name` is appended to the tool and assistant role tokens. No generation prompt: the model writes `<|message_model|>` itself.
- `render_tool` is the one formatter for client history and generated calls.

### Differences from the template (on purpose, all rejected with HTTP 400 unless noted)
- A tool message without `name` that only has `tool_call_id`: the template looks the name up in earlier assistant calls. `chat_message` receives one message and keeps one 64-bit state, so the lookup is not possible. Clients must send `name`. Encoding the name in the call ID does not work: loom-serve assigns the ID natively as `call_<request id>_<index>` after `parse_tools` (`serve/service/text/tool_calls.c`, `loom_serve_text_tool_calls_append`); the policy returns only name and arguments and cannot set or read an ID (`render_tool` gets name, arguments and index only; history IDs never reach the policy). The template does not print the ID, so a future framework change that passes the earlier call names to `chat_message` would make the lookup exact with no other change. The 400 text says that `name` is needed.
- An assistant message with both `name` and tool calls.
- Role `developer` or anything but the four roles (the template raises too).
- Images, video and audio (`chat_image_limits` returns zeros). Text parts only.
- Numeric effort with more than six decimals or outside [0, 0.99]; effort given as a string number.
- Not rejected, but different: JSON numbers inside tool schemas and tool-call arguments keep their source spelling (Python would print `1e3` as `1000.0`, `1.0` as `1.0`); duplicate keys are not merged; a non-string tool `description` becomes `""`; a content `null` and `[]` render nothing like the template, but an empty string renders an empty message (as the template).

## 3. Request options (`request_options`, `request_defaults`, `bind_request`)

One 24-byte record per request: `f32 temperature, f32 top_p, i32 top_k, i32 flags, i64 seed` (same layout as the library's Qwen record). Flags: bit 0 explicit seed, bit 1 tool calls forbidden. Verified through the real HTTP path (`test_options.py`, 32 checks per GPU): a test package copies the record to the epoch metadata and the kernel spells it back.

| Field | Rule |
|---|---|
| `temperature` | number, 0 (greedy) or a finite F32 up to 3.4e38. Default 1.0. |
| `top_p` | number in [1.18e-38, 1]. Default 1.0. |
| `top_k` | integer 0..2147483647 (0 off). Default 0. |
| `seed` | signed 64-bit integer. Without it the seed is the server's request id (differs per request). |
| `tool_choice` | `"none"` removes `<|content_invoke_tool_json|>` (token 200049) from every draw; any other value is ignored. |
| `reasoning_effort` | see section 2; read by `chat_begin`. |
| `max_tokens`, `max_completion_tokens` | native (1..2147483647, the larger-priority `max_completion_tokens`); the source has no hook. |
| null | keeps the default for the sampling fields and `tool_choice`. |

Duplicate sampling keys and wrong types are errors. Fields the model cannot honour are rejected unless they carry their neutral value (clients send these): `n` (1), `frequency_penalty` and `presence_penalty` (0), `repetition_penalty` (1), `min_p` (0), `logprobs` (false); `stop`, `top_logprobs`, `logit_bias`, `response_format` only as null (`logprobs`, `top_logprobs` and `stop` stay rejected on purpose). `chat_template_kwargs` is described below. Other unknown fields (`user`, `metadata`, `store`, `parallel_tool_calls`, ...) pass untouched. Deployment defaults: the definition `options` object may hold `{"sampling": {temperature, top_p, top_k, seed}}`; it is strict (unknown keys or null values fail the registration). `chat_reasoning()` returns `(1, 63, 4)`: none..xhigh selectable, default high.

`chat_template_kwargs` (read by `chat_begin`, and checked by `request_options`): null, or an object with only these keys; any other key is HTTP 400.

| Key | Meaning |
|---|---|
| `reasoning_effort` | The same values and rules as the top-level `reasoning_effort` (names, numbers in [0, 0.99] with at most six decimals; null means unset). llama.cpp takes numeric effort only this way. |
| `enable_thinking` | `false` is effort `none`, `true` is the default effort (high). null means unset; any other type is 400. |

Combination rules (all 400 with a message that names the fields): a top-level and a kwargs `reasoning_effort` that differ; `enable_thinking: false` with a non-zero effort (top-level or kwargs); `enable_thinking: true` with effort `none`. `enable_thinking: true` with a non-zero effort keeps that effort. Top-level `enable_thinking` is still accepted and ignored (the template has no such variable). A client may still see reasoning with effort none: the model card says minimum effort does not fully disable reasoning.

The row table is written by `bind_request` into the control state at byte `40 + row * 24`; the control state must be at least `40 + 24 * rows` bytes and grow with `resize_state`. The epoch must hand the table to the sampler (the test package copies it behind the plan records).

## 4. Output channels, tool calls and history

`output_next` states: 0 waiting for a message, 1 inside a reasoning message (kind 2), 2 inside a text message (kind 1). Markers are hidden (kind 0). A message header followed by anything but `<|content_thinking|>` or `<|content_text|>` (a tool call, `<|content_xml|>`, ...) is not consumed: the native loop stops and passes the rest of the response to `parse_tools`. While streaming, a trailing partial `<|end_message|>` is withheld; at the end it is released as text. Text with no message marker at all is plain content (state 2), which keeps the placeholder model usable.

`parse_tools` reads every `<|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{...}}<|end_message|>` block, checks that both names agree and are `[A-Za-z0-9_.-]{1,128}`, and returns the OpenAI records (`arguments` is the compact JSON of `args`). A malformed call fails the request (no partial calls). When the response was truncated, an unfinished last call is dropped and `finish_reason` stays `length`. Blocks of other kinds after the first call are skipped.

`complete_text` returns the canonical transcript: prompt, then `<|message_model|><|content_thinking|>R<|end_message|>` (if `R`), then `<|message_model|><|content_text|>T<|end_message|>` (if `T` is not empty or there is neither `R` nor a call), then the rendered calls, then `<|content_model_end_sampling|>`. The retained device state is reusable only if the generated bytes equal that text exactly and the parser ended outside a message. So reuse happens when the client sends the same assistant message back including `reasoning_content`; a client that drops it replays the history (correct, slower). Measured (`test_output.py`): after a scripted reasoning turn the echoed history reports `cached_tokens` 34 = prompt 17 + 18 generated minus the terminal token; with the reasoning dropped `cached_tokens` is 0.

`prepare_input`: user text becomes `<|message_user|><|content_text|>…<|end_message|>`; a terminated boundary needs no prefix because the canonical transcript already ends with the terminal text; an open boundary first writes the terminal token. Media markers are rejected.

## 5. Sampler (`chat_sampler.loom`)

`inkling_sample_rows(rows)[rows, logit_stride, policies, identities, logits, outputs]`: one 256-thread workgroup per selected row. `identities` is `view<[rows]x2xi32>` `{owner, position}` (owner indexes the 24-byte table, position is the absolute index of the drawn token); `logits` is F32 with 200058 columns per row and `logit_stride` elements between rows (0 reads row 0 for every draw, used by the tests); `outputs` is `view<[rows]xi32>`. The draw is the AMD motif `sampling.categorical_excluding_f32`: temperature, top-k, nucleus (smallest prefix with mass >= top_p), then an exponential race with Philox4x32-10 keyed by `(seed, position, token id)`, exact lowest-ID tie rule, greedy at temperature 0 or top_k 1, one excluded token. Nothing depends on row order or packing. The two motif files are vendored verbatim in `package/test/chat/vendor/` (MIT, AMD ROCm hrx-loom-library: `motif/tensor/sampling.loom`, `random.loom`); inkH must put them into `package/lib/vendor/` and `VENDORED.md`. The logits layout of inkF is not known yet: the kernel takes the stride as an argument and expects the 200058 real columns first.

Tests (`test_sampler.py`, loom-test with `--device`, 20000 draws per case, eight tokens with p = .30 .20 .15 .12 .10 .07 .04 .02 spread over the vocabulary, all others -30): same seed twice equal; chi-square p > 0.001; support (top-k, nucleus, exclusion, greedy); a different seed changes 64-86% of the draws. Tokens are bit-identical on gfx1151 and gfx1201. Limit: a token with probability below about 1e-6 is under-sampled by up to 15% (the uniform has 24 bits; temperature 2 with 200050 tail tokens at -30: 381 of 20000 draws outside the set against 451 expected). Irrelevant for text, but it is not an exact sampler in the far tail.

## 6. Tests and results

All run under `flock -s ~/hrx-pr1/gpu.lock`, servers started and stopped by pid.  The test packages put the option table at byte 128 (the inkC skeleton layout; `chat_build_test_pkg.sh` patches the constant, the card layout uses 40). Commands (box, from `~/inkling/inkI`, venv `~/loom-serve/.venv-inkI`):

```sh
scripts/chat_run_all.sh gfx1151 gfx1201       # template counts, template echo, output channels, options: builds four test packages per GPU
# other root, venv and ports: INKLING_CHAT_ROOT, INKLING_CHAT_PY, INKLING_CHAT_PORT_1151, INKLING_CHAT_PORT_1201
scripts/chat_run_defaults.sh gfx1151          # deployment defaults and rejected deployment options
cd package/test/chat && python test_sampler.py amdgpu://id/000000e2000000808e270f2b0bbb4e00 amdgpu://GPU-e0880c05003c324d
scripts/chat_golden.py 18400 oracle/chat_template.jinja golden.json   # needs llama-server of inkB on port 18400
```

| Test | What | gfx1151 | gfx1201 |
|---|---|---|---|
| `test_template.py` (counts) | input token count of the 44 cases equals the reference; the cases with an effort run a second time with the effort in `chat_template_kwargs`; 8 kwargs equivalences and 12 kwargs rejections | 83 PASS | 83 PASS |
| `test_template.py --echo` | echo kernel spells the prompt token IDs; all IDs equal the reference prompt tokens | 83 PASS | 83 PASS |
| `test_output.py` | 13 scripted generations, JSON and SSE (content, reasoning, calls, truncation, errors) and state reuse | 28 PASS | 28 PASS |
| `test_options.py` | option record, 15 accept and 20 reject cases | 36 PASS | 36 PASS |
| `chat_run_defaults.sh` | deployment defaults (4 checks) and 4 rejected option sets | 8 PASS | 8 PASS |
| `test_sampler.py` | 9 configurations: determinism, support, chi-square or fixed choice, seed change | 35 PASS | 35 PASS |

## 7. Open questions

- Tool-call grammar: the model card gives no tool format; the policy follows the template text only. If the model emits `<|content_xml|>` or `<|content_invoke_tool_text|>` blocks for calls they are passed to `parse_tools` and skipped (no call). Needs a real generation to confirm.
- Tool result name lookup by `tool_call_id` (see section 2): blocked by the native ID `call_<request id>_<index>` and by the one-message policy interface. Today the request fails with a clear message.
- Images: `chat_image_limits` is all zero. The template has `<|content_image|><|unused_200054|>`; vision is a separate component.
- Reasoning visibility: reasoning arrives as `reasoning_content`; the model card says minimum effort does not fully disable reasoning, so thinking blocks can appear with effort none.
- Marker tokens typed as plain pieces (section 1) cannot be told from real markers.
- A client that does not echo `reasoning_content` always replays; the prefix cache (`prefix_cache_capacity`) could soften this and is not enabled here.
- `max_tokens` is native; it counts the terminal token and cannot be tied to the effort.
- Rejected on purpose: silent acceptance of penalties, `stop`, `n`, `logprobs`, `response_format`, `min_p` with non-neutral values; guessing tool names from IDs; trimming message content (the template does not).

## 8. Lane CHATFIX changes (2026-11-24; details and evidence in docs/OPT_CHATFIX.md)

Section 2 "Differences from the template" and the tool-name text above describe the base. The chat policy now differs from the template in these further ways, all deliberate: a `tool` message without `name` is accepted and rendered without a name; role `developer` is a `system` message; an `assistant` message with tool calls ignores its `name`; the assistant `reasoning` string is read when `reasoning_content` is absent; unknown `chat_template_kwargs` keys are ignored; `tool_choice` `none` leaves the tool declaration out of the prompt. The prefix cache is enabled (`prefix_cache_capacity` 16); the sentence in section 7 about it not being enabled no longer holds.
