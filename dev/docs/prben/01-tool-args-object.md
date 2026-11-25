# PR 1: tool call arguments as a JSON object

Branch: loom-serve-tool-args-object (fork Geramy/hrx-system, https://github.com/Geramy/hrx-system/tree/loom-serve-tool-args-object)
Commit: 6b174210, author Geramy Loveless <geramy.loveless@amd.com>, base ROCm/hrx-system main 5fbb30ba
Patch: 01-tool-args-object.patch
Diff stat: chat.c 25 lines, tool_policy_test.py 25 lines (2 files, 42 insertions, 8 deletions)

## Title
[Loom] Accept JSON object tool call arguments in chat history

## Body
Assistant `tool_calls[].function.arguments` in request history must be a JSON string today. The OpenAI SDK sends a string, but llama.cpp, vLLM and several agent clients send the arguments as a JSON object. loom-serve answers these with HTTP 400.

Repro (loom-serve 0.3.0 and main `5fbb30ba`), `POST /v1/chat/completions`:

```json
{"model":"M","messages":[
 {"role":"user","content":"weather in Paris?"},
 {"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"get_weather","arguments":{"city":"Paris"}}}]},
 {"role":"tool","tool_call_id":"call_1","content":"20 C"}]}
```

Before: HTTP 400 `chat field 'arguments' expects JSON type string`.
After: HTTP 200. The policy sees `{"city":"Paris"}` (the object text as the client sent it).

Change: `loom_serve_text_chat_function_decode` takes `allow_object_arguments`. Request history passes true. Calls that the model makes pass false, so they still need a string (the existing test `test_later_invalid_record_publishes_no_partial_result` still covers this). Number, array, null and boolean arguments are still rejected.

Design note: the object is passed as its original text, without compaction or re-encoding, so whitespace and escapes are the client's. A policy that wants a canonical form must normalize it. I chose this because the runtime does not need to parse the value, and the string path stays bit for bit unchanged. If you prefer compact re-encoding, that is a small change.

Scope: no ABI change, no policy change, no unrelated edits. It touches the same function as the "tool call type" and "tool name from id" PRs; they are independent and I can rebase in any order.

Tests: new `test_history_arguments_accept_a_json_object` in `tool_policy_test.py` (object form gives the same prompt as the equivalent string; 7, [], null and true are rejected).

Verification: built with the CMake lane (clang, Release) on Linux x86-64 and ran `tool_policy_test.py` (text_request_check + portable_chat.loom), `source_chat_test.py` (control.loom), `chat_policy_test` and `chat_test`. All pass. clang-format, ruff format and ruff check are clean. Not run: the Bazel lane, sanitizers, `chat_fuzz`.
