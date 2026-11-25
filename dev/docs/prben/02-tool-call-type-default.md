# PR 2: default tool call type to function

Branch: loom-serve-tool-call-type-default (https://github.com/Geramy/hrx-system/tree/loom-serve-tool-call-type-default)
Commit: 8f4f6b56, author Geramy Loveless <geramy.loveless@amd.com>, base main 5fbb30ba
Patch: 02-tool-call-type-default.patch
Diff stat: chat.c 6 lines, tool_policy_test.py 20 lines (2 files, 24 insertions, 2 deletions)

## Title
[Loom] Default the tool call type to function in chat history

## Body
An assistant `tool_calls` entry in request history without `type` fails today. `function` is the only defined type, and some agent clients leave the field out.

Repro (0.3.0 and main `5fbb30ba`), `POST /v1/chat/completions`:

```json
{"model":"M","messages":[
 {"role":"user","content":"hi"},
 {"role":"assistant","content":null,"tool_calls":[{"id":"call_1","function":{"name":"get_weather","arguments":"{\"city\":\"Paris\"}"}}]},
 {"role":"tool","tool_call_id":"call_1","content":"20 C"}]}
```

Before: HTTP 400 `assistant tool_calls must be functions`. After: HTTP 200.

Change: `type` may be absent or null and then means `function`. `"function"` is accepted as before. Any other value (`"custom"`, an empty string, a number) is still rejected. History only: the check for calls that the model makes is not changed.

Scope: 6 lines in `chat.c` and one test (`test_history_call_type_defaults_to_function`).

Verification: CMake lane (clang, Release) on Linux x86-64: `tool_policy_test.py`, `source_chat_test.py`, `chat_policy_test`, `chat_test` all pass; clang-format, ruff format and ruff check clean. Not run: the Bazel lane, sanitizers, `chat_fuzz`.
