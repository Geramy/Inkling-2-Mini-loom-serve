# PR 3: pass the function name of the call to tool messages

Branch: loom-serve-tool-name-from-call-id (https://github.com/Geramy/hrx-system/tree/loom-serve-tool-name-from-call-id)
Commit: 4bd399d5, author Geramy Loveless <geramy.loveless@amd.com>, base main 5fbb30ba
Patch: 03-tool-name-from-call-id.patch
Diff stat: chat.c 107 lines, tool_policy_test.py 76 lines (2 files, 177 insertions, 6 deletions)
Status: design proposal. The owner should confirm the approach before it is opened (see "Alternative").

## Title
[Loom] Give tool messages the function name of their call

## Body
The OpenAI protocol identifies the call that a tool message answers with `tool_call_id`; `name` is optional (and legacy). Chat templates (for example the llama.cpp minja render of the Inkling template) print the function name in the tool message header, resolved from the id. A text policy gets one message at a time and a 64-bit state, so it cannot look the name up. Our Inkling policy had to print a header without the name, which differs from the template.

Repro: any tool loop that sends `{"role":"tool","tool_call_id":"call_1","content":"20 C"}` after an assistant message with `tool_calls[0].id == "call_1"` and `function.name == "get_weather"`.

Before: `chat_message` gets the tool message JSON without any name.
After: `chat_message` gets `{"name":"get_weather","role":"tool","tool_call_id":"call_1","content":"20 C"}` in its `raw` argument.

Change: the history renderer records the decoded id and function name of each assistant tool call. A tool message with `tool_call_id` and no `name` key gets the name of the latest earlier call with that id, added to the raw message as if the client had sent it. A message with a `name` key (null included), an unknown id, or no id reaches the policy unchanged. The `chat_message` signature and the text service version do not change. Cost: one scan of a small record list per nameless tool message; no change for requests without tool messages.

Alternative I did not choose: add a trailing `tool_name` buffer to `chat_message`. It is cleaner, but it changes the exact-signature binding, every existing policy (fixtures, the tiny-text example, published packages) and the text service version. If you prefer it, the call-name table in this PR stays and the raw-message rewrite is replaced by one more argument.

Known limits: the role is matched on its raw text `tool`, so an escaped spelling of the role gets no name. A tool message that comes before its call gets no name.

Tests: `test_tool_message_reaches_source_with_the_name_of_its_call` runs a copy of `portable_chat.loom` that renders `%raw`, and checks: the id resolves, the latest call wins for a repeated id, a name with a quote is escaped, and a client name, a null name, an unknown id and no id stay as sent. The test fails on main and passes with the change (checked). I added an optional `control` argument to the test helper.

Verification: CMake lane (clang, Release) on Linux x86-64: `tool_policy_test.py` (11 tests), `source_chat_test.py`, `chat_policy_test`, `chat_test` all pass; clang-format, ruff format and ruff check clean. Not run: the Bazel lane, sanitizers, `chat_fuzz`.
