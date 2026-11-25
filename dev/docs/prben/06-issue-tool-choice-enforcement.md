# Issue draft 6: tool_choice required and named function need a sampler token set or mode

Type: issue / design note (not filed). No PR.

## Title
Chat: enforce `tool_choice` "required" and a named function (sampler token set or grammar)

## Body
`tool_choice` `"required"` and `{"type":"function","function":{"name":"X"}}` are accepted but not enforced. The model can answer in text with no call.

Repro (Inkling on loom-serve 0.3.0), tools `[search]`, `tool_choice: "required"`, user message "Say hello". Observed: content "Hello", `finish_reason` "stop", zero calls. Expected: a call to a declared tool (to `search` for the named form).

What a text policy can do today: only soft steering. Our policy ends the prompt with a system message "The next assistant message must be a call of a declared tool", and for a named function declares only that tool. That worked in 3 of 3 live cases but is not a guarantee. The sampler contract lets a policy exclude one token: excluding the text-content token made the model end the turn with the end token and still no call, so the exclusion cannot express "must call".

Proposal (needs a design from the maintainer): a per-request sampler constraint supplied by the policy, for example
- a token set that is allowed at the first generated position (the call token, or the call tokens of one named function), or
- a mode that forces the next structural token(s) and then releases the sampler.
The policy already knows its token ids at `chat_begin` time, so the request could carry the constraint next to the existing excluded-token value.

Also related: `tool_choice: "none"` is only reliable if the tools are left out of the prompt (we do this in the policy); excluding the call token alone gave an empty answer because the model had already decided to call.
