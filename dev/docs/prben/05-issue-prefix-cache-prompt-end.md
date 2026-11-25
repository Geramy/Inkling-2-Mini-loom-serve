# Issue draft 5: prefix cache should capture the state at the end of the prompt

Type: issue draft (not filed). No PR: the change is in the service capture path (`service.c` `loom_serve_text_capture_prefix`, `prefix_cache.c`) and interacts with continuation epochs and the pending token, so it needs the maintainer's design.

## Title
Prefix cache: capture the prompt-end state, not only prefill boundaries

## Body
A follow-up request recomputes up to one prefill window of the previous prompt, even when the whole previous prompt is a prefix of the new one.

Repro (0.3.0, preset with `prefill_capacity` 2048 and `prefix_cache_capacity` 16, Inkling Q4 on gfx1201; chat with `reasoning_content` not echoed, so the retained row cannot be reused):
1. Send a chat request whose prompt is 5,133 tokens, `max_tokens` 1.
2. Send the same conversation plus the assistant reply and one short user message (5,165 tokens).

Observed `cached_tokens` on request 2: 4096. Expected: 5133 (the full previous prompt). Request 2 recomputes 1,069 tokens instead of 32.

Measured (same setup, one output token, TTFT of the follow-up): prompt 5133 cached 4096 0.47 s; prompt 9024 cached 8192 0.35 s; prompt 18006 cached 16384 0.72 s. With `prefill_capacity` 512 the hit is at most 511 tokens short (0.15 s at 18k), but cold prefill is 27 percent slower, so users trade one against the other.

Cause (reading main `5fbb30ba`): captures happen after each prefill epoch while the row is still in the PREFILL phase (`loom_serve_text_capture_prefix`). The epoch that consumes the last prompt tokens also selects the first output, so the row leaves PREFILL and no endpoint is captured at the prompt end. The documented near-end extension (`extension_start`) only adds a later boundary inside the last window.

Proposal: also capture the endpoint at the exact consumed frontier of the prompt (the state after the last prompt token, before any generated token), into an unused record, with the same rule that it never evicts another prefix. If continuation epochs can consume generated tokens before the capture runs, capture only when the row has consumed exactly the prompt.

Related documentation (see PR "Document the recurrent fork contract and prefix cache limits"): `prefix_cache_capacity` and `session_cache_capacity` semantics.

Also useful to know: a hit must give the same tokens as a cold run. We verified bitwise equal greedy tokens on 40 requests once the program copies the fork reader slot into the writer slot (a copy kernel in our package); without that copy the first span after each capture or hit read stale ring and conv state.
