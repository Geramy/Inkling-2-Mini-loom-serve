# Issue draft 7: generated-token checkpoints or common-prefix rewind

Type: issue / design note. No PR.

## Title
Reuse generated state when the client does not echo `reasoning_content` (checkpoints of generated tokens or common-prefix rewind)

## Body
A retained row keeps the state of its last conversation, prompt and generated tokens. A new request reuses it only if its prompt is an exact extension of that history. Most chat UIs and agents do not send `reasoning_content` back, so the next prompt differs from the retained history at the thinking block, nothing matches, and the prefix cache can reuse only the prompt side (up to the last prefill boundary).

Measured (0.3.0, Inkling, system prompt about 4.5k tokens, three turns, TTFT of streaming):

| Turn | reasoning echoed: cached / TTFT | not echoed, prefix cache 0: cached / TTFT | not echoed, prefix cache 16: cached / TTFT |
|---|---|---|---|
| 2 | 5249 / 0.069 s | 0 / 1.46 s | 4096 / 0.34 s |
| 3 | 5660 / 0.071 s | 0 / 1.47 s | 4096 / 0.35 s |

Proposal, either of:
1. Rewind a retained row to the longest common prefix with the new prompt (positions beyond it are dropped; KV pages are position indexed, but recurrent or ring state cannot rewind, so this needs a checkpoint at or before the divergence), or
2. Capture endpoints at generated-token boundaries (for example at the end of each message and the end of the thinking block) so the common prefix has an endpoint.

For models with recurrent state, (2) is the workable one: a rewind has no state to rewind to without a checkpoint. We did not change the template or the generated tokens to force a hit.

Related report from another project: loom-skills issue 40, "Bolt-E4B: no prefix reuse across turns with thinking on" (github.com/loom-fun/loom-skills, issue 40; not linked here on purpose).
