# Inkling 2 Mini loom-serve package: plan

Inputs: `ARCHITECTURE.md` (inkA), `ORACLE.md` and `oracle/` (inkB), `LOOM_SERVE_NOTES.md` and the skeleton in `package/` (inkC). Status at the time of writing: the skeleton registers on both GPUs, binds the real GGUF, tokenizes exactly like llama.cpp and answers with a placeholder that is checked against a host replay. No model math exists yet.

## 1. Decision: derive from the AMD library, write the Inkling parts

`ROCm/hrx-loom-library` (MIT) has working text packages with the same ABI, GGUF loading, MoE and recurrent state (see `LOOM_SERVE_NOTES.md` section 0). Plan:

- Vendor, do not link, the shared text-service providers into `package/lib/`: `program/text/qwen3_8/{encode,epoch,publish.cxx,storage}.loom`, `motif/sequence/qwen3_8/{packet,feedback,span,input}.loom`, and the GGML readers `motif/format/ggml/q8_0.loom`. Their names carry `qwen38`; rename in the copy (one `sed`), keep the algorithms. This replaces the skeleton's hand-written `encode_epoch`/`publish_epoch`/`epoch`/`resize_state`/`storage.loom`, which only handle one 4-byte state word per token.
- Write the Inkling chat policy ourselves (about 400 lines). Qwen's `chat.loom` is ChatML with XML tools and is 87 KB; Inkling's template, reasoning and tool grammar differ, so copy only its structure (`output_next` state machine, `complete_text` canonicalisation).
- Write all numerical kernels for Inkling; reuse PR1 and library motifs where they fit (section 6).

## 2. State per sequence and storage geometry

Per visit v = 0..47 (`ARCHITECTURE.md` section 3): 8 global visits (v = 5, 11, 17, 23, 29, 35, 41, 47), 40 local visits. Every visit has its own KV and conv state. The three loop passes need no extra sequence state: each pass restarts from the prelude output inside the same token step (section 6.1 of the architecture), so `blocks`, `bank` and the messages live in activation workspace only.

| State | Size | Where | Why |
|---|---|---|---|
| Global KV, 8 visits, K and V F16 [1024] each | 4096 B/visit/token, 32 KiB per pooled token | pooled paged arena, one plane per global visit, page = 64 tokens (256 KiB per plane page), geometry region `{alloc 1, origin 0, planes 8, plane bytes pool*4096, page bytes 262144}` | grows with context, shared by forks and prefix endpoints through the host page map |
| Local KV, 40 visits, ring of 1024 tokens F16 | 4096 B x 1024 x 40 = 167,772,160 B | row-private recurrent slot, written at `position % 1024` | window is fixed, so a ring is 160 MiB per sequence instead of 4096 B x 40 per token without reclaim (32k tokens = 5 GiB) |
| Conv history, 48 visits x 3 x 6656 F32 | 3,833,856 B | same slot | zero at sequence start |
| Total slot | 171,606,016 B (163.7 MiB) | | a fork or endpoint copies one slot |

Consequences:
- `rows = 4`, `checkpoint_capacity = 4` gives `2*rows + pins = 12` slots = 2.0 GiB. Start with `session_cache_capacity = 0`, `prefix_cache_capacity = 0` (each automatic endpoint costs a slot) and enable later when the cost is measured.
- KV pool: 1 GiB per 32k pooled tokens. For 32k context and 4 rows use `pool` about 131k tokens = 4 GiB. R9700 budget: weights 16.4 GiB + slots 2 GiB + pool 4 GiB + workspace stays under 32 GB; the 8060S fits more.
- A global-layer decode reads `4096 B x n` per visit (about 1 GB at 32k, 1.6 ms at 620 GB/s). Use split-KV for decode spans.
- Local attention in a prefill chunk longer than 1024 must read the chunk's own K/V from the activation buffer (not the ring) and write only the last 1024 to the ring at the end of the layer. Chunk size must not change results (`ARCHITECTURE.md` section 9).
- Rewind: a ring cannot be rolled back. Same limit as the library's GDN state. The host keeps immutable endpoints at frontiers (checkpoints, prefix cache), each a slot copy.
- Position data: the plan record carries `position` and `source position`. No RoPE tables exist (`rope.dimension_count = 0`). `tau` (global) and relative distances need the absolute position of each token only.

Decision to revisit at M7: page the local KV as well (prefix sharing between sequences, no slot copy) if endpoint cost is too high.

## 3. Mixed prompt and decode

One command per shape bucket (the host owns the `{token capacity, span capacity}` catalog from `prefill_capacity`). All kernels are token-packed: GEMMs run over the T live rows, attention runs per span. A first kernel builds a token map from the nine-field plan: `{span, row, position, index in span}` (Flash-Next keeps a 4-word map). Decode spans are length 1 and use the same kernels; kernels read the live count from the metadata header so the shape specialisation is only about tile choice. Execution is layer-major over all T tokens, which is valid because token t's visit v only needs visit v-1 of the same token and of earlier tokens.

Per visit launch list (about 20, so about 1000 per epoch; measured launch cost is under 1 us, see notes):
1. `mix_attn` (S <= 9 states, nq = 1 local or 3 global; rms, dots, softmax, weighted sum)
2. passes >= 1 on looped layers: message mix (S = 2, nq = 4) and fused controller for Q, K, V
3. input rms norms (`attn_norm`, or q/k/v_inp_norm on global)
4. projections q, k, v, rel, sink (BF16), gate; global adds the two 32-row dynamic conv projections
5. conv k, v (4-tap + state, dynamic taps on global), L2 clip q/k/r with scales, v rms, `tau`, KV append (pages or ring)
6. attention (rel-bias gathered in-kernel from `r` and `rel_bias`, sink as extra logit, GQA 3:1, local window or full, paged)
7. gate multiply, o-proj, post conv, residual or block store
8. `mix_ffn`, controller (looped passes >= 1), `ffn_norm`
9. router GEMV F32, fused top-8 (sigmoid, +bias, top-k, normalise, x8)
10. expert gate_up, SiTU, expert down with the per-expert scale, weighted combine, post-ffn conv, residual
Epilogue (selected spans only): `mix(output_res_query)`, rms norm, lm_head, soft-cap 30, slice to 200058, device sampler.

## 4. Weights: loader policy and binding

- One GGUF is one checkpoint source: `source(directory, path)` (direct definition) or `parameters.model.files` with `"format":"gguf"` (card). Tensor names are the GGUF names (`blk.N.attn_q.weight` ...). The skeleton proves that all 744 tensors (17.6 GB) bind in one command and load in about 10 s on both GPUs. `prepare_weight` returns 0 (bytes unchanged). Re-layout in place (the library does it for Q8_0 projections) is an optimisation for M8.
- Bind with `command.parameter %bank, "blk.{}.attn_q.weight"[%layer] : view<3072x72x34xi8>` (rows x blocks x 34 B for Q8_0), `view<NxF32>` for F32, `view<NxBF16>` for BF16, and for NVFP4 experts `view<128x2304x36x36xi8>` (gate_up: expert x row x block x byte) and `view<128x2304x18x36xi8>` (down). The layer is a compile-time index of the layer command, so per-layer dtype choices (layers 20-23 have Q8_0 or BF16 experts and no `.scale`) are `scf.if` on the layer as in the library's `experts.loom`. A tensor that no launch consumes is not loaded.
- Types per checkpoint (verified inventories): Q4_K_XL = Q8_0 + NVFP4 + F32 + BF16 only; Q8_K_XL = BF16 + NVFP4 + F32. No K-quants occur. A single package serves both through a config flag `inkling.dense_type` (0 Q8_0, 1 BF16) set by the card variant.
- Quantized bytes reach kernels as raw buffer views; blocks are decoded in Loom. Q8_0: `ggml_q8_0_load_block` (f16 scale + 32 i8). BF16: `view<..xbf16>` + `scalar.extf`. NVFP4 (new motif `nvfp4.loom`): 36 B block = 4 x UE4M3 scales + 32 B of nibbles; sub-block s uses `qs[8s .. 8s+7]`, low nibble = element `16s+j`, high nibble = `16s+j+8`. Verified on both GPUs (`scripts/experiments/nvfp4_block.loom`): `f8E4M3` bitcast + `extf` for the scale, integer arithmetic for E2M1.
- Experts: gate_up tensor {2304, 2304, 128}: expert e row r (rows 0..1151 gate, 1152..2303 up) starts at byte `(e*2304 + r) * 1296`; down {1152, 2304, 128}: `(e*2304 + r) * 648`. Layers 20-23 (Q8_0): 2448 B and 1224 B per row. The per-expert F32 `.scale` multiplies the matmul output (gate_up output and down output) on every visit including passes 2 and 3.
- Size limits: no 4 GiB view limit (library binds a 28.8 GB view; the skeleton reads a 17.6 GB bank). Largest single tensor here is 721 MB. Count limit: 744 bindings in one command worked.

## 5. Tokenizer, chat, reasoning, tools

- Tokenizer: done. `scripts/gen_tokenizer.py` makes `tokenizer.json` from the GGUF; `/tokenize` equals `llama-tokenize` on 23 prompts. Publication can use the original repo's `tokenizer.json` if it exists; verify with `scripts/tokenizer_check.py`.
- Template: skeleton renders effort header + system/user/assistant. To do: tool declaration message (`tool_declare` + `<|content_xml|>` + sorted compact JSON), `tool` role with name, assistant `reasoning_content` and `tool_calls` (`<|content_invoke_tool_json|>{"name":..,"args":{sorted}}`), numeric `reasoning_effort`, image part (out of scope: reject).
- Reasoning: `chat_reasoning()` returns `(1, 63, 4)` (none..xhigh, default high). `max` aliases `xhigh`.
- Output: the model writes `<|message_model|>` itself. `output_next` is a state machine over `<|message_model|>` (hidden) then `<|content_thinking|>` (kind 2) or `<|content_text|>` (kind 1) until `<|end_message|>` (hidden); a block whose header is a function name followed by `<|content_invoke_tool_json|>` is withheld for `parse_tools`. Open question: whether the server decodes the response with special tokens kept (the library's Qwen policy matches `</think>` text; Inkling markers are special tokens). Test first in M6 with a recorded token stream.
- `complete_text` must return the exact generated bytes appended to the prompt, so retained state is reused for the next turn only when the client sends back the same assistant message. The template drops `reasoning_content` unless the client sends it; mismatch means replay, which is correct but slower.
- Sampling: temperature 1.0, top-p 1.0, top-k 0 default (GGUF `general.sampling.*`); `temperature 0` greedy for tests. Device sampler kernel from the library's `motif/tensor/sampling.loom` + `random.loom` (Philox, seeded per request). Request fields `temperature`, `top_p`, `top_k`, `seed` parsed in `request_options` (the library's `chat.loom` has a strict parser to copy).

## 6. Kernel list and reuse

Shapes from `ARCHITECTURE.md` section 14. "PR1" = `git show hrx-llama-pr1:ggml/src/ggml-hrx/kernels/...`. "Lib" = hrx-loom-library.

| Kernel | Reuse | New |
|---|---|---|
| Q8_0 GEMV and GEMM (q,k,v,o,gate,rel, lm_head; layers 20-23 experts) | PR1 `motifs/q8_0.loom` + `matmul/mul_mat.loom` (vec, tiled, per-arch tuned) and Lib `motif/linear/ggml/q8_0_q8_1_x4.loom`, `q8_0_bf16_matrix.loom` (wave32 matrix path) | launch ABI adapter |
| BF16 and F32 GEMV/GEMM (Q8_K_XL, router, loop controllers, sink, dynamic conv proj) | PR1 `motifs/float.loom` | small-row (24, 32, 128) split-K variant |
| NVFP4 expert GEMV (decode) and grouped GEMM (prefill) | PR1 `mul_mat_id.loom`, `mul_mat_id_wmma.loom` for the routing/bucketing structure; Lib `expert_routes`/`expert_buckets` | `nvfp4.loom` motif, `.scale` epilogue, f16 decode into the WMMA tile |
| top-8 router select | PR1 `argsort_f32`, `sum_rows_f32` | fused select kernel |
| SiTU | PR1 `unary_f32`/`swiglu_f32` for structure | tanh/sigmoid caps, fused into the down prologue |
| rms norm (weighted, unweighted, per head) | PR1 `norm/rms_norm_f32.loom`, Lib `motif/tensor/rmsnorm.loom` | per-head variant |
| L2 clip, rel-bias, tau, KV append | | new (fused prep kernel) |
| short conv with state (static taps) | PR1 `ssm_conv_f32.loom`, Lib `motif/tensor/convolution.loom` | dynamic-tap form, state write |
| attention (rel-bias, sink, window, paged, GQA 3:1) | PR1 `flash_attn/*` and `motifs/flash_attn.loom`, `kq.loom` for tiles; Lib `qsa_split_attention.loom` for split-KV over pages | new kernel (bias from `r` and `rel_bias`, sink logit, ring window) |
| mix (attention residual) and loop message/controller | | new (fused) |
| embedding gather + norm, lm_head soft-cap + slice, sampler | Lib `embedding.loom`, `sampling.loom`, `random.loom` | soft-cap epilogue |

Sharing PR1 motifs: put the PR1 motif files in `package/lib/` and list them in `definition.sources` (or card `programs.*.libraries`). The linker resolves only symbols reachable from `command.def` roots, so unused PR1 kernels cost nothing. PR1 kernels are written against `target.decl @ggml_wave32/@ggml_wave64` and `config.decl @ggml.mul_mat.*`; in serve these configs are set with `config_i64(stage, key, value)` in `prepare` and the target comes from the live device (`amdgpu-rdna3-5` on gfx1151). The installed SDK (0.2.0) parses the PR1 files (`loom-format --check` reports formatting only). Compiling them inside a stage is the first step of M1.

Per-arch tune data (owner rule: data per arch plus portable default, no arch strings): the cold program has no device query. Two routes, test both in M1: (a) `template.def` providers per target selected by the JIT target (the PR1 mechanism); (b) one card preset or variant per arch whose `config` carries the tune row (`inkling.tune.*`), chosen by `serve.sh` from the device file. Route (b) is the fallback and is plain data.

## 7. Package file layout (target)

```
package/
  model.json            deployment (skeleton) ; card.json (variants ud-q4-k-xl, ud-q8-k-xl; presets per arch)
  prepare.loom          cold: shapes -> stages, sources, parameter bindings, storage records
  control/chat.loom, chat_history.loom, encode.loom, epoch.loom, publish.loom   (vendored + Inkling chat)
  config.loom           checkpoint facts and portable tune defaults (skeleton has the facts)
  storage.loom          planes, slots, geometry (section 2)
  weights.loom          prepare_weight policy (returns 0 at first)
  lib/                  vendored PR1 and Lib motifs, nvfp4.loom
  kernel/               embed, mix, attn_prep, conv, attention, router, experts, loop_ctrl, head, sampler
  dispatch/             visit.loom, layer.loom, model.loom (command.def composition)
  tokenizer.json        generated from the GGUF
  devices/gfx1151.json, gfx1201.json    machine-local placement (not part of the card)
  test/                 loom-test cases, oracle comparison scripts
```

## 8. Milestones with oracle checks

Oracle: `oracle/dumps/Q8_K_XL_{A,B,C}` (f32 tensors per op, manifest with `layer` = cache slot), `oracle/gen/*` (64 greedy tokens), prompts A (16 tokens), B (308), C (25, chat templated). Tolerances from `ARCHITECTURE.md`: relative L2 per tensor about 1e-2 (NVFP4 and Q8_0 activation quantisation differences about 1e-3 per matmul), then logit top-k and KL.

To compare intermediates from a running server, expose them through the `sequence` service (`kind: sequence`, `capture_epoch` returns source-defined bytes per span, HTTP `/v1/sequences`). The same programs run as a debug deployment that captures a chosen visit's tensor; this avoids ad-hoc host readback.

| M | Deliverable | Check |
|---|---|---|
| M0 (done) | skeleton, tokenizer, GGUF bank, placeholder, host-replay oracle | `scripts/placeholder_oracle.py` MATCH; `scripts/tokenizer_check.py` 0/23 |
| M1 | Real storage geometry (vendored encode/epoch/publish) and one stage that runs embedding + norm; compile PR1 `mul_mat` in a stage; decide tune route (a) or (b) | `embd`, `norm` of prompt A; GEMV Q8_0 vs scalar reference on `blk.0.attn_q` |
| M2 | Visit 0 attention path (no attn_res, local): q,k,v conv, clip, attention, gate, o-proj, post conv | `Qcur-0`, `Kcur-0`, `Vcur-0`, `Rcur-0`, `kq_b-0`, `kqv_out-0` (A and C) |
| M3 | MoE on layer 0: router, top-8, NVFP4 experts, SiTU, scale, combine | `ffn_moe_logits/probs/topk/weights_norm/weights_scaled-0`, `l_out-0` |
| M4 | Layers 0-5 including global layer 5 (dynamic conv, tau, paged KV) | `l_out-1..5`; B (308 tokens) is shorter than the 1024 window, so add a synthetic prompt of more than 1100 tokens (ring wrap, chunk longer than the window) via the oracle tool |
| M5 | Loop passes (messages, controllers) and coda | `l_out` for slots 24..47 (pass 1, 2), `l_out-47` |
| M6 | Output head, soft-cap, sampler, chat policy (reasoning, tools) | `result_norm`, `result_output` top-k and KL; greedy 64 tokens equal `oracle/gen/Q8_K_XL_{A,B,C}`; chat C through HTTP |
| M7 | Multi-sequence: mixed prefill+decode epochs, chunk-size independence, fork/checkpoint, cancellation | same logits for chunk 16/128/512; two concurrent sequences equal isolated runs; fork equals replay |
| M8 | Per-arch performance (tile rows, small-row GEMV, expert grouping, prefill GEMM) | interleaved, warmed, `flock -x` runs against floors: 6.47 ms (R9700) and 17.4 ms (8060S) per decoded token for Q4_K_XL, FLOP floor from section 12.2 for prefill |
| M9 | Q8_K_XL variant, card, lock, memory estimates, serve scripts | perplexity vs `oracle/logs` (7.65), card fetch + serve |

## 9. What could not be done in Loom today, and risks

Cannot (or not verified):
1. Query the device from the cold program. Per-arch rows must come from target-selected templates or from a per-arch card variant (section 6).
2. Native FP4 matrix instructions. NVFP4 is decoded to F16 (exact for E2M1 x UE4M3 products below F16 range; check the extreme scales) and fed to wave32 matrix ops. On gfx1201 an FP8 matrix path with per-16 block scale applied on the accumulator might be faster; not available on gfx1151.
3. Page reclaim for a sliding window. Local KV uses a ring in the row slot (section 2).
4. Free-form debug readback of intermediates in the text service. Use the sequence service (section 8).
5. Dynamic loop count (`loop.varying`). The model here is fixed at 3 passes.

Open risks:
- JIT time of the full model command per shape bucket (not measured; Flash-Next presumably accepts it). Keep kernels shape-agnostic (live counts from the header) so distinct JIT keys stay few. Measure in M1.
- The framework request floor varies from 1 to 23 ms with `context_capacity` and `prefill_capacity` (notes section 4); measure with the real page size.
- Slot copy cost (164 MiB) for every fork, checkpoint, prefix endpoint.
- Precision: NVFP4 with F32 activations differs from llama.cpp (Q8 activations) by about 1e-3 per matmul; layer-wise tolerance must be set from a measured baseline.
- Loop controllers (F32 in Q4 file, 227 MB read per token) are 5.7% of the traffic; fuse early.
- The special-token decode question for `output_next` (section 5).
- gfx1151 and gfx1201 may need different tile choices for the same kernel; the per-arch data route must exist before M8.
