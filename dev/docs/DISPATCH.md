# Dispatch framework and package foundation (inkH)

Status 2026-11-20: M1 done, framework in place with stub hooks. Files: `package/{prepare,storage,config,card_defaults,capture_config}.loom`, `package/dispatch/*.loom`, `package/lib/vendor/`, `package/card.json`, `scripts/{serve,sync,restart,diag,card_test,capture_check,mulmat_check,peek_capture,floor_bench,jit_time}`.

## Layers

- `lib/vendor/` library v0.2.0 text-service providers (encode, publish, epoch, storage, packet geometry). `prepare.loom` builds the stage catalog (one stage per host shape bucket, root `inkling_model`) and the nine prepare results. `storage.loom` has the geometry below.
- `dispatch/model.loom` command `inkling_model` (one per shape bucket): allocates the shared buffers, runs `inkling_prep_spans`, the embed hook, the schedule, the head hook.
- `dispatch/layer.loom` command `inkling_visit(layer, visit, pass)`: the per-visit launch list. `visit.loom` has the pure helpers (`inkling_layer_is_global`, `inkling_layer_in_loop`, `inkling_expert_wtype`, `inkling_visit_index`). Everything is exact at plan time: global versus local, controllers (layers 6-17 with pass >= 1), expert weight type (layers 20-23 dense).
- Schedule (unrolled): layers 0-5 (visits 0-5), then passes 0..2 of layers 6-17 (visit = layer for pass 0, 24 + 12 (pass - 1) + (layer - 6) for later passes), then layers 18-23, then the head hook. 48 visits.
- Hooks: one `command.decl` per kernel step in `layer.loom`; one stub file per area (`stub_mix`, `stub_attn`, `stub_dense`, `stub_experts`, `stub_debug`). To integrate an area, replace its stub file in the source list (`scripts/serve.sh` `SOURCES`, `card.json` `libraries`) by a file that defines the same commands.

## Hook signature

Every hook: `command.def public @inkling_hook_<name>(%layer: index, %visit: index, %pass: index) launch(<20 buffers>)`. Buffers in order: `parameters` (weights bank), `span_table`, `token_map`, `metadata` (packet), `origins`, `arena`, `cur`, `blocks`, `bank`, `x`, `xn`, `small`, `q`, `k`, `v`, `r`, `gate`, `ab`, `ob`, `y`. Special: `inkling_hook_embed()` launch(parameters, metadata, tokens, embd, cur); `inkling_hook_head()` has the 20 buffers plus `outputs`; `inkling_hook_debug(id)` is the debug probe. Weight views come from `command.parameter %parameters, "blk.{}.x.weight"[%layer] : view<...>`. Hooks may add more buffers by allocating their own scratch inside (`buffer.alloca<global>`). If a hook needs another shared tensor, ask inkH to add a buffer to the list (one edit in `layer.loom`, `model.loom`, the stubs).

| Hook | Area | Launch list (docs/PACKAGE_PLAN.md section 3) | Reads | Writes |
|---|---|---|---|---|
| `mix_attn` | mix | attention-input mix (nb stored blocks, nq 1 local or 3 global) | blocks, cur | x, small (rinv) |
| `mix_msg` | mix | loop message mix, S = 2, nq = 4; only passes >= 1 on layers 6-17 | bank, cur | x, small |
| `ctrl_qkv` | mix | controller for Q, K, V (nc = 3); same condition | x, small | x |
| `norm_in` | dense | input rms norms (`attn_norm`, or q/k/v_inp_norm on global) | x | xn |
| `proj` | dense | q, k, v, rel, sink (BF16), gate projections; global adds the two dynamic conv projections | xn | q, k, v, r, gate, small |
| `attn_prep` | attn | conv k and v, L2 clip q/k/r with scales, v rms, tau, KV append (pages or ring) | q, k, v, r, small, arena | q, k, v, r, arena |
| `attn` | attn | attention (bias from r and rel_bias, sink, GQA 3:1, window or full, paged) | q, k, v, r, arena | ab |
| `oproj` | dense | gate multiply and o-proj | ab, gate | ob |
| `post_attn` | mix | post conv (conv_res, conv_commit), residual or block store | ob, cur, blocks | cur, blocks, bank, arena (conv state) |
| `mix_ffn` | mix | ffn mix | blocks, cur | x |
| `ctrl_ffn` | mix | ffn controller (nc = 1); passes >= 1 on layers 6-17 | x | x |
| `norm_ffn` | dense | ffn_norm | x | xn |
| `router` | dense (or experts) | router GEMV F32 and fused top-8 (exact kernel, not the matrix path) | xn | small (logits, top-k ids, weights) |
| `experts` | experts | gate_up, SiTU, down with per-expert scale, combine | xn, small | y |
| `post_ffn` | mix | post-ffn conv, residual, optional bank copy | y, cur | cur, bank, arena |
| `head` | dense (+mix) | `mix(output_res_query)`, rms norm, lm_head, soft-cap 30, slice 200058, sampler | cur, blocks | outputs |
| `embed` | dense | token gather and rms norm with `token_embd_norm` | tokens | cur (and embd for capture) |

The router row follows the stub assignment; inkD's doc may move it. The `small` buffer is `[cap][240]` F32 words per token: rinv 0-3, sink 4-27, dyn conv k 28-59, dyn conv v 60-91, router logits 92-219, top-k ids (i32 bits) 220-227, top-k weights 228-235. Other shared buffers (F32 words per token): `cur` 2304, `blocks` 8 x 2304, `bank` 2 x 2304, `x` 4 x 2304 (column stride cap x 2304), `xn` 4 x 2304, `q` 3072, `k` 1024, `v` 1024, `r` 384, `gate` 3072, `ab` 3072, `ob` 2304, `y` 2304. All are `cap = inkling.tokens` rows (shape bucket capacity); kernels read the live count from the metadata header.

## Packet, span table, token map

- `metadata` (library packet): header three i32 {live tokens, span count, selected outputs}; descriptors at byte 12, six i32 each {length, causal position, token begin, row, source position, selection}; selections, options, policy, identities follow (`lib/vendor/packet.loom`). Token ids: `tokens` buffer, i32, packed token order.
- `span_table` (built by `inkling_prep_spans`): four i32 header {span count, live tokens, 0, 0}; then eight i32 per span {first token, length, read slot lo, read slot hi, write slot lo, write slot hi, position of the first token, row}. Slot words are arena byte offsets of the START of the row recurrent region (rings first, then conv state). Read points at the fork anchor only for the first span after a fork. Capacity: `4 + 8 x span_cap` words, `span_cap <= 256`.
- `token_map`: four i32 per token {span, row, position, index in span}.

## Storage geometry (docs/PACKAGE_PLAN.md section 2)

- Arena: pooled global KV first, then one slot per row (and per pin and fork writer). Pool: 8 planes (global visits 5, 11, ... 47 in order, ordinal g = visit / 6), plane stride `pool x 4096`, page 64 tokens = 262144 B, token entry 4096 B (K F16 [1024] then V F16 [1024]). Region record `{alloc 1, origin 0, planes 8, plane bytes, page bytes 262144}`. Physical page from `origins[row].map[position / 64]` (`lib/vendor/span.loom` helpers).
- Slot (stride 171,608,576 B): control 256 B, recurrent region 171,606,016 B at +256, token view 2048 B, progress 256 B. Recurrent region: local ring of 40 visits (visit j = v - v / 6, entry `(j x 1024 + position % 1024) x 4096`, K then V), then conv state at `167,772,160 + v x 79,872` (3 rows x 6656 F32, row 0 oldest; channels k 0-1023, v 1024-2047, post-attn 2048-4351, post-ffn 4352-6655).
- Origins record per row (5 i64): {read anchor, write slot, base 0, fork frontier, page-map offset}. Geometry header {page tokens 64, map origin, 0, feedback split, recurrent origin, slot stride, slot length, slot count}.
- Allocations: 0 residual (capture buffer, `packet cap x capture_width x 4`), 1 arena, 2 metadata, 3 tables, 4 tokens, 5 outputs. Control state is the library layout (40 B header, 24 B sampling record per row at byte 40, recurrent recipe, upload table); `resize_state` grows it.
- `rows = 4`, `checkpoint_capacity = 4` gives 12 slots (2.0 GiB). KV is `pool x 32768` B.

## Capture route (main debugging tool)

A `kind: sequence` deployment (`INKLING_MODE=sequence`, default of `scripts/serve.sh`, or the card service `capture`) runs the same programs. `POST /v1/sequences` with the prompt token ids returns the captured tensor per span as base64 F32 rows `[tokens][capture_width]`. The tensor comes from `capture_epoch` (reads the residual buffer) and the capture config: `inkling.capture_id`, `inkling.capture_visit`, `inkling.capture_width`. Ids: 1 embd, 2 norm (x0), 11 xn, 16 ab, 20 cur (layer output) at the given visit; 10-18 (x, q, k, v, r, ob, y) are reserved in `layer.loom` comments and are added as the hooks fill them; 100 PR1 mul_mat probe, 101/102 tune probes.

```sh
INKLING_CAPTURE=2,0,0,2304 INKLING_GGUF=<Q8_K_XL gguf> bash scripts/restart.sh gfx1201 18101 30     # id,visit,occurrence,width
ssh box '. ~/loom-serve/.venv-inkC/bin/activate; cd ~/inkling/inkH; flock -s ~/hrx-pr1/gpu.lock python scripts/capture_check.py 18101 ~/inkling/inkB/oracle/dumps/Q8_K_XL_A norm 0 --mul <gguf>:token_embd_norm.weight [--chunk 5]'
```

`capture_check.py` sends the dump's tokens (optionally in appends of `--chunk`), compares by manifest name and occurrence (NMSE, cosine). Oracle `norm*` tensors are unweighted rms, so pass `--mul <gguf>:<weight tensor>`. A different tensor needs a restart (config is compile time); use the `capture` card service or `restart.sh` with a new `INKLING_CAPTURE`.

Verified: `embd` exact (Q8_K_XL, prompt A), `norm` NMSE 4e-15, chunked appends (5 tokens) equal, gfx1201 and gfx1151, direct definition and card.

## Measurements (stub model, Q4_K_XL, context 32768, prefill 512, rows 4, checkpoints 4, page 64)

| | gfx1201 | gfx1151 |
|---|---|---|
| cold prepare (host program) | 58 ms | 95 ms |
| stage compile, 55 shape buckets, 75 kernels (sum over 4 workers) | 2.5 s | 2.3 s |
| same with the PR1 `mul_mat` probe linked (66 kernels) | 8.5 s | 9.3 s |
| request, 1 token, warm median / min | 5.3 / 3.9 ms | 3.7 / 3.5 ms |
| request, 16 tokens, warm median | 6.7 ms | 3.7 ms |

`scripts/jit_time.sh <arch> <port>` repeats this. The compile cost is per distinct kernel specialization (a bucket count of 55 does not multiply it); real kernels will dominate. Re-measure after each area is integrated.

## Integration checklist per area (from the kernel docs and lead notes)

- mix (inkG): conv visit offset is `167772160 + v x 79872` (config `inkling.conv_origin`); the loop bank is written by `conv_res` at ip 12 and 17; message mix of visit X reads `mix_out` of X - 1; oracle norm occurrences on loop layers are ordered (k, v, q) and on global layers (k, v, q, ffn). Span table is the i32 layout above (inkG reads words 0-5).
- attn (inkE): span words 6 and 7 (position, row) exist; the buffer holds `4 + 8 x span_cap` words; size the decode partial buffer and `max_splits` from the shape bucket; per-arch `split_blocks` goes through `docs/TUNE_ROUTE.md`; link `lib/inkling/attn_targets.loom`, `attn_matrix.loom`, `attn_common.loom`.
- dense (inkF): link `lib/inkling/dense_targets.loom` (and its other `dense_*.loom` libraries); set `inkling.dense.gemv.cols` per stage in `prepare` to the largest live row count (1 for the decode bucket); head logits layout `[slot][200064]`; `inkling_head_select` reads plan fields 0, 4, 6; router and controllers use exact kernels. Target resolution under serve is the same mechanism as the tune probe (alias defined in a linked file selects the provider).
- experts (inkD): weight type is an exact launch scalar from `inkling_expert_wtype(layer)` (0 NVFP4, 1 Q8_0, 2 BF16); verify in a serve stage that it folds (not yet verified; fallback one thin export per dtype); pass the live token count as `n` where possible; matrix path from n = 48; `dout` workspace 37.7 MB at n = 512; router input is `norm-S x ffn_norm.weight`.
- chat (inkI): `chat.loom` exports `text_*_v1` (works in the card; direct definitions need plain-name wrappers). Control state in the library layout keeps the sampling table at byte 40 with 24 B per row (not 128): `bind_request` must write there, and `encode_epoch` already copies it into the packet. Tokenizer: the 12 structure tokens must be non-special (`scripts/chat_tokenizer.py`); not applied yet. Vendor `sampling.loom` and `random.loom` from the library `motif/tensor` into `lib/vendor/` (not done).
