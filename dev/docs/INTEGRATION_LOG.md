# Integration log (inkJ, continued by inkK)

Bead llama-cpp-tg3.10. Box dir `~/inkling/inkJ`, venv `~/loom-serve/.venv-inkJ`, laptop dir `C:\Development\inkling-loom-serve`. No commits, nothing pushed. Newest state first.

## First measurements (inkK, 2026-11-21; not tuned)

Method: GPU lock `-x`, warmed, no tuning; box shared. Q4_K_XL decode split: `scripts/jsplit.sh` (sequence route, one token appended to a session after prompt A, `INKLING_STOP` = number of hooks kept per visit, 30 steps, minimum of the step times, ms; the first row also contains the sequence service round trip of 1.0 to 1.5 ms). Floors from docs/ARCHITECTURE.md section 12 (Q4_K_XL, 4011 MB per token: 6.47 ms on the R9700, 17.44 ms on the 8060S).

| piece (hooks) | gfx1201 ms | gfx1151 ms | floor gfx1201 / gfx1151 ms |
|---|---|---|---|
| embed + head (final mix, lm_head 492 MB, sampler) | 1.74 | 3.06 | 0.79 / 2.14 (lm_head) |
| mix_attn (48 visits) | 0.48 | 0.40 | |
| message mix + controller QKV (24 visits) | 0.55 | 1.39 | controllers + msg 227 MB: 0.37 / 0.99 |
| norm_in + proj (q, k, v, rel, gate, sink) | 2.43 | 5.89 | |
| attn_prep + attn (context 16) | 0.71 | 1.01 | |
| oproj | 0.59 | 2.03 | proj + oproj 1374 MB: 2.2 / 5.97 |
| post_attn | 0.21 | about 0 | |
| mix_ffn | 0.64 | 1.04 | |
| controller FFN (with message mix again) | 0.57 | 0.44 | |
| norm_ffn + router | 0.26 | 0.64 | router 57 MB: 0.09 / 0.25 |
| experts (NVFP4 and dense arm) | 3.65 | 10.37 | 1847 MB: 2.98 / 8.03 |
| post_ffn | 0.27 | 0.53 | |
| total decode step | 12.10 | 26.75 | 6.47 / 17.44 |

So decode is 1.9 times the floor on the R9700 and 1.5 times on the 8060S. Largest gaps (ms over the floor, R9700): attention projections +0.8 (GEMV efficiency), experts +0.7, mix/controllers/convs/post about 3 ms of small dependent launches (about 480 launches of 3 to 8 us), embed + head +0.95. M8 targets in this order: merge the small mix/norm/conv launches, the head (sampler and lm_head launch chain), `proj`.

### Chat route numbers (card route, presets `interactive-q4`/`interactive-q8`, `scripts/bench_card.sh` + `scripts/bench_chat.py`, GPU lock `-x`, 2 warm-ups, median of 5 (prefill) or 4 (decode, 64 streamed tokens, temperature 0))

Prefill: wall time of a chat request with `max_tokens 1` (includes tokenization, template, one decode step; prompts above 512 tokens run as 512 token chunks). Decode: median gap between streamed tokens; the 8k context is a prompt of 8005 tokens.

| | prompt 121 | prompt 520 | prompt 2053 | decode ctx 16 | decode ctx 8005 | floor decode (ARCHITECTURE 12) |
|---|---|---|---|---|---|---|
| gfx1201 Q4_K_XL | 97.7 ms, 1239 tok/s | 236 ms, 2202 tok/s | 910 ms, 2257 tok/s | 86.6 tok/s (11.55 ms) | 74.9 tok/s (13.34 ms) | 154.6 tok/s |
| gfx1201 Q8_K_XL | 100.8 ms, 1201 tok/s | 238 ms, 2186 tok/s | 930 ms, 2208 tok/s | 69.7 tok/s (14.36 ms) | 61.9 tok/s (16.16 ms) | 107.3 tok/s |
| gfx1151 Q4_K_XL | 234 ms, 516 tok/s | 578 ms, 900 tok/s | 2259 ms, 909 tok/s | 38.9 tok/s (25.68 ms) | 33.3 tok/s (30.05 ms) | 57.3 tok/s |
| gfx1151 Q8_K_XL | 259 ms, 467 tok/s | 625 ms, 833 tok/s | 2397 ms, 857 tok/s | 29.3 tok/s (34.09 ms) | 26.1 tok/s (38.31 ms) | 39.8 tok/s |

Decode reaches 56 percent (R9700, Q4) to 74 percent (8060S, Q8) of the bandwidth floor at short context. The 16 token prompt alone takes 38 ms (R9700 Q4), 88 ms (8060S Q4). Prefill is not near any floor yet (4.5 TFLOP per 512 token chunk: 2200 tok/s is about 10 TFLOP/s on the R9700); the stage composition is untuned.

Memory (VRAM of the R9700, GTT of the 8060S, after the first request, and after the 2k and 8k prompts): Q4_K_XL 19.1 GB then 20.7 GB; Q8_K_XL 24.1 GB then 25.7 GB (rows 4, context 32768, checkpoints 4; weights 17.6 and 22.9 GB). Two servers on the R9700 do not fit.

Start up (kernel cache warm): prepare 1.5 s, registration ready after 2.0 s (R9700) and 3.1 to 3.6 s (8060S), the first request activates the weights (3.2 s Q4 and 4.1 s Q8 on the R9700, 1.4 and 1.7 s on the 8060S); later shape buckets show no first use delay (second request equals the first). A first compile of the whole package (after code changes, empty cache) took 60 to 100 s wall (stage compile time summed over the 4 workers: 75 to 155 s in `/v1/status`); a changed source file alone did not trigger it (the cache is keyed by kernel content), so no cold compile was timed separately.

## M6 done (inkK, 2026-11-21): head, sampler, chat, tokenizer

Checks (Q8_K_XL, capture route `INKLING_CAPTURE=31,0,0,200064`, `scripts/head_check.py`; both GPUs equal to the third digit): `result_output` of the last token A: NMSE 7.0e-4, KL 1.1e-3, top-5 5/5, top-20 20/20; B (308 tokens): NMSE 3.7e-2 (cos 0.995), KL 5.7e-3; C: NMSE 1.9e-5, KL 4.6e-11. `result_norm` (A) NMSE 3.9e-3. Teacher forced greedy (prompt + oracle continuation in chunks of 128, argmax of every row against the next oracle token): A 63/64 (mismatch at 12), B 64/64, C 63/64 (mismatch at 57); the mismatches are near ties. Chat through HTTP (`scripts/jchat.sh <arch> <port> <q4|q8>`, card route, prompt C, temperature 0, 64 tokens, stream and not stream): all four combinations (gfx1201 and gfx1151, Q4_K_XL and Q8_K_XL) answer; Q8 equals `oracle/gen/Q8_K_XL_C` for the first 55 tokens (then a near tie: "toward observer. Two sentences" instead of "in all directions"; gfx1151 Q8 equals the oracle for all 64 tokens, gfx1201 Q8 diverges at the tie as in the teacher forced run); Q4_K_XL text on both GPUs equals the Q8 oracle text for all 64 tokens and differs from `oracle/gen/Q4_K_XL_C` from the 4th token ("asks for an explanation" instead of "wants a two-sentence explanation"; the Q4 oracle quantizes activations, we keep F32).

Design:
- `embed`: the real `inkling_embed_{q8_0,bf16}`; `dispatch/temp_embed.loom` is deleted (capture id 1 `embd` is no longer filled, id 2 `norm` works).
- `head` (generated, `head_hook` in `gen_hooks.py`; the command got a `%residual` buffer for the captures): `mix_states` nb 4 over all tokens (mode 0, `output_res_query`) into `x`, `inkling_head_select` (rewritten for the library packet: header word 2 and the selection list; `all_rows` 1 selects every live token when `inkling.capture_id` is 30 or 31), `inkling_head_norm` into `xn`, lm_head GEMV (rows <= 8) or GEMM with act 2, cap 30, `d_stride` 200064, then `inkling_sample_rows`. The logits scratch is `span_cap` rows (800 KB each), or `cap` rows in capture mode. Capture 30 = `result_norm` rows, 31 = logits rows (width 200064).
- `control/chat_sampler.loom`: `inkling_sample_rows(rows)[rows, logit_stride, epoch, logits, outputs]` reads the packet identity and policy tables (like the library Qwen kernel); `package/test/chat/test_sampler.py` still uses the old signature.
- `control/chat.loom` `bind_request` writes at byte 40 (was 128; docs/CHAT_POLICY.md fixed). `control/identity.loom` gives `text_model_name_v1` (the card route needs it). `lib/vendor/{sampling,random}.loom` vendored (VENDORED.md). `package/tokenizer.json` = GGUF tokenizer with the 12 structure tokens non-special (`scripts/gen_tokenizer.py` then `scripts/chat_tokenizer.py`); the card already points at it. `card.json` services `chat` (text v1) and `capture` (sequence v1) list the sampler, vendor and identity files.
- Chat route start: `scripts/serve_card.sh <arch> <q4|q8> [port] [preset]` on the box, `scripts/jcard.sh` / `scripts/jchat.sh` from the laptop. Only one server per GPU: two Q8 servers do not fit in the R9700 (RESOURCE_EXHAUSTED, the model becomes unavailable until restart).

## M5 done (inkK, 2026-11-21): loop passes, controllers, coda

All 48 visits run and match the oracle at the drift floor (Q8_K_XL, prompt A 16 tokens, gfx1201; capture-all per visit): `l_out` NMSE 4.6e-4 (visit 17, pass 0), 2.0e-4 (24), 1.4e-3 (29), 1.0e-3 (36), 2.1e-3 (47), coda 18: 2.2e-3, 20: 1.6e-3, 21: 1.5e-3, 23: 5.6e-4; `y` of the dense expert arm (20, 21, 23) 5.3e-3, 2.9e-3, 2.7e-3. Prompt B at 47: 2.2e-3, C: 7.6e-4. The inputs of visit 17 (pass 0, no controllers) are already at 1e-3: the error is the oracle's Q8_0 activation rounding and routing near ties accumulating over layers, not one bug (a wrong bank state or prelude copy would give errors of order 1e-1). Do not gate deep visits at 2e-4; the real check is `result_output` (M6). Chunk independence at visit 47: B 128+128+52 equals the single chunk (2.212e-3 both); A 5+5+6 and 1+15 differ per token from the second chunk on (up to 5e-3 per token, whole 5e-4 and 6e-5 between runs): the F16 history path versus in-flight rounding, amplified by 48 layers; not tightened.

Design (all in `scripts/gen_hooks.py`, no persistent message buffer):
- `mix_msg` stays a no-op. `ctrl_qkv` and `ctrl_ffn` run the message mix themselves (`inkling_mix_states` nb 1, nq 4, mode 0, scratch `buffer.alloca` for the 4 message columns and `rinv_m`), then `ctrl_down`, `ctrl_z`, `ctrl_apply` (nc 3 or 1; FFN weight set repeated). `ctrl_ffn` recomputes the message mix (4 columns, 6 us decode, 50 us at 512 tokens; M8: keep it).
- On controlled visits (layers 6..17, pass >= 1) `mix_attn` and `mix_ffn` write the mix to `xn` (instead of `x`); the controller reads it from `xn` and writes the controlled columns to `x` (Q, K, V = columns 0, 1, 2; FFN column 0). `norm_in` then norms `x` columns 0..2 with `attn_norm` (local; global uses q/k/v_inp_norm as before) into `xn` columns 0..2; `proj` reads `xn` columns 0, 1, 2 for q, k, v on controlled visits. `mix_norm` is not used. Capture id 10 (`x`) is now taken after `ctrl_qkv`.
- Prelude copy: `blocks` slot 7. `post_ffn` of layer 5 copies `cur` to slot 7 (`mix_states` nb 0 is a copy), `mix_attn` of layer 6 with pass >= 1 restores `cur` from slot 7 first.
- `kernel/mix/mix_states.loom` got three launch scalars `blocks_off`, `cur_off`, `out_off` (elements, after `sel_on`; `package/test/mix` still calls the old ABI): the message mix reads bank set `pass - 1` (state 0 as the block, state 1 as `cur`), the copy uses slot 7. Logged in KERNELS_mix.md.
- `scripts/capture_all.py`: visits >= 24 use norm occurrences k, v, q, ffn (xn = occurrence 2, xn_ffn = 3), no `x` compare.
- Scripts now use the box dir `~/inkling/inkK` and venv `.venv-inkK`.

## State at the end of M6 (inkK, 2026-11-21)

M5 and M6 are done: all 48 visits, loop passes, controllers, head, sampler, chat service and tokenizer work on gfx1201 and gfx1151 for both checkpoints, and the first measurements are below. The M4 text further down is the history. What remains: per-token tuning (M8: merge small launches, head chain, proj, experts), prefill above 512 tokens per stage (needs the experts kernels to loop 512 token slices), a real cold compile timing, `test_sampler.py` for the new sampler signature, and the `package/test/mix` re-run.

## State at the end of M4 (2026-11-21)

M2, M3 and M4 are done for Q8_K_XL on gfx1201 and gfx1151 (prompts A, B, C and the 1332 token prompt; tables below). Real hooks: `mix_attn`, `norm_in`, `proj`, `attn_prep`, `attn`, `oproj`, `post_attn`, `mix_ffn`, `norm_ffn`, `router`, `experts`, `post_ffn`. Still no-ops: `mix_msg`, `ctrl_qkv`, `ctrl_ffn` (loop controllers, M5) and `head` (M6). `embed` is still the temporary kernel of `temp_embed.loom`. The schedule already contains all 48 visits and compiles for every bucket (78 buckets at prefill 1536, 55 at 512), so visits 6..47 run today without controllers: visits 6..11 (loop pass 0) are checked against the oracle, visits >= 12 and passes 1 and 2 are not.

## Hard limits found

- `inkling_experts_routes` needs `n <= 512`. A stage with more than 512 tokens (tested with `INKLING_PREFILL=1536`) hangs the GPU for minutes (`gpu_busy_percent` 100, the next server start fails with `DATA_LOSS` until the GPU recovers, about 2 minutes). Keep `prefill_capacity <= 512` (model.json, `INKLING_PREFILL`) until the experts hook loops over 512 token slices (needs origin offsets in the kernels) or the kernel limit is lifted.
- Attention kernels need `span_cap <= 256`: `prepare.loom` clamps `inkling.spans` per stage to 256 (the library shape catalog has buckets with 512 spans).
- Model weights through `command.parameter` need exact view footprints (finding 1 below).

## Results (Q8_K_XL, NMSE against the oracle, capture route, `capture_all.py`)

Prompt A, 16 tokens, one chunk. Columns: kqv_out times gate (`ab`), o-proj output (`ob`), expert output (`y`), layer output (`l_out`). gfx1151 numbers differ from gfx1201 in the third digit only.

| visit | layer | ab | ob | y | l_out | gfx1201 | gfx1151 |
|---|---|---|---|---|---|---|---|
| 0 | 0 local | 2.1e-6 | 4.6e-6 | 1.1e-4 | 4.5e-5 | pass | pass |
| 1 | 1 local | 2.0e-5 | 2.0e-5 | 1.1e-4 | 3.0e-5 | pass | pass |
| 2 | 2 local | 1.9e-5 | 3.1e-5 | 3.6e-4 | 9.7e-5 | pass | pass |
| 3 | 3 local | 9.7e-5 | 1.4e-4 | 2.6e-4 | 7.2e-5 | pass (xn, ob, xn_ffn 1.0e-4 to 1.4e-4, just above the 1e-4 gate) | same |
| 4 | 4 local | 4.5e-5 | 5.8e-5 | 1.7e-4 | 5.2e-5 | pass | pass |
| 5 | 5 global | 2.7e-5 .. 4.7e-5 | 5.2e-5 | 4.5e-4 | 7.0e-5 | pass | pass |
| 6 | 6 local (loop pass 0) | 6.0e-5 | 7.2e-5 | 1.8e-4 | 3.3e-5 | pass | not run |

Visit 0 detail (gfx1201 / gfx1151 equal): xn 6e-15, q_raw 2.0e-6, k_raw 1.6e-6, v_raw 1.1e-6, r_raw 5.8e-7, gate 8.7e-9, sink 2.8e-14, q 1.9e-6, k 1.8e-6, v 8.7e-7, r 3.1e-7, ffn_inp 1.8e-6, xn_ffn 2.6e-6, router logits 1.3e-7.
`y` is limited by the oracle (it quantizes the expert activations to Q8_0; the expert kernels are closer to exact math, see KERNELS_experts.md): the gate for `y` is 5 x 2e-4.

Prompts B (308 tokens) and C (25), `l_out` of slot 5 (all six layers in the chain): gfx1201 B 9.3e-5, C 1.1e-4; gfx1151 B 1.1e-4, C 1.1e-4.

Chunk independence (results equal to the single chunk run, same NMSE to 3 digits): A at visit 5 with chunks 5+5+6, 1+15 and 16 x 1 (gfx1201); B at slot 5 with 128+128+52, 256+52, 300 then 8 single tokens (gfx1201), 128+128+52 (gfx1151); the 1332 token prompt (oracle `Q8_K_XL_L1`, ring wrap) at slots 0, 5, 6, 11 with chunks 444 x 3, 512+512+308 and 512+512+276 then 32 single tokens (gfx1201 and gfx1151), identical numbers per chunking up to the F16 decode path rounding (l_out 6.4e-5 versus 6.8e-5). A single chunk of 1332 tokens was not run (hard limit above); the oracle itself is bitwise chunk invariant.

1332 token prompt, `kqv_out` times gate (`ab`), gfx1201 (gfx1151 within 15%): slot 0 local ring wrap 1.5e-6, slot 5 global 5.1e-5, slot 6 local 6.0e-5, slot 11 global 2.3e-3. `y` of slot 6 is 2.8e-3: exactly 50 of 1332 tokens select a different top-8 expert set than the oracle (router logits agree to 4e-6, sets differ at near ties; those 50 tokens carry 59 percent of the error, the median token is at 1.2e-4). The drift of the residual stream through the routing flips of layers 6 to 10 explains slot 11 (median per-token NMSE of `ab` 6.6e-4). The attention kernels themselves are exact on the oracle inputs (KERNELS_attn.md, slots 6 and 11 1e-8 to 2e-7). Do not tighten the gates for deep slots of long prompts.

Matrix-op selection under loom-serve (Step 0): measured with `scripts/jtimed.sh` (prompt B, 308 tokens in the 512 bucket, 48 visits, `INKLING_STOP` differences, GPU lock `-x`, median of 10 after 2 warm-ups):

| piece (hooks added) | gfx1201 | gfx1151 | expected with matrix ops (from the kernel docs) | scalar fallback |
|---|---|---|---|---|
| proj (6 GEMMs, 0.59 TFLOP) | 13.9 ms (42 TFLOP/s) | 32.8 ms (18 TFLOP/s) | 12 ms / 35 ms | 5 to 15 times slower |
| attention (tile, merge, kv_append) | 7.2 ms | 13.8 ms | about 10 / 15 ms | |
| experts (matrix path) | 91 ms | 265 ms | 91 / 260 ms | portable path 410 / 940 ms |

So the dense, attention and expert matrix-op providers are selected on both GPUs (`dense_targets.loom`, `attn_targets.loom`, `experts_wmma.loom` aliases are linked, `dispatch/targets.loom` keeps only the tune-route aliases). Full 308 token prefill (all hooks up to experts) takes 131 ms on gfx1201 and 365 ms on gfx1151 (not tuned).

## Files (all generated parts come from `scripts/gen_hooks.py`; edit the generator, not the output)

- `scripts/gen_hooks.py` writes `package/dispatch/{hooks_dense,hooks_attn,hooks_mix,hooks_experts,layer}.loom` and `package/dispatch/capture_layout.json`. Run it with python3 (the laptop has python only in WSL: `wsl python3 /mnt/c/Development/inkling-loom-serve/scripts/gen_hooks.py`, set `MSYS_NO_PATHCONV=1` in Git Bash).
- `scripts/gen_kernel_decls.py` writes `package/dispatch/kernel_decls.loom` (one `kernel.decl` per `kernel.def`); the generator copies the needed lines into each hook file (a cross-file kernel launch needs the decl in the file).
- Source lists: `scripts/serve.sh` `SOURCES` and `package/card.json` `libraries` (both service entries). The four stub files are deleted; `stub_debug.loom` stays.
- Scripts: `scripts/jrun.sh <arch> <port> all,<visit> [Q8|Q4] [wait]` (sync, restart, diagnostics), `scripts/jcheck.sh <port> <A|B|C> <visit> [flags]` (runs `capture_all.py` under `flock -s`), `scripts/jvisits.sh` (loop over visits), `scripts/jlink.sh` (link check of every kernel), `scripts/jstage.sh <tokens>` (link check of the model command for one bucket). Env of jrun: `INKLING_STOP=N` runs only the first N hooks of each visit (bisect aid, hook order in `HOOKS` of the generator), `INKLING_PREFILL`, `INKLING_ROWS`, `INKLING_CONTEXT`.
- `restart.sh` now passes `INKLING_ROOT`/`LOOM_VENV` (box dir and inkJ venv) and kills the old server by its pid file (TERM, then KILL after 8 s).

## Capture-all mode

`INKLING_CAPTURE=99,<visit>,0,<total>` (jrun: `all,<visit>`, total from `package/dispatch/capture_layout.json`) copies every shared tensor of the visit into column ranges of the residual buffer (table in `capture_layout.json`: x, xn, q_raw, k_raw, v_raw, r_raw, gate, sink, q, k, v, r, ab, ob, ffn_inp, xn_ffn, logits, y, l_out). `scripts/capture_all.py <port> <dump> <visit> --gguf <Q8 gguf> [--chunk N | --chunks a,b,c]` compares every item with the oracle of cache slot = visit (layers 0..5 of prompts A; the long dumps work for the items they contain). `ab` is compared as oracle `kqv_out` times sigmoid of the captured gate (the oracle `kqv_out` is the attention output BEFORE the gate; our `ab` is after the gate, the kernels apply it). Tolerances: NMSE 1e-4 (y 5e-4, l_out 2e-4); expert outputs are limited by the oracle (it quantizes activations to Q8_0).

## Buffer aliasing in the hooks (the shared buffers are re-used for kernel-private layouts)

Kernels take buffers at origin 0 with fixed per-tensor shapes, so these hooks use free shared buffers as scratch (live ranges checked):

| Tensor | Buffer | Written by | Read by |
|---|---|---|---|
| sink `[cap][24]` | `small` | proj | attn tile, merge |
| dyn conv k `[cap][32]` | `ob` | proj (global) | attn_prep |
| dyn conv v `[cap][32]` | `y` | proj (global) | attn_prep |
| k_act F16 `[cap][1024]` | `xn` | attn_prep | attn tile, kv_append |
| v_act F16 `[cap][1024]` | `x` | attn_prep | attn tile, kv_append |
| router logits `[cap][128]` | `small` | router | (capture only) |
| router ids i32 `[cap][8]` | `ab` | router | experts routes |
| router weights `[cap][8]` | `ob` | router | experts combine |
| router input F16 `[cap][2304]` | `gate` | router | experts gate_up |
| tasks, act, dout, partial | per-hook `buffer.alloca` | | |

`q`, `r` are clipped in place by attn_prep. Layer 0 (visit 0 and, later, all `nb = 0` visits): mix_attn does nothing and norm_in reads `cur` instead of `x`. `bank` is now 4 x 2304 words per token (two sets of two states for the loop message, see KERNELS_mix.md).

## Findings and fixes (read before touching kernels or hooks)

1. Every `command.parameter` view must have the exact byte footprint of the tensor (a 16-byte view made the first launch fault with DATA_LOSS; the base buffer is not resident beyond the view). The generator has a table of view types (`WV`), Q8_0/BF16 dense types are selected with `scf.if` on `inkling.dense_type`, experts use exact sizes per layer type. `command.parameter.base` gives the weight buffer at origin 0.
2. Backend bug (not ours to fix): `BACKEND/044 scheduler stopped ... s_cselect_b32` for `inkling_mix_states` (specialised launch scalars) and `inkling_attn_merge`. Workarounds in kernel code: `inkling_mix_states` no longer selects a buffer (new `inkling_mix_load`: conditional load with two views, KERNELS_mix.md), `inkling_attn_merge` computes `lo_shift` with compare and select instead of `index.max`/`index.sub` (KERNELS_attn.md).
3. `pure` functions called across files need `public`; all `func.def` in `lib/inkling` and `kernel/*` are now `public`. `template.apply` cannot be used in a launch configuration or a `pure` function: the dense GEMV kernels got a third workload value `%per_unit` (the template result of `@inkling.dense.gemv.xs_wg_per_unit`, 4 on gfx11, 2 otherwise) that the hook passes (KERNELS_dense.md).
4. The oracle `kqv_out` is the attention output before the gate multiply.
5. Capture bounds: `capture.loom` needs `index.assume` bounds on the loop index (SUBRANGE/024).
6. Hooks are `command.def public target(@inkling_execution)` (templates resolve under that target).
7. A catalog bucket with more than 256 spans violated the attention kernel contract (TARGET/059); fixed in `prepare.loom`.
8. After a long hang of one GPU the next server start can fail with `DATA_LOSS` ("compute code publication"); wait about two minutes and start again. Always run clients with `timeout` (the scripts do).

## Commands (laptop, Git Bash; python only through WSL)

```sh
# one visit, all captured tensors against the oracle (restart per visit, about 3 minutes per GPU)
bash scripts/jvisits.sh gfx1201 19101 A 16 "0 1 2 3 4 5"      # arch port prompt prefill visits
bash scripts/jvisits.sh gfx1151 19151 A 16 "0 1 2 3 4 5"
# server for one visit with prefill 512 (needed for B and C), then check
INKLING_PREFILL=512 bash scripts/jrun.sh gfx1201 19101 all,5 Q8 600 ; bash scripts/jcheck.sh 19101 B 5 [--chunks 128,128,52]
# 1332 token prompt, slots 0 5 6 11, three chunkings each
bash scripts/jlong.sh gfx1201 19101 "0 5 6 11"
# timing of a group of hooks (INKLING_STOP=N keeps the first N hooks of every visit)
bash scripts/jtimed.sh gfx1201 19101 5      # stop count, prompt B, exclusive GPU lock
# link check of all kernels / the model command of one bucket
bash scripts/jlink.sh ; ssh box '. ~/loom-serve/.venv-inkJ/bin/activate; bash ~/inkling/inkJ/scripts/jstage.sh 1'
# regenerate the hooks after editing scripts/gen_hooks.py
MSYS_NO_PATHCONV=1 wsl python3 /mnt/c/Development/inkling-loom-serve/scripts/gen_hooks.py
```

Never start two jobs for the same arch and port (they restart each other's server; seen once, results discarded). `jrun.sh` waits until the server is ready or failed (polling, max 600 s by default). GPU runs of the scripts use `flock -s`, timed runs `flock -x`.

## What inkK must do next

M5 (loop passes and coda, `l_out` of slots 6..47, `l_out-47`):
1. Implement the hooks `mix_msg`, `ctrl_qkv`, `ctrl_ffn` in `scripts/gen_hooks.py` (`mix_file`) with `inkling_mix_states` (message mix: `nb 1`, `nq 4`, mode 0, blocks = bank set, cur = bank + cap*E) and the controller kernels `ctrl_down`, `ctrl_z`, `ctrl_apply`, `mix_norm` (ABI in KERNELS_mix.md). Per-stage config (`ctrl_chunk`, `ctrl_tokens`, `ctrl_bf16`) is already set in `prepare.loom` (`config_kernels`). The `rinv` arrays go to `small` as `[4][cap]` (already what mix_attn/mix_ffn write, mode 0; the live ranges in the aliasing table are compatible: rinv is consumed by `ctrl_z` before `proj` overwrites `small` with the sink). The weights are BF16 for Q8_K_XL and F32 for Q4_K_XL (`inkling.mix.ctrl_bf16`); declare exact `command.parameter` footprints (`loop_*_ctrl_down` 64 x 4608, `forget`/`gain` 2304 x 64, `loop_msg_query` 4 x 2304 F32). For local loop layers the three controllers read different inputs, so `norm_in` and `proj` must read `xn` columns 0..2 on controlled visits (today `norm_in` uses three columns only on global layers, and `proj` uses column 0 for q, r, gate, sink; check section 6.2 of ARCHITECTURE.md and the oracle norm order k, v, q).
2. Passes 1 and 2 restart from the prelude output (visit 5): keep a copy of `cur` after visit 5 and `blocks[0]` (x0); `blocks` has 8 slots and only 4 are used, so slot 7 can hold the copy (needs a small copy kernel or an `inkling_conv_res` style launch; none exists yet). Restore `cur` and reset the block list at the start of passes 1 and 2 (`model.loom`). The bank (4 x 2304 words per token) is written by `post_attn` at layer 12 and `post_ffn` at layer 17 of passes 0 and 1 (already in the hooks via `bank_on`, `bank_origin`); the message mix of pass p+1 reads set p.
3. `inkling_visit_index` gives the visit of a layer in a pass; conv state, KV ring and pool offsets are computed from `%visit` in the hooks (`conv_origin`, `kv_origin`); only visits 0..11 were exercised. Check visits 12..47 with `capture_all.py` (extend its oracle name mapping: loop layers have three `norm` occurrences ordered k, v, q before the ffn one, `mix_out` occurrence 2 is the message mix, `l_out` of slot X is oracle `l_out-X`).
4. Layers 20..23 use dense experts (wtype 1 Q8_0 for the Q4 file, 2 BF16 for the Q8 file): first runtime use of the dense arm of the `experts` hook; the NVFP4 matrix path is exercised, the dense matrix path (Q8_0, BF16 `gate_up_wmma`) is not. Expect to compare `y` of slots 20..23 (experts doc: 1.8e-3 .. 2.4e-3 on BF16).
5. Do not raise `prefill_capacity` above 512 (hard limit above).

M6 (head, sampler, chat, tokenizer, measurements):
1. `hooks_dense.loom` `inkling_hook_head` is a no-op: wire `mix_states` with `sel_on 1` and `output_res_query` (mode 1 with `output_norm`), `inkling_head_select`, `inkling_head_norm`, the lm_head GEMV/GEMM (`inkling_gemv_*` / `inkling_gemm_*` with `act 2`, `cap 30`, `d_stride 200064`) and the sampler from the vendored library (not vendored yet: `sampling.loom`, `random.loom` from `motif/tensor` into `lib/vendor/`); outputs go to the `outputs` buffer; the sampling record is in the control state at byte 40 (24 B per row), see DISPATCH.md and CHAT_POLICY.md. The head GEMV kernels take the extra workload value `%per_unit` (template `@inkling.dense.gemv.xs_wg_per_unit`, see `linear()` in the generator).
2. Replace `temp_embed.loom` by `inkling_embed_*` of `kernel/dense/embed.loom` if wanted (the temporary kernel is correct).
3. Tokenizer patch (12 structure tokens non-special, `scripts/chat_tokenizer.py`), chat control (`control/chat*.loom`), `INKLING_MODE=text` serve path, and `card.json` (library lists were updated for both services but the card was not run; `python -c json.load` passes).
4. Measurements: decode (cap 1 bucket) per token against floors, prefill 512; use `jtimed.sh`-style interleaved GPU-locked runs. First numbers above (prefill 308: proj, attention, experts).
5. Known performance gaps for M8, not touched: experts n is the bucket capacity (live rows are computed for the whole bucket), `gemm_wide_q8_0` is not selected, decode partial buffer sized from the context, `norm_ffn`/`mix` run as separate launches.
