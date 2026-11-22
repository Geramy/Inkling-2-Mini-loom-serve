# Inkling 2 Mini numerical oracle

Mainline llama.cpp, CPU only, on the GPU box (`~/inkling/inkB/`). It loads the Unsloth `Mainline_Support` GGUFs and gives reference tensors for layer-by-layer checks of the Loom package.

## Exact versions

| Item | Value |
| --- | --- |
| llama.cpp master (shallow clone) | `418fe97b5f01b11eeb89f30d8a0459433edbf1d2` (sycl: Improve Grouped XMX GEMM ..., 2026-11-20) |
| Fix PR #32704 | NOT in master at clone time. Commit `2a28df0f9bd55c4975b6bffb35229a6020720f13` ("model : inkling2_loop re-share the loop slots after the scale pass"), fetched with `git fetch --depth 50 origin pull/32704/head:inkling2-loop-fix`, cherry-picked on master |
| Oracle base (branch `oracle-base`) | `1393492dd117cf3d7f276cc8e5ed5586ee582e6b` = master + cherry-picked fix |
| Local changes on top (uncommitted) | `oracle/tools/inkling2-debug-names.patch` (only `cb()` names and an `il_dbg` member, no numeric change) and the new tool `examples/oracle-dump` |
| Model files | `~/models/inkling2-mini/Mainline_Support/Inkling-2-Mini-UD-{Q8_K_XL,Q4_K_XL}.gguf` |

Arch in the files: `inkling2_loop`, 24 physical layers, loop core layers [6, 18), 3 passes, so 48 cache slots (slot index 0..5 prelude, 6..17 pass 0, 18..23 postlude, 24..35 pass 1, 36..47 pass 2 where slots 24.. are the physical layers 6..17 again; see `layer_phys()` in `src/models/inkling2-loop.cpp`). Weights are shared between passes; KV and conv state are per slot. `n_embd` 2304, 128 experts, 8 used, vocab 201024 (200058 unpadded), attn-res block size 6, conv kernel 4.

Note: the Q4_K_XL file is mostly Q8_0 (llama.cpp reports ftype Q8_0, 16.4 GiB) and the Q8_K_XL is mostly BF16 (21.3 GiB). 26.07 B parameters.

## Build

```
cd ~/inkling/inkB/src
cmake -S . -B ../build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DGGML_HIP=OFF -DGGML_VULKAN=OFF -DGGML_CUDA=OFF -DGGML_HRX=OFF -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF
cmake --build ../build -j 12 --target llama-cli llama-server llama-perplexity llama-eval-callback llama-tokenize llama-bench llama-oracle-dump
```

Binaries are in `~/inkling/inkB/build/bin/`. To rebuild from scratch: clone, `git fetch origin pull/32704/head:inkling2-loop-fix`, cherry-pick `2a28df0f9` (skip this once #32704 is in master), `git apply oracle/tools/inkling2-debug-names.patch`, copy `oracle/tools/oracle-dump.cpp` to `examples/oracle-dump/` with a CMakeLists like `eval-callback`'s (target `llama-oracle-dump`) and add `add_subdirectory(oracle-dump)` to `examples/CMakeLists.txt`.

## Run commands

Chat (the sanity check, greedy, CPU):

```
llama-cli -m $MD/Inkling-2-Mini-UD-Q8_K_XL.gguf --jinja -ngl 0 -c 4096 --temp 0 -n 64 -t 16 -p "What is the capital of France? Answer briefly." -st
```

Gave a short thinking block and "Paris". Log: `oracle/logs/cli_q8_sanity.txt`.

Perplexity: `llama-perplexity -m <gguf> -f ~/models/wikitext-2-raw/wiki.test.raw -c 512 --chunks N -ngl 0 -t 16`.

Speed: `llama-bench -m <gguf> -p 128 -n 32 -t 16 -ngl 0 -r 2`.

Dumps and continuations: `~/inkling/inkB/make_oracle.sh` (copy in `oracle/tools/`). It runs `llama-oracle-dump` once per prompt. Set `ONLY_Q8=1` to skip the Q4_K_XL continuations. `llama-oracle-dump` takes the usual common args (`-m -t -c -ctk ...`) and env vars:

- `ORACLE_TOKENS`: file with space separated token ids (the whole prompt is one batch, so one forward pass)
- `ORACLE_OUT`: output directory
- `ORACLE_MODE`: `all`, `light` (only `l_out-*`, `result_norm`, `result_output`) or `none` (no dump)
- `ORACLE_GEN`: greedy-decode N tokens after the dump (`gen_ids.txt`, `gen_text.txt`). Dumping is off for the generated tokens.

Runs use `-ctk f32 -ctv f32` (f32 KV cache) so the oracle has no KV rounding. Thread count does not change the result: a `-t 8` run matches the `-t 16` dump with NMSE < 1e-6 on all 1629 tensors.

## Prompts (`oracle/prompts/`)

| Id | Content | Tokens |
| --- | --- | --- |
| A | `A.txt`, "The quick brown fox ..." | 16 |
| B | `B.txt`, a long essay paragraph | 308 |
| C | `C_user.txt` rendered with the chat template (`C_templated.txt`) | 25 |

Token ids are in `X.ids.txt` (made with `llama-tokenize --ids`, no BOS added by the vocab; C by the server `/apply-template` text and `/tokenize` with `parse_special`, `C_tok.json`). The template output for C is `<|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|><|message_user|>...<|end_message|>` and has no generation prompt: the model emits `<|message_model|>` itself (see C continuation).

## File layout

```
oracle/prompts/        A,B,C text and token ids
oracle/dumps/Q8_K_XL_{A,B,C}/   manifest.json + NNNNN_<name>.bin (+ gen_ids.txt, gen_text.txt)
oracle/gen/{Q8,Q4}_K_XL_{A,B,C}/ gen_ids.txt, gen_text.txt (64 greedy tokens, f32 KV)
oracle/logs/           perplexity, bench, sanity chat logs
oracle/tools/          oracle-dump.cpp, make_oracle.sh, inkling2-debug-names.patch
scripts/compare_to_oracle.py
```

Dump sizes: A (all tensors, 1629 files) 162 MB; B (50 files, l_out and logits) 131 MB; C 12 MB. The same tree is on the box at `~/inkling/inkB/oracle`.

### Manifest and file format

`manifest.json` has `tokens`, `meta` (the GGUF scalar metadata) and `tensors`: `{seq, name, base, layer, occurrence, op, dtype, ne, file}`. Files are raw little-endian f32 (or i32 for expert ids), `ne[0]` fastest, so the numpy shape is `ne[::-1]`. Tokens are the columns (`ne[1]` = n_tokens for `[n_embd, n_tokens]`).

`layer` is the cache slot (0..47), not the physical layer; `physical = slot < 24 ? slot : 6 + (slot - 24) % 12`. Pass of a slot: 0 for slot < 24 (core layers 6..17 are pass 0), pass 1 for slots 24..35, pass 2 for 36..47. A name is `<base>-<slot>`; `occurrence` counts repeats of the same name (names can repeat, see below).

### Tensors in the dump (prompt A)

| Base name | What |
| --- | --- |
| `embd` | token embedding rows |
| `norm` (no slot) | token embedding after `tok_norm` |
| `norm-S` | RMS norms in slot S, in order of use: attn input (Q, K, V inputs for full-attention layers, one for SWA), then FFN norm. Use `occurrence` |
| `mix_out` | depth mixing result (attn-res), `mix_out-S` repeats per call; in passes 1 and 2 the call before a layer is tagged with the previous slot |
| `proj_q-S`, `proj_r-S` | Q and relative-position projections |
| `conv_in-S` | input of each short conv: k proj, v proj, attn out proj, FFN out (4 per slot, in that order; `occurrence` 0..3 when names repeat) |
| `conv_out-S` | output of short conv for k and v (the attn out and FFN conv outputs are renamed `attn_out` / `ffn_out`; `attn_out` is renamed `ffn_inp` in the first layer of each residual block, so `attn_out` has 40 entries) |
| `Qcur-S`, `Kcur-S`, `Vcur-S`, `Rcur-S` | after clip/norm (and tau scale for Q, R in full-attention layers) |
| `kq_b-S`, `kqv_out-S` | gathered relative bias with sink; attention output |
| `attn_out-S`, `ffn_inp-S` | attention branch output after post conv; post-attn residual (input of the FFN mix) |
| `ffn_moe_logits-S`, `ffn_moe_probs-S`, `ffn_moe_probs_biased-S`, `ffn_moe_topk-S` (i32 ids), `ffn_moe_weights-S`, `ffn_moe_weights_norm-S`, `ffn_moe_weights_scaled-S` | router |
| `ffn_out-S` | expert FFN output after post conv |
| `l_out-S` | layer output |
| `result_norm`, `result_output` | final norm (last token), logits (200058, last token) |

Not dumped (size): per-expert intermediates, KV caches, masks.

Loop passes 1 and 2 (slots 24..47) have the extra control path (`ctrl_*` mix of messages) inside `norm`/`conv_in` inputs; the loop control tensors are not named separately.

## Comparison script

```
python scripts/compare_to_oracle.py oracle/dumps/Q8_K_XL_A <test_dump_dir> [--profile f32|q8|q4] [--only REGEX] [--skip REGEX] [--allow-missing] [--quiet]
```

Numpy only. A test dump has the same layout (`manifest.json` + `.bin` files, same names and occurrences). Tensors are paired by (name, occurrence). Output per tensor: max abs, mean abs, NMSE (sum err^2 / sum ref^2), cosine, PASS/FAIL. Exit code 1 on any FAIL.

Thresholds (profile, NMSE below, cosine at least, fraction of i32 id rows whose expert set differs at most):

| Profile | NMSE | cos | ids | Use |
| --- | --- | --- | --- | --- |
| `f32` (default) | 1e-6 | 0.999999 | 0 | f32 kernels with the same weights (dequantized or f32 matmul) |
| `q8` | 1e-3 | 0.9995 | 2% | kernels that quantize activations (Q8 dot paths) on the same weights |
| `q4` | 1e-2 | 0.995 | 25% | a different quantization of the model against the Q8_K_XL oracle |

`f32` is the gate from the brief. The `q8` limits are a documented guess based on the observed Q4_K_XL vs Q8_K_XL gap below; tune them with real Loom results. Expert ids are compared per row as sets (order is free). A router near-tie can flip an expert after a small error, so for the quantized profiles compare router logits and layer outputs first and treat ids as a secondary signal.

Self-checks done: oracle vs oracle -> 1629 PASS; Q8 `-t 8` re-run vs `-t 16` oracle -> 1629 PASS (f32). Q4_K_XL vs Q8_K_XL on prompt A (different weights): `l_out` NMSE 3e-4 to 4e-3 (cos >= 0.998), logits NMSE 1.6e-3, cos 0.9995, 6 to 38% of router id rows differ; under `--profile q4` 1579 of 1629 tensors pass (the failures are `conv_in`/`ffn_out`/ids in deep slots). So that comparison is a sanity check of the tool, not a gate.

## Numbers (CPU, 16 threads, 32-thread box, shared with other agents)

Perplexity (wikitext-2 test, `-c 512`):

| Model | 4 chunks | 32 chunks |
| --- | --- | --- |
| Q8_K_XL | 6.5778 +/- 0.519 | 6.4181 +/- 0.175 |
| Q4_K_XL | 6.6090 +/- 0.523 | 6.4284 +/- 0.175 |

Unsloth reports about 7.65 for Q8_K_XL (their setup is not known here: different context length or the full set; our 32 chunk value is 6.42, so the 7.65 was not reproduced with these settings; the running value was still moving between 5.2 and 8.7 across chunks). Q4_K_XL is close to Q8_K_XL (about +0.03 on the same 4 chunks).

Speed (`llama-bench -p 128 -n 32 -t 16 -ngl 0 -r 2`):

| Model | pp128 t/s | tg32 t/s |
| --- | --- | --- |
| Q4_K_XL | 72.63 +/- 2.99 | 18.29 +/- 0.03 |
| Q8_K_XL | 75.35 +/- 0.12 | 14.33 +/- 0.00 |

llama-cli on Q8_K_XL: prompt 45.6 t/s, generation 14.4 t/s (short prompt).

## Greedy continuations (64 tokens, `oracle/gen/`)

Q8_K_XL and Q4_K_XL give the same text for A (the first 64 tokens are an explanation of the sentence); both C continuations start `<|message_model|><|content_thinking|>The user ...` but the wording differs from the first sentence. Tokens: `gen_ids.txt`; text: `gen_text.txt`. Tokens are produced by argmax over the 200058 real logits and generation stops at an EOG token.

## Pitfalls

- PR #32704 is required. Without it passes 1 and 2 of the loop run NVFP4 experts without per-expert scales (perplexity about 130).
- llama.cpp CPU API changed: `llama_process(ctx, LLAMA_PROCESS_TYPE_DECODE, batch)` replaces `llama_decode`.
- `llama-cli` without `-st` stays interactive; with `--jinja` it applies the chat template and the model thinks first (`[Start thinking]`). `-no-cnv` and `--no-warmup` are not accepted by `llama-oracle-dump` (warmup is off in code).
- cb names can repeat and are renamed by later `cb()` calls on the same tensor (see the table). Rely on (name, occurrence), not on position in the file.
- The oracle runs with f32 KV cache. Use the same KV type when comparing (not measured how much f16 KV moves the result).
- Do not run the oracle while GPU jobs hold all CPU cores: timing numbers above were taken while other agents ran, so use them as rough values.
- Perplexity on 4 chunks has a large error bar (+/- 0.5).
