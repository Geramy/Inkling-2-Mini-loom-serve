# Inkling 2 Mini inference architecture (for the Loom package)

Status legend used in this file:
- VERIFIED: read from the GGUF files on the box (metadata, tensor inventory, bytes) or from llama.cpp source in C:\Development\llama.cpp.
- INFERRED: derived from VERIFIED facts by arithmetic or by analogy; low risk.
- UNVERIFIED: public description only, or my reconstruction. Must be checked against a reference before it is coded.

## 0. Provenance and the one big caveat

- The GGUFs (`Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf`, `...Q8_K_XL.gguf`) have `general.architecture = inkling2_loop`, `block_count = 24`.
- The llama.cpp tree on the laptop (branch hrx-llama-tinystories-15m, `src/models/inkling2.cpp`) implements arch `inkling2` (the Inkling 2 Small layout: no loop). It does NOT contain `inkling2_loop` (that is llama.cpp PR #32694, fixed by PR #32704; the README in Mainline_Support says so). `git log --all` and a grep of the box found no copy.
- Therefore: everything about a single layer visit (attention, convs, attention-residual mixing, router, SiTU experts, output head, tokenizer, chat template) is VERIFIED against `inkling2.cpp` and the loop GGUF tensor shapes (all shapes match the formulas below). The cross-pass message machinery (`loop_*` tensors, section 6) is NOT in any code I could read. Section 6 gives the tensor shapes (VERIFIED) and a reconstruction from public descriptions (UNVERIFIED). A network fetch of the PR diff was denied in my session; see Open Questions. UPDATE: the loop arch source (llama.cpp #32694 + #32704 + 1393492) has since been read; section 6 is now verified, see section 16.
- Notation: ggml shapes are written `{ne0, ne1, ...}` with ne0 contiguous in memory. A weight `{in, out}` is `out` rows of `in` elements; `y = W x` means `y[o] = sum_i W[o][i] x[i]`. Tensor names: `blk.N.<name>.weight` unless noted.

## 1. Hyperparameters (VERIFIED, from GGUF metadata, identical in Q4 and Q8 files)

| Key | Value | Meaning |
|---|---|---|
| block_count | 24 | physical layers (weights) |
| loop.start / loop.end / loop.count | 6 / 18 / 3 | physical layers 6..17 (12 layers) run 3 times; 48 layer visits per token |
| loop.gate_rank | 64 | rank of message controllers |
| loop.varying | false (Q8 file only) | fixed pass count |
| embedding_length (n_embd) | 2304 | |
| attention.head_count | 24 for all layers | n_head |
| attention.head_count_kv | 8 for all layers | n_head_kv, GQA group = 3 |
| attention.key_length / value_length (+ _swa) | 128 / 128 (+ 128/128) | head dim, same for local and global |
| rope.dimension_count | 0 | NO RoPE anywhere |
| attention.layer_norm_rms_epsilon | 1e-6 | all RMS norms |
| attention.sliding_window | 1024 | local window |
| attention.sliding_window_pattern | per physical layer, True = local: layers 0-4 L, 5 G, 6-10 L, 11 G, 12-16 L, 17 G, 18-22 L, 23 G | global layers: 5, 11, 17, 23 |
| attention.relative_length (n_relative) | 16 | relative-bias channels per head |
| attention.relative_extent | 1024 | distance table length (both local and global) |
| attention.qk_clip | 22.627416610717773 (= 16*sqrt(2)) | L2 clip threshold for q and k |
| attention.relative_clip | 8.0 | L2 clip threshold for r |
| attention.log_scale_floor | 8192 | |
| attention.log_scale_alpha | 0.1 | |
| attn_res.block_size | 6 | attention-residual block length in layer visits |
| shortconv.l_cache (K) | 4 | conv kernel size |
| expert_count / expert_used_count | 128 / 8 | |
| expert_feed_forward_length (n_ff_exp) | 1152 | also written as feed_forward_length |
| expert_weights_scale | 8.0 | |
| activation.situ_beta | 4.0 | |
| activation.situ_linear_beta | 25.0 | |
| final_logit_softcapping | 30.0 | |
| vocab_size (padded) / vocab_size_unpadded | 201024 / 200058 | embedding and lm_head have 201024 rows; logits are cut to 200058 |
| context_length | 1048576 | |
| general.sampling.* | temp 1.0, top_k 0, top_p 1.0, min_p 0.0 | recommended sampling (README: no truncation) |

No dense FFN, no shared expert, no bias on attention projections (tensor inventory has none). n_expert_used is 8 on every layer.

## 2. Tensor inventory and quant types (VERIFIED)

Full dumps: `docs/gguf_inventory_q4_k_xl.txt`, `docs/gguf_inventory_q8_k_xl.txt` (metadata, every tensor with shape/type/bytes, family summary). 744 tensors in both files. Total tensor bytes: Q4_K_XL 17,617,780,736 B; Q8_K_XL 22,855,473,152 B.

IMPORTANT: neither file contains Q4_K, Q5_K or Q6_K. The only types are Q8_0, BF16, F32 and NVFP4. "Q4_K_XL" has `general.file_type = 7` (MOSTLY_Q8_0) and its 4-bit part is NVFP4 experts; "Q8_K_XL" is BF16 for everything that is not an NVFP4 expert (file_type 32).

Shapes are ggml `{ne0, ne1[, ne2]}`. nh = 24, nk = 8, hd = 128, E = 2304, K = 4, nr = 16.

| Family (per physical layer unless noted) | Shape | Q4_K_XL type | Q8_K_XL type | Layers |
|---|---|---|---|---|
| token_embd.weight | {2304, 201024} | Q8_0 | BF16 | once |
| token_embd_norm.weight | {2304} | F32 | F32 | once |
| output.weight (lm_head, not tied) | {2304, 201024} | Q8_0 | BF16 | once |
| output_norm.weight | {2304} | F32 | F32 | once |
| output_res_query.weight | {2304} | F32 | F32 | once |
| attn_q | {2304, 3072} | Q8_0 | BF16 | 0-23 |
| attn_k, attn_v | {2304, 1024} | Q8_0 | BF16 | 0-23 |
| attn_output | {3072, 2304} | Q8_0 | BF16 | 0-23 |
| attn_gate (the "wqkv_gate" output gate) | {2304, 3072} | Q8_0 | BF16 | 0-23 |
| attn_rel | {2304, 384} (384 = nh*nr) | Q8_0 | BF16 | 0-23 |
| attn_sink_proj | {2304, 24} | BF16 | BF16 | 0-23 |
| attn_q_clip | {128, 24} | F32 | F32 | 0-23 |
| attn_rel_clip | {16, 24} | F32 | F32 | 0-23 |
| attn_rel_bias | {16, 1024} | F32 | F32 | 0-23 |
| attn_k_conv, attn_v_conv | {4, 1024} | F32 | F32 | 0-23 |
| attn_post_conv, ffn_post_conv | {4, 2304} | F32 | F32 | 0-23 |
| attn_norm (single input norm) | {2304} | F32 | F32 | local layers (20 phys) |
| attn_q_inp_norm, attn_k_inp_norm, attn_v_inp_norm | {2304} each | F32 | F32 | global layers 5,11,17,23 |
| attn_k_conv_proj, attn_v_conv_proj (dynamic conv taps) | {2304, 32} (32 = K*nk) | BF16 | BF16 | global layers only |
| attn_res_query | {2304} local, {2304, 3} global | F32 | F32 | layers 1-23 (none on layer 0) |
| ffn_res_query | {2304} | F32 | F32 | 0-23 |
| ffn_norm | {2304} | F32 | F32 | 0-23 |
| ffn_gate_inp (router) | {2304, 128} | F32 | F32 | 0-23 |
| exp_probs_b.bias (selection bias) | {128} | F32 | F32 | 0-23 |
| ffn_gate_up_exps.weight | {2304, 2304, 128} (rows 0..1151 gate, 1152..2303 up) | NVFP4 layers 0-19, Q8_0 layers 20-23 | NVFP4 layers 0-19, BF16 layers 20-23 | 0-23 |
| ffn_down_exps.weight | {1152, 2304, 128} | NVFP4 layers 0-19, Q8_0 layers 20-23 | NVFP4 layers 0-19, BF16 layers 20-23 | 0-23 |
| ffn_gate_up_exps.scale, ffn_down_exps.scale | {128} F32 per-expert outer scale | F32 | F32 | layers 0-19 only (NVFP4 layers) |
| loop_{q,k,v,ffn}_ctrl_down.weight | {4608, 64} | F32 | BF16 | looped layers 6-17 |
| loop_{q,k,v,ffn}_ctrl_forget.weight | {64, 2304} | F32 | BF16 | 6-17 |
| loop_{q,k,v,ffn}_ctrl_gain.weight | {64, 2304} | F32 | BF16 | 6-17 |
| loop_msg_query.weight | {2304, 4} | F32 | F32 | 6-17 |

Layers 20-23 (the last four, all in the coda) have unquantized-to-NVFP4 experts (Q8_0 or BF16) and no `.scale` tensors. All 24 layers have all non-loop tensors; only layers 6-17 have `loop_*`.

Per-physical-layer byte sizes (VERIFIED from inventory):
- NVFP4 gate_up 382,205,952 B (2,985,984 per expert); down 191,102,976 B (1,492,992 per expert). Rows: gate_up row = 36 blocks x 36 B = 1296 B (2304 elems), down row = 18 blocks x 36 B = 648 B (1152 elems).
- Q8_0 (Q4 file): q/o/gate 7,520,256 B each, k/v 2,506,752 B, rel 940,032 B. Q8_0 experts (layers 20-23): gate_up 721,944,576 B/layer, down 360,972,288 B/layer.
- BF16 (Q8 file): q/o/gate 14,155,776 B each, k/v 4,718,592 B, rel 1,769,472 B.
- lm_head and embedding: 492,106,752 B (Q8_0) or 926,318,592 B (BF16) each.

### 2.1 Layouts of the types that occur

- F32, BF16: plain row-major, ne0 contiguous. BF16 = upper 16 bits of an f32.
- Q8_0: block of 32 elements, 34 B: `half d; int8 qs[32]`; value = d * qs[i]. Row length must be a multiple of 32 (2304 = 72 blocks, 3072 = 96 blocks, 1152 = 36 blocks).
- NVFP4 (ggml type 40, `block_nvfp4`, QK_NVFP4 = 64, 36 B per block of 64 elements):
  ```
  uint8 d[4];    // UE4M3 scales, one per 16-element sub-block s = 0..3
  uint8 qs[32];  // 4-bit E2M1 codes; sub-block s uses qs[16*s .. 16*s+7]  (8 bytes = 16 values)
  ```
  Sub-block s, byte j (0..7): `qs[8*s + j]`; low nibble is element `16*s + j`, high nibble is element `16*s + j + 8`. (ggml-quants.c `dequantize_row_nvfp4`: yb[j] = low, yb[j + 8] = high.) Note the d[] and qs[] strides: qs offset for sub-block s is `s*8` (QK_NVFP4_SUB/2 = 8), NOT 16.
  - E2M1 code to value, `kvalues_fp4` (stored doubled): `{0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12}` (index = 4-bit code, bit 3 = sign, so code 8 is -0). Real E2M1 value = kvalue / 2.
  - UE4M3 scale decode `ggml_ue4m3_to_fp32(x)`: if x == 0 or x == 0x7F return 0; exp = (x >> 3) & 15, man = x & 7; raw = (exp == 0) ? man * 2^-9 : (1 + man/8) * 2^(exp-7); return raw * 0.5 (the 0.5 cancels the doubled kvalues). Sign bit is ignored (unsigned).
  - Element value inside the block: `w = kvalue[code] * ue4m3_to_fp32(d[s])` = E2M1(code) * raw_UE4M3(d[s]).
  - Final expert weight used by the model: `w_eff[e][r][c] = kvalue[code] * ue4m3_to_fp32(d) * scale[e]` where `scale` is the per-expert F32 tensor `ffn_{gate_up,down}_exps.scale` (values about 4e-5 .. 1.7e-4). ggml applies `scale[e]` as a multiply on the matmul OUTPUT of expert e (see section 8), mathematically identical. The source checkpoint dequant is `E2M1 * E4M3(scale16) * (amax[e] / outer_scale_denominator)` (conversion/inkling2.py); the GGUF `.scale` is that per-expert factor (INFERRED, magnitude consistent: observed UE4M3 bytes ~110, scale ~5e-5).
  - ggml CPU `vec_dot` for NVFP4 quantizes the activation vector to Q8_0 first and accumulates integer dot products (`ggml_vec_dot_nvfp4_q8_0`); `build_lora_mm_id` marks NVFP4 expert matmuls GGML_PREC_BF16 accumulate. A Loom kernel that dequantizes to f16/f32 and multiplies f32 activations will differ at about 1e-3 relative; this is expected, tolerate it in oracle checks.
- No MXFP4 / F8_E4M3 / K-quant tensors occur in these files.

## 3. Layer schedule and loop (VERIFIED structure, messaging per section 6, all VERIFIED)

Visit plan (`scripts/cost_model.py` prints it): visits v = 0..47, physical layer
```
phys(v) = v            for v in 0..5          (prelude)
phys(v) = 6 + (v-6)%12 for v in 6..41         (loop: pass p = (v-6)/12 in {0,1,2}, layers 6..17)
phys(v) = v - 24       for v in 42..47        (coda: physical 18..23)
```
- Per-visit state (KV cache for that visit, conv history for that visit) is separate for each visit v: 48 KV caches and 48 conv-state blocks per sequence. Weights are shared across passes. (Public: "each visit keeps its own KV and short-convolution state", consistent with llama.cpp unrolling to 48 logical layers; INFERRED.)
- Local/global pattern per visit follows the physical layer: global visits are v = 5, 11, 17, 23, 29, 35, 41, 47 (8 global); the other 40 are local.
- Attention-residual block boundary: a new block starts when `v % 6 == 0` (v = 0, 6, 12, ..., 42), using the physical index ip (equal to the visit index mod 6, see section 6.1). Verified: the code tests `ip % 6` (physical index); the loader requires core_start and core_end to be multiples of 6.
- The per-expert NVFP4 `.scale` tensors apply on EVERY visit, including passes 2 and 3 (README: dropping them on later passes gives perplexity ~130). Nothing else about weights differs per pass; what differs per pass is the message mixing (section 6) and the state slot (KV, conv).

## 4. Embedding and output

Embedding: `x0 = rms_norm(token_embd[token_id]) * token_embd_norm` (eps 1e-6). Embedding rows are BF16 or Q8_0 in the file; dequantize the single row. For raw image embeddings the norm is skipped (not needed for text). Token ids are < 200058.

Output (after the last visit v = 47):
```
cur   = mix(blocks, cur, output_res_query)          # 1 query, section 5.3, result [2304]
cur   = rms_norm(cur) * output_norm                  # eps 1e-6
logit = lm_head . cur                                # [201024], accumulate in f32
logit = 30 * tanh(logit / 30)                        # final_logit_softcapping
logits = logit[0 : 200058]                           # drop padded vocab rows
```
In prefill only the rows requested (last token) go through output norm and lm_head.

## 5. One layer visit (physical layer l, visit v)

Let `cur` be the running partial sum (shape [2304] per token). `blocks` is the list of stored block states (each [2304]). Let `swa(l)` = layer is local.

### 5.1 State at the start of a visit
```
residual = cur
```

### 5.2 Position-dependent scalar (global layers only)
```
tau = 1 + alpha * ln( max(1, (pos + 1) / 8192) )      # alpha = 0.1, pos = absolute position of the token (0 based)
```
`tau` multiplies q and r of global layers. Local layers do not use it.

### 5.3 Attention-residual mix `mix(blocks, cur, query)`
Inputs: `states = blocks + [cur]` (S = len(blocks) + 1 vectors of 2304; order does not matter), `query` with nq columns ({2304} counts as nq = 1).
```
s_i      = rms_norm(states_i)                 # eps 1e-6, no weight, over the 2304 dim
score_iq = dot(s_i, query_q) / sqrt(2304)
w_q      = softmax_i(score_iq)                # over the S states
out_q    = sum_i w_iq * s_i                   # weighted sum of the NORMALIZED states
```
Result: `out` with nq vectors of 2304. Note the weighted sum uses the rms-normalized states, not the raw ones.

### 5.4 Attention-input selection
```
if l == 0 (no attn_res_query):  in_q = in_k = in_v = cur
else:
    mixed = mix(blocks, cur, attn_res_query[l])          # nq = 1 for local layers, 3 for global layers
    local : in_q = in_k = in_v = mixed[0]
    global: in_q = mixed[0], in_k = mixed[1], in_v = mixed[2]
local : in_q = rms_norm(in_q) * attn_norm ;  in_k = in_v = in_q
global: in_q = rms_norm(in_q) * attn_q_inp_norm ; in_k = rms_norm(in_k) * attn_k_inp_norm ; in_v = rms_norm(in_v) * attn_v_inp_norm
```
(eps 1e-6.) In loop passes 2 and 3 the message controllers modify in_q / in_k / in_v before the norms, and local layers then have three different inputs (section 6, VERIFIED).

### 5.5 Projections
```
q = Wq in_q            # [3072] = 24 heads x 128
k = Kconv( Wk in_k )   # [1024] = 8 heads x 128, see 5.6 (conv offset 0)
v = Vconv( Wv in_v )   # [1024],                             (conv offset 1024)
r = Wrel in_q          # [384]  = 24 heads x 16
sink = Wsink in_q      # [24]   one logit per head
gate = sigmoid( Wgate in_q )   # [3072]
```
Then, per head (vectors over the head dim; heads index the second axis):
```
clip(x, scale, T):   x = x * scale (elementwise, if scale given);  n = sqrt(sum(x^2)) + 1e-6;  return x * clamp(T / n, 0, 1)
q = clip(q.reshape(128, 24), attn_q_clip {128, 24}, 22.6274)
k = clip(k.reshape(128, 8),  none,                 22.6274)
v = rms_norm(v.reshape(128, 8))                    # eps 1e-6, no weight
r = clip(r.reshape(16, 24),  attn_rel_clip {16, 24}, 8.0)
if global: q = q * tau ;  r = r * tau
```
The clip scales are per (dim, head) elementwise multipliers, not per-token.

### 5.6 Short convolution `convolve(x, w, dynamic, state)` (used 4 times per visit)
`x` is [C] per token, `w` is {K=4, C} F32 with `w[0][c]` multiplying the OLDEST tap and `w[3][c]` the current token. The conv is causal depthwise.
```
xcat[c, t]   = concatenation of the K-1 = 3 stored previous inputs of channel c and the current inputs   (inputs, i.e. pre-conv x)
y[c, t]      = sum_{i=0..3} xcat[c, t + i] * w[i][c]             # t indexes the current token; xcat[c, t+3] is x_t
if dynamic:   # global layers, k and v convs only
    d[lag, h, t] = dynamic_proj(in)[ lag + 4*h ]                   # dynamic_proj: {2304, 32}; heads h = 0..7
    y[c, t]     += sum_{lag=0..3} d[lag, h(c), t] * x_{t-lag}[c]   # h(c) = c / 128; lag 0 = current token
out[c, t] = (x_t[c] + y[c, t]) / sqrt( (dynamic ? 2 : 1) * 4 + 1 )    # divisor sqrt(5) static, sqrt(9) dynamic
state_new = last 3 inputs x (pre-conv) of this channel
```
Verified against `ggml_ssm_conv` CPU code (dot of 4-wide window with `w[:, c]`) and `inkling2.cpp::convolve`. For `dynamic`: `attn_k_conv_proj`/`attn_v_conv_proj` are applied to `in_k`/`in_v` respectively (the same input as the matching projection). The conv state is stored pre-conv, so decode needs only the 3 previous inputs, zero-initialised at sequence start.

The four conv sites per visit and their state rows (channels):
| Site | Input | w | Channels C | State row offset |
|---|---|---|---|---|
| k | Wk in_k | attn_k_conv (+ dynamic k on global) | 1024 | 0 |
| v | Wv in_v | attn_v_conv (+ dynamic v on global) | 1024 | 1024 |
| post-attn | Wo (gated attn out) | attn_post_conv | 2304 | 2048 |
| post-ffn | MoE output | ffn_post_conv | 2304 | 4352 |

State per visit = 3 x 6656 floats (6656 = 2*1024 + 2*2304), F32 = 79,872 B (llama.cpp allocates the max over layers; all layers are equal here).

### 5.7 Attention
Per token t, head h (kv head g = h / 3), over keys j in the visit's KV cache:
```
dist(t, j) = pos_t - pos_j
bias(t,h,j) = sum_{m=0..15} r[m][h][t] * rel_bias[m][ d ]   with d = dist if 0 <= dist < 1024 else 1024; rel_bias row 1024 is zero ("pad")
score(t,h,j) = dot(q[:,h,t], k[:,g,j]) / sqrt(128) + bias(t,h,j)       # masked: -inf
p_j = exp(score_j) / ( exp(sink[h][t]) + sum_j' exp(score_j') )       # sink is an extra softmax logit with value vector 0
o[:,h,t] = sum_j p_j v[:,g,j]
o = o * gate                                                          # elementwise [3072]
```
- Mask: causal (`pos_j <= pos_t`) and, for local visits, `pos_t - pos_j < 1024` (LLAMA_SWA_TYPE_STANDARD: masked if p1 - p0 >= n_swa). Global visits: causal only (full context up to 1M).
- `rel_bias` is {16, 1024} for ALL layers (local ones too; n_swa = extent = 1024). For global layers keys farther than 1023 get bias 0.
- sink is per token and per head (a projection of in_q), not a learned constant.
- No RoPE, no alibi. Scores are not additionally clamped.
- llama.cpp flash-attention path folds the sink into the bias (`bias - sink`) and uses a zero sink; the non-FA path passes the sink to soft_max_ext. Both equal the formula above.
- Output projection and post conv: `a = Wo o` ([2304]); `a = convolve(a, attn_post_conv, none)` (section 5.6, divisor sqrt(5)).

### 5.8 Block bookkeeping and FFN input
```
if v % 6 == 0:  blocks.append(residual)     # start of a new block: store the visit input; cur is the attention output ALONE
else:           cur = residual + cur
residual = cur
m   = mix(blocks, cur, ffn_res_query[l])    # nq = 1
x   = rms_norm(m) * ffn_norm[l]
y   = moe(x)                                # section 7
y   = convolve(y, ffn_post_conv, none)      # sqrt(5) divisor
cur = residual + y
```
At visit 0 `blocks` is empty and the attention uses `cur` directly (no attn_res_query); the visit-0 input `residual` (the normalized embedding) is stored as block 0.

## 6. Loop messaging (passes 2 and 3) - VERIFIED against llama.cpp `src/models/inkling2-loop.cpp` and `inkling2.cpp::{build_layer, control}` (mainline clone ~/inkling/inkB/src, HEAD 1393492)

Notation: `ip` = physical layer index, core = [6, 18), passes p = 0, 1, 2. `mix(states, cur, query)` is section 5.3 (it takes a list of states plus one more state).

### 6.1 Graph order
```
cur = embed(tokens); blocks = []
for ip in 0..5:   cur = layer(ip, cur, blocks)          # prelude; layer 0 pushes block b0 (ip % 6 == 0)
prelude = cur ; prelude_blocks = blocks                  # = [b0]
for pass in 0..2:
    if pass > 0:
        bank     = blocks[len(prelude_blocks)+1 :]       # drops the first core block (= the prelude output pushed by ip 6)
        bank_out = cur                                   # output of the previous pass
        cur      = prelude                               # EVERY PASS RESTARTS FROM THE PRELUDE OUTPUT
        blocks   = prelude_blocks                        # and from the prelude block list only
    for ip in 6..17:
        if pass == 0: cur = layer(ip, cur, blocks)
        else:         msgs = mix(bank, bank_out, loop_msg_query[ip])      # 4 queries -> 4 messages [Q, K, V, FFN]
                      cur = layer(ip, cur, blocks, msgs)
for ip in 18..23: cur = layer(ip, cur, blocks)           # coda; uses the blocks left by the LAST pass
output: mix(blocks, cur, output_res_query) ...           # section 4
```
- The blocks pushed during a pass are at ip = 6 (input = prelude output) and ip = 12 (input = state after ip 11). So after a pass `blocks = [b0, p6, p12]`; `bank = [p12]`, `bank_out` = the pass output. The previous-pass message therefore reads exactly 2 states: `p12` and the pass output. Pass 1 (second pass) reads pass 0's states, pass 2 reads pass 1's.
- Within a pass > 0 the ordinary attention-residual mixes (5.4, 5.8) use `blocks` of the CURRENT pass (prelude block list plus the blocks pushed so far in this pass); the coda continues with the last pass's list.
- Block boundaries use the physical layer index (`ip % 6 == 0`). The loader requires `core_start % 6 == 0` and `core_end % 6 == 0`, so this equals the visit index modulo 6.
- Pass 0 uses no loop tensors; it is identical to a non-looped layer. A fixed loop needs `loop.count >= 2`; `loop.varying` is false for this model (the varying variant only lowers the pass count).
- Cache slots: the prelude and pass 0 use slot `il = ip` (0..17); pass p >= 1 uses `il = 24 + (p-1)*12 + (ip-6)` (24..47); the coda uses `il = ip` (18..23). Each slot has its own KV and conv state; weights come from physical layer `ip`.

### 6.2 Message and controller (layers ip 6..17, passes >= 1)
Message queries: `loop_msg_query {2304, 4}`: column 0 = Q, 1 = K, 2 = V, 3 = FFN (the converter stacks `[q[2i], kv[2i], kv[2i+1], q[2i+1]]` = q, k, v, mlp queries). The mix uses states = `bank + [bank_out]` (S = 2): rms_norm without weight, score `s_i = u_c . rms(state_i) / sqrt(2304)`, softmax over the 2 states, `m_c = sum_i w_i rms(state_i)`. Messages are per token.

Controller c in {Q, K, V, FFN}, input `x` and message `m_c` (both [2304], per token):
```
z      = ctrl_down_c [ rms_norm(x) ; rms_norm(m_c) ]      # eps 1e-6, no weight; concat order: x first, m second; ctrl_down {4608, 64}
gain   = tanh( ctrl_gain_c   z )                          # {64, 2304}
forget = tanh( ctrl_forget_c z )                          # {64, 2304}
x'     = x - forget * x + gain * m_c
```
Placement:
- Attention (layers 6..17, pass >= 1): after the attention-residual mix of 5.4 and BEFORE the input norms: `in_j = control(in_j, m_j, ctrl j)` for j = Q, K, V, where `in_j` is the 5.4 mix output (local layers: the same mix output for all three j; global layers: mixed[0..2]). THEN the norms: local layers apply `attn_norm` to each of the three (so in_q, in_k, in_v differ in passes >= 1); global layers apply q/k/v_inp_norm as usual.
- FFN: `x = mix(blocks, cur, ffn_res_query)`, then `x = control(x, m_FFN, ctrl FFN)`, then `rms_norm * ffn_norm`, then the MoE.
- Controller weights are F32 (Q4 file) or BF16 (Q8 file); the reference runs `down`, `gain`, `forget` as ordinary matmuls. Accumulate in f32.

### 6.3 NVFP4 `.scale` per pass and the "re-share" semantics
- Pass-p layers are cache slots whose `layers[il]` struct is a copy of `layers[ip]` (same weight pointers). The slots were copied in `load_arch_tensors`, before the generic loader pass attached the optional `ffn_gate_up_exps.scale` / `ffn_down_exps.scale` tensors, so slots 24..47 had null scale pointers and ran NVFP4 experts unscaled (perplexity ~130). Commit 1393492 copies the slot structs again after the scale pass. Net effect: all 3 passes use identical weights and identical per-expert scales. There is no per-pass scale; loop controllers and queries are indexed by physical layer and shared by passes 1 and 2.
- Mapping: `ffn_gate_up_exps.scale {128}` is loaded as `ffn_up_exps_s` and multiplies the whole gate_up output (gate and up halves) of each selected expert; `ffn_down_exps.scale` multiplies the down output (llama-graph.cpp `build_lora_mm_id`: `get_rows(scale, ids)` then `mul`). Layers 20-23 have no scale tensors (no multiply).

## 7. MoE FFN (every visit, VERIFIED against `build_moe_ffn` and `inkling2.cpp`)

Input `x` = ffn-normed vector [2304].
```
logits = ffn_gate_inp x                 # F32 matmul with F32 accumulate, {2304,128} -> [128]
p      = sigmoid(logits)
sel    = p + exp_probs_b                # bias only for selection
idx    = top-8 indices of sel (argsort descending)
w      = p[idx]                         # unbiased sigmoid probabilities
w      = w / (sum(w) + 1e-20)           # normalized over the 8
w      = w * 8.0                        # expert_weights_scale
for e in idx:
    gu    = W_gate_up[e] x * scale_gu[e]            # [2304]; NVFP4 layers only multiply by scale (layers 20-23 have none)
    gate  = gu[0:1152] ; up = gu[1152:2304]
    a     = 4 * tanh(gate/4) * sigmoid(gate)  *  25 * tanh(up/25)         # SiTU, beta = 4, linear_beta = 25
    d     = W_down[e] a * scale_down[e]               # [2304]
y = sum_e w[e] * d
```
- SiTU: `situ(gate, up) = beta*tanh(gate/beta) * sigmoid(gate) * lb*tanh(up/lb)` (soft-capped swish gate times soft-capped linear branch).
- Top-k tie behaviour follows ggml_argsort_top_k; the selection order inside the 8 does not matter for the sum.
- Router in F32 (`hparams.expert_router_f32`), weights norm eps 1e-20.
- No expert groups, no shared expert, no expert biases.

## 8. State that persists across decode steps (per sequence)

| State | Per visit | Visits | Total |
|---|---|---|---|
| Global KV: K and V, F16 [1024 each] per token (8 kv heads x 128) | 4096 B per token | 8 (v = 5, 11, 17, 23, 29, 35, 41, 47) | 32 KiB per token; 1 GiB at 32k tokens; 32 GiB at 1M |
| Local KV: ring of the last 1024 tokens | 4096 B x 1024 = 4 MiB | 40 | 160 MiB (needs only distances 0..1023; llama.cpp keeps cell positions) |
| Conv history: 3 x 6656 F32 | 79,872 B | 48 | 3.65 MiB |
| Attention-residual `blocks` and the loop bank | transient within a token step (block states [2304] each: up to 8 blocks) | | tiny; recomputed per token, NOT cached across tokens |
Position `pos` per token (needed for tau and relative distance). KV cache type is F16 by default (common.h cache_type_k/v); the graph does not require it. Recurrent (conv) state is zero at sequence start. llama.cpp also supports rollback of the recurrent state (`llm_arch_supports_rs_rollback`); not required here.

## 9. Prefill vs decode

- Same math. Differences are only: (1) prefill processes T tokens at once, so convs read the last 3 stored inputs plus the T-1 earlier inputs of the chunk and write the final 3; (2) the attention mask is causal within the chunk and over the cache; (3) only the last token's logits are needed; (4) the MoE routes each token independently, so prefill uses expert-grouped GEMMs (MUL_MAT_ID with tokens sorted by expert).
- `tau`, relative distance and SWA depend on the absolute positions, so prefill chunking must give each token its true position.
- Results must not depend on chunk size (other implementations needed FP32 controllers for exactly this reason).

## 10. Tokenizer and chat format (VERIFIED)

- `tokenizer.ggml.model = gpt2` (byte-level BPE), `tokenizer.ggml.pre = inkling2` which maps to `LLAMA_VOCAB_PRE_TYPE_GPT4O` with BOS = none. 200058 tokens (ids 0..200057), 424,413 merges. `clean_spaces = false`.
- Pre-tokenizer regex (llama-vocab.cpp, case LLAMA_VOCAB_PRE_TYPE_GPT4O, as implemented by llama.cpp):
  ```
  [^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))*((?=[\p{L}\p{M}])([^A-Z]))+(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|[^\r\n\p{L}\p{N}]?((?=[\p{L}\p{M}])([^a-z]))+((?=[\p{L}\p{M}])([^A-Z]))*(?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+
  ```
  Original tokenizer.json regex: `[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+`.
- 60 non-normal tokens (all type CONTROL = 3), ids 199998..200057. Named ones: 199998 `<|unused|>`, 199999 `<|endoftext|>` (padding_token_id), 200000 `<|message_user|>`, 200001 `<|message_model|>`, 200002 `<|message_system|>`, 200003 `<|message_tool|>`, 200004 `<|content_text|>`, 200005 `<|content_image|>`, 200006 `<|content_model_end_sampling|>` (EOS and EOT), 200008 `<|content_thinking|>`, 200010 `<|end_message|>`, 200022 `<|content_tool_error|>`, 200024 `<|content_xml|>`, 200028 `<|begin_of_text|>` (not added), 200049 `<|content_invoke_tool_json|>`, 200054 `<|unused_200054|>` (image placeholder in template), 200057 `<|content_invoke_tool_text|>`; the rest are `<|unused_NNNNNN|>`.
- `add_bos_token = false`, `add_eos_token = false`; eos = eot = 200006. Generation stops on 200006 (a normal end of an assistant turn is `<|end_message|>` then `<|content_model_end_sampling|>`; llama.cpp's parser treats `<|end_message|>` as block end).
- Chat template: `docs/chat_template.jinja` (extracted from the GGUF, 6868 chars). Format, with `reasoning_effort` default 0.9 (names: none 0.0, minimal 0.1, low 0.2, medium 0.7, high 0.9, xhigh/max 0.99):
  ```
  [tools]   <|message_system|>tool_declare<|content_xml|>{sorted-key compact JSON array of {description,name,parameters,type}}<|end_message|>
  [effort]  <|message_system|><|content_text|>Thinking effort level: 0.9<|end_message|>     (emitted once, before the first non-system message, or at the end if there is none)
  user      <|message_user|><|content_text|>TEXT<|end_message|>
  system    <|message_system|><|content_text|>TEXT<|end_message|>
  tool      <|message_tool|>NAME<|content_text|>TEXT<|end_message|>
  assistant [<|message_model|><|content_thinking|>REASONING<|end_message|>]<|message_model|><|content_text|>TEXT<|end_message|>
            [<|message_model|>FN<|content_invoke_tool_json|>{"name":"FN","args":{sorted compact JSON}}<|end_message|>]*<|content_model_end_sampling|>
  image     <|message_user|><|content_image|><|unused_200054|><|end_message|>
  ```
  The template has NO generation prompt: for Inkling 2 the model itself writes `<|message_model|>` (tests/test-chat.cpp: "Inkling 2 writes it"). Model output grammar (common/parsers/inkling.cpp): zero or more blocks `<|message_model|>` + (`<|content_thinking|>` text `<|end_message|>` | `<|content_text|>` text `<|end_message|>`), then optional tool calls `<|message_model|>NAME<|content_invoke_tool_json|>{"name":"NAME","args":{...}}<|end_message|>`.
  The GGUF template wraps assistant messages in `{%- generation -%}...{%- endgeneration -%}` (training mask, not in the repo copy); a Jinja engine without that tag must strip it (render the `rendered` set-block directly).
- Sampling: temperature 1.0, top_p 1.0, top_k 0 (off), min_p 0 (no truncation), as in README and GGUF `general.sampling.*`. Public caveat: the model can repeat reasoning; cap output tokens.

## 11. Ggml ops and exact semantics used by the reference graph

`ggml_rms_norm(x, eps)`: `x / sqrt(mean(x^2) + eps)` over ne0. `build_norm(RMS, w)`: that times `w`. `ggml_soft_max_ext(x, mask, scale, bias)`: `softmax(x*scale + mask)` over ne0. `ggml_scale_bias(x, s, b)`: `x*s + b`. `ggml_sigmoid`, `ggml_tanh`, `ggml_sqrt`, `ggml_sqr`, `ggml_sum_rows` (sum over ne0), `ggml_clamp(x, lo, hi)`, `ggml_log`, `ggml_cast`, `ggml_concat`, `ggml_get_rows` (gather rows by I32 ids), `ggml_pad`, `ggml_ssm_conv(xcat {K-1+n, C, seqs}, w {K, C})` = depthwise causal conv: `out[c,t] = sum_i xcat[c,t+i] * w[i,c]`, `ggml_mul_mat_id(W {in,out,E}, x {in,used or 1,T}, ids {used,T})` = per (token, slot) matmul with expert `ids`, `ggml_argsort_top_k` (indices of the k largest along ne0). SiTU is composed from `scale/tanh/sigmoid/mul`, not a fused op. Soft-cap = `scale(tanh(scale(x, 1/cap)), cap)`. All activations are F32 in the reference (FA mask F16). Attention uses `build_attn` with bias and sink arguments (flash path: `fa_mask = F16(bias - sink + mask)`, sinks 0; non-flash: `kq_b = bias*sqrt(128)`, `sinks` tensor).

## 12. Cost model per token

Assumptions: 8 distinct experts per visit-layer (no reuse between passes or tokens; worst case for decode), loop controllers/messages read on passes 2 and 3 only, F32 router read in full on each visit, all weights read from VRAM once per visit, KV and activations excluded unless stated. Produced by `scripts/cost_model.py` (output saved in `docs/cost_model_output.txt`).

### 12.1 Decode (n = 1), weight bytes read per token

| Family (all 48 visits) | Q4_K_XL | Q8_K_XL |
|---|---|---|
| Experts, NVFP4, 8/visit (44 visits on layers 0-19 incl. coda layers 18-19) | 1576.6 MB | 1576.6 MB |
| Experts, layers 20-23 (4 visits, Q8_0 / BF16) | 270.7 MB | 509.6 MB |
| Attention projections q,k,v,o,gate,rel (+sink BF16) | 1368.7 MB (Q8_0) + 5.3 (BF16 sink) | 2581.7 MB (BF16) |
| lm_head | 492.1 MB (Q8_0) | 926.3 MB (BF16) |
| Loop controllers + msg queries (passes 2, 3 on 24 visits) | 227.4 MB (F32) | 114.1 MB (BF16) |
| Router F32 (1.18 MB x 48) | 56.6 MB | 56.6 MB |
| Dynamic conv proj (global visits) | 2.4 MB | 2.4 MB |
| Conv F32, norms, clip, rel_bias, res_query | 11.0 MB | 11.0 MB |
| **Total** | **4010.8 MB (3.74 GiB)** | **5778.3 MB (5.38 GiB)** |
| Floor @ R9700 620 GB/s | 6.47 ms = 154.6 tok/s | 9.32 ms = 107.3 tok/s |
| Floor @ 8060S 230 GB/s | 17.44 ms = 57.3 tok/s | 25.12 ms = 39.8 tok/s |

Notes: the 3 passes mean each looped layer's weights are read 3 times per token (no reuse is assumed possible across passes since the three passes are sequentially dependent; an L2/LDS cache cannot hold a layer of ~100 MB). If a runtime keeps the looped layer's active experts hot (same experts selected in two passes) the real traffic is lower; do not count on it.

Additional per-token traffic proportional to context n (KV read, F16): local visit: `4096 B x min(n, 1024)` x 40 visits = up to 168 MB per token at n >= 1024; global visit: `4096 B x n` x 8 visits = 32 KiB per context token (1 GB at n = 32k, +1.6 ms at 620 GB/s). Conv state 3.8 MB read+write. Activations: small.

Expert bytes per visit (NVFP4): 8 x (2,985,984 + 1,492,992) = 35.8 MB. Fraction of the Q4 floor: experts 46%, attention 34%, lm_head 12%, loop 6%.

### 12.2 FLOPs per token (2 x MACs), prefill (weights amortized)

Per visit: q 2304x3072 + k,v 2 x 2304x1024 + o 3072x2304 + gate 2304x3072 + rel 2304x384 + sink 2304x24 + router 2304x128 + 8 x (2304x2304 + 1152x2304) MACs; global visits add 2 x 2304 x 32; per looped visit on passes 2-3 add 4 x (2 x 4608x64 + ...) controller MACs.
- 8.84 GFLOP per token (without attention-over-context and without lm_head). 48 visits.
- Attention over context: `4 x nh x hd = 12,288` FLOPs per (token, key) pair per visit: local visits cap at 1024 keys (40 visits), global visits scale with the position (8 visits). A 512-token prefill chunk at position 0: about 512*513/2 x 12288 x 48 = 77 GFLOP total (small next to 4.5 TFLOP of the dense part).
- lm_head: 0.93 GFLOP per output token.
- A 512-token chunk costs about 4.5 TFLOP; the router is cheap; MoE is 8 x 2304 x 3456 x 2 x 48 = 6.1 GFLOP/token of the 8.84.

## 13. Per-op list (one visit); decode n = 1 and prefill n = 512

M = tokens (1 or 512). Weight types: AP = attention weight type (Q8_0 or BF16), EX = expert type (NVFP4, or Q8_0/BF16 on layers 20-23).

| # | Op | Inputs | Output | Decode shape | Prefill shape |
|---|---|---|---|---|---|
| 1 | mix(attn_res) | blocks [<=8][2304][M], cur, query {2304,nq} | [nq][2304][M] | S<=9 states: rmsnorm 9 vecs, 9 dots, softmax, weighted sum | same x512 tokens |
| 2 | rms_norm x(1 or 3) with weight | [2304][M], w {2304} | [2304][M] | | |
| 3 | matmul q | in_q, Wq {2304,3072} AP | [3072][M] | GEMV 3072 rows x 2304 | GEMM 3072 x 512 x 2304 |
| 4 | matmul k, v | in_k/in_v, {2304,1024} AP | [1024][M] | GEMV 1024 x 2304 | GEMM |
| 5 | matmul rel | in_q, {2304,384} AP | [384][M] | GEMV | GEMM |
| 6 | matmul sink | in_q, {2304,24} BF16 | [24][M] | GEMV 24 rows | GEMM |
| 7 | matmul gate (+sigmoid) | in_q, {2304,3072} AP | [3072][M] | GEMV | GEMM |
| 8 | dynamic conv proj (global) | in_k/in_v, {2304,32} BF16 x2 | [32][M] | GEMV 32 rows | GEMM |
| 9 | short conv (k, v, post-attn, post-ffn) with state | x [C][M], w {4,C}, state [3][C], (dyn [32][M]) | [C][M] | elementwise 4-tap | tiled 4-tap |
| 10 | clip (L2) with scale | q [128][24][M], k [128][8][M], r [16][24][M] | same | per (head,token) reduce over 128/16 | same |
| 11 | rms_norm per head (v) | [128][8][M] | same | | |
| 12 | tau scale (global) | q, r, pos | | | |
| 13 | rel-bias table: r [16] . rel_bias {16,1024} -> [1024] per (head, token), gather by distance | r [16][24][M], rel_bias | bias [n_kv][24][M] | 1024x16x24 MACs + gather over n_kv | 1024x16x24x512; do in-kernel, never materialize [n_kv][24][M] |
| 14 | attention with sink, bias, mask, GQA 3:1, head dim 128 | q, K/V cache (F16), bias, sink | [3072][M] | n_kv = <=1024 local, n global | tiles 16-64 q x kv |
| 15 | gate multiply | [3072][M] x sigmoid | | | |
| 16 | matmul o | {3072,2304} AP | [2304][M] | GEMV 2304 x 3072 | GEMM |
| 17 | residual add / block store | | | | |
| 18 | mix(ffn_res) + rms_norm(ffn_norm) | | [2304][M] | | |
| 19 | router matmul F32 | {2304,128} F32 | [128][M] | GEMV | GEMM, f32 accumulate |
| 20 | sigmoid + bias + top-8 + normalize + scale | [128][M] | idx [8][M], w [8][M] | | |
| 21 | expert gate_up matmul (id) | x [2304], W {2304,2304,128} EX, idx | [8][2304][M] | 8 GEMVs 2304 x 2304 (NVFP4) | tokens grouped per expert: 512*8/128 = 32 tokens avg per expert |
| 22 | SiTU activation | gate, up [1152] x 8 | [8][1152][M] | | |
| 23 | expert down matmul (id) | {1152,2304,128} EX | [8][2304][M] | 8 GEMVs 2304 x 1152 | grouped GEMM |
| 24 | scale[e] (NVFP4 layers), weight w[e], sum over 8 | | [2304][M] | | |
| 25 | loop controllers (passes 2-3 only) | 4 x (down 4608x64, forget/gain 64x2304) + message mix | | small GEMVs | |
| 26 | final: mix(output_res), rms_norm, lm_head matmul, softcap, slice | | [200058] | GEMV 201024 x 2304 (AP-type: Q8_0 or BF16) | last token only |
| 27 | embedding gather + rms_norm | | | | |

## 14. Kernel inventory (what the Loom package needs)

PR1 motif sources: `git show hrx-llama-pr1:ggml/src/ggml-hrx/kernels/<file>` (matmul/{mul_mat,mul_mat_id,mul_mat_wmma,mul_mat_id_wmma}.loom, motifs/{q8_0,q6_k,float,flash_attn,kq,wmma,reduce,common}.loom, flash_attn/*.loom, ssm_conv/ssm_conv_f32.loom, softmax, norm/rms_norm_f32.loom, reduce/{argsort,sum_rows}, rows/get_rows_f32, copy/concat).

| Kernel | Inputs -> outputs | Shapes | Quant types | Existing PR1 motif | New? |
|---|---|---|---|---|---|
| matmul vec (GEMV) | W x -> y | {2304,{3072,1024,384,2304}}, {3072,2304} | Q8_0, BF16, F32 | `mul_mat.loom` vec + motifs/q8_0, motifs/float (F32/F16/BF16 superblocks) | no (K-quants not needed) |
| matmul tiled / WMMA (prefill) | | same | Q8_0, BF16, F32 | `mul_mat_wmma`, `mul_mat` tiled | no |
| lm_head GEMV | | {2304, 201024}, cut to 200058 | Q8_0 / BF16 | `mul_mat` vec | no (large row count; check occupancy) |
| small GEMV, 24 and 32 rows | sink, dynamic-conv proj | {2304, 24/32} | BF16 | `mul_mat` vec | maybe tune: few rows, use split-K |
| MoE gate_up/down GEMV (decode) | x, ids -> | {2304,2304,128}, {1152,2304,128} | NVFP4 (+per-expert F32 scale), Q8_0/BF16 for layers 20-23 | `mul_mat_id` vec | NEW: NVFP4 weight motif (decode of block_nvfp4: UE4M3 scale + E2M1 LUT), `.scale` epilogue |
| MoE grouped GEMM (prefill) | | | NVFP4 / Q8_0 / BF16 | `mul_mat_id_wmma`/tiled | NEW: NVFP4 dequant into the WMMA tile (f16 operands) |
| router matmul F32 | | {2304,128} | F32 | `mul_mat_f32` | no |
| top-8 select + normalize | probs, bias -> idx, w | [128] -> [8] | | `argsort_f32` + `sum_rows` | NEW fused small kernel (sigmoid, +bias, top-k, gather, normalize, scale) |
| SiTU | gu [2304] -> [1152] | | | `swiglu_f32`/`unary_f32` | NEW (tanh, sigmoid, caps); fuse into down matmul prologue if possible |
| rms_norm (weighted / unweighted / per-head) | | [2304], [128] x heads | | `rms_norm_f32` | no (per-head variant check) |
| L2-clip (per head, with scale) | | q [128][24], k [128][8], r [16][24] | | `sum_rows`, `l2_norm_f32` | NEW fused elementwise+reduce |
| short conv 4-tap with state | | C = 1024, 1024, 2304, 2304 | | `ssm_conv_f32` (+ concat, cpy for state) | partially: static conv ok; dynamic taps NEW |
| mix() attention-residual | blocks, cur, query -> out | S <= 9, 2304 | | rms_norm + dots + `soft_max` | NEW fused kernel (per token: S rms, S x nq dots, softmax, weighted sum); also used with S up to 9 and nq = 1 or 3 |
| attention with rel-bias gather + sink | q, K, V, r, rel_bias, sink, pos | nh 24, nk 8, hd 128 | F16 KV | `flash_attn_ext_f32_f16_wmma` (mask F16, no sink support in soft_max) | NEW: bias computed in-kernel from r and rel_bias, sink as extra logit, local ring-buffer window; GQA 3:1 |
| KV write + positions | | | F16 | `set_rows_f32`/`cpy_f32` | no |
| lm-head softcap | logits -> | 200058 | | `unary_f32` (tanh) | NEW small op (fuse into lm_head epilogue) |
| loop controller | x, m -> x' | 4608 -> 64 -> 2304 x2 | F32/BF16 | GEMV + elementwise | NEW fused: 2 unweighted rms_norm, concat GEMV 4608 to 64, two GEMV 64 to 2304, tanh, x - forget*x + gain*m; message = mix kernel with S = 2, nq = 4 |
| embedding gather + norm | token -> [2304] | | Q8_0/BF16 | `get_rows` (Q8_0/BF16 variants) | q8_0/bf16 get_rows: check |
| sampling (host or device) | logits | 200058 | | | host is fine |

## 15. Open questions (still unverified)

1. No numerical run yet: everything is read from code. The first oracle comparison (inkB) should capture per-visit states to confirm sections 5 and 6 end to end.
2. Whether the GPU backend used for the oracle quantises activations like the CPU path (see tolerance below; CUDA uses Q8_1 via `vec_dot_nvfp4_q8_1`).
3. Top-k tie behaviour of `ggml_argsort_top_k` (only matters for exact ties).
4. `loop.varying = true` models (variable pass count) are not in these files; only the fixed 3-pass path is specified.
5. The quality gap between the Q4_K_XL and Q8_K_XL files is not measured here.

### Oracle tolerance (from the reference code)
- NVFP4 expert matmuls (CPU): `ggml_vec_dot_nvfp4_q8_0`: the activation vector is quantised to Q8_0 (blocks of 32, scale amax/127, int8) and dotted with the NVFP4 weights in integers: `dy * d_ue4m3 * sum(q8 * kvalue)`; the result is then multiplied by `scale[e]` in f32. The result is marked BF16 accumulate. Expect about 1e-3 relative difference per matmul against a kernel that keeps f32 activations; end to end this is within the model's own quantisation noise (README: PPL 7.653 vs 7.651).
- Q8_0 weights: activations are quantised to Q8_0 (`ggml_vec_dot_q8_0_q8_0`). BF16 weights on CPU: activations are converted to BF16 (`vec_dot_type = BF16`, about 4e-3 relative per element, averaging out in the dot). F32 router and F32/BF16 loop controllers: f32.
- Use relative L2 per layer output (start at about 1e-2) and compare logit top-k and KL (the README reports KLD 0.0006 between two builds).

## 16. Changes versus the first version of this spec (where the real code differs)

1. Section 6 replaced (was an UNVERIFIED reconstruction). Differences from my reconstruction: (a) the bank is exactly 2 states, `[p12 block, previous pass output]`, not all blocks of the previous pass; the current partial sum is not in the bank; (b) every pass resets `cur` to the prelude output and `blocks` to `[b0]`; (c) controllers act on the 5.4 mix output BEFORE the input norms, and on the FFN mix output before `ffn_norm`; (d) formula is `x - tanh(F z) x + tanh(G z) m` with F = forget, G = gain, z from `[rms(x); rms(m)]` (x first), matching the Goodfire form; (e) messages are `mix(...)` outputs with query columns Q, K, V, FFN; (f) pass 0 uses no loop tensors.
2. Section 5.4: in passes >= 1 local layers do not share one input: three controlled copies, each normed with the single `attn_norm`.
3. Section 3: cache-slot indexing. Prelude, pass 0 and coda use the physical index as slot; slots 24..47 are passes 1-2 (the order of slots does not change the math). Block starts use the physical index (equal to the visit index mod 6).
4. Section 6.3: the NVFP4 scale is not per pass. One scale tensor per physical layer is shared by all passes; the bug was that the slots lacked the scale pointers.
5. Confirmed (previously INFERRED, now VERIFIED): gate_up row order is `[gate; up]` halves (converter de-interleaves `w13` rows; the NVFP4 path writes the even rows, gate, first and then the odd rows, up); `.scale` is a per-selected-expert multiply on the matmul output; `v % 6` equals `ip % 6`; the sink is a per-token, per-head projection folded as `bias - sink` with a zero sink (the inkB tree has only this flash-attention form; same math as the older non-FA branch); gate = sigmoid(Wgate in_q); bias gather uses the distance clipped to `extent` with a zero pad row; tau only on global layers.
6. Cost model unchanged (controllers on 24 visits as assumed; the message mix has S = 2 and is negligible). Kernel inventory: the message mix is the `mix` kernel with S = 2, nq = 4; the controller is a fused kernel (see section 14 row "loop controller").
