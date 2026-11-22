# Served context per GPU (Q4_K_XL, published route, bead llama-cpp-tg3.18)

Method: `scripts/ctx_table.py <r9700|strix> ROWSxCONTEXT ...`. For each case the script copies the materialized preset directory as symlinks plus an edited `model.json` (`rows`, `context_capacity`; `prefill_capacity` as in the preset (512 in the first table rows, 2048 for the wave 1 rows), `checkpoint_capacity` 4, `max_tokens` 4096). It starts `loom-serve --device=<URI> --model=DIR --preset=<preset> --pool_backing=fixed` (no `--memory_bytes`: the fixed backing sizes its pools itself). It waits for registration and the first answer (weights load), then runs `ctx_admit.py --tokens <context - 100>` with `--complete` (waits for the one-token answer of the full prefill) or without it (admission only: accepted, then cancelled). The publish tree is not changed. Raw lines: `bench/<gpu>/ctx_table_results.jsonl`. Memory is read from `/sys/class/drm/card*/device/mem_info_{vram,gtt}_used` after the first request (the pool does not grow afterwards).

## Radeon AI PRO R9700 (gfx1201, preset chat-r9700, 34.2 GB VRAM)

| Rows | context_capacity | Registers | VRAM used after load | ctx_admit | Result |
|---|---|---|---|---|---|
| 4 | 32768 (preset) | yes | 21.21 GB | `--complete` | COMPLETED 32668 tokens in 22 s |
| 4 | 65536 | yes | 22.50 GB | `--complete` | COMPLETED 65436 tokens in 59 s |
| 4 | 131072 | yes | 25.09 GB | `--complete` | COMPLETED 130972 tokens in 183 s |
| 2 | 131072 | yes | 24.40 GB | `--complete` | COMPLETED in 184 s |
| 1 | 131072 | yes | 24.06 GB | `--complete` | COMPLETED in 184 s |
| 4 | 262144 | yes | 30.25 GB | `--complete` | COMPLETED 262044 tokens in 622 s |
| 4 | 327680 | yes | 32.80 GB | admission only | ADMITTED 327580 tokens (not run to the end) |
| 4 | 393216 | yes | n/a | first request | RESOURCE_EXHAUSTED while the weights load; the model stays unavailable |
| 4 | 524288 | no | n/a | n/a | registration fails with RESOURCE_EXHAUSTED (available 16.4 GB of 34.2 GB at that moment) |

## Radeon 8060S (gfx1151, preset chat, unified memory, 123 GiB host RAM)

| Rows | context_capacity | Registers | Unified memory in use after load (gtt) | ctx_admit | Result |
|---|---|---|---|---|---|
| 4 | 32768 (preset) | yes | 21.27 GB | `--complete` | COMPLETED 32668 tokens in 52 s |
| 4 | 65536 | yes | 22.56 GB | `--complete` | COMPLETED 65436 tokens in 142 s |
| 4 | 131072 | yes | 25.15 GB | `--complete` | COMPLETED 130972 tokens in 443 s |
| 4 | 262144 | yes | 30.32 GB | `--complete 2700` | COMPLETED 262044 tokens in 1520 s (a first try with the script's default 720 s read limit ended with a client timeout, not a server error) |
| 4 | 524288 | yes | 40.65 GB | admission only | ADMITTED 524188 tokens (not run to the end) |
| 4 | 1048576 (native) | yes | 61.32 GB | admission only | ADMITTED 1048476 tokens (not run to the end) |

Rows 1 and 2 were not run on the 8060S (the effect is small, see the R9700 table).

## What limits the context

- Fit, both GPUs: used memory is about 19.9 GB + 39.4 KB per context position at 4 rows (weights 17.6 GB, slots and buffers about 2.3 GB, global KV 32 KiB per position plus the pool overhead). Each row less saves about 0.34 GB (rings, conv state, per-row buffers), so rows do not decide the limit. The 8060S shows the same line in `gtt` (about 21.3 GB at 32768, 61.3 GB at 1048576).
- R9700: the VRAM decides. 4 rows fit up to 327680 positions (32.8 GB of 34.2 GB); 393216 needs about 35.4 GB and fails when the weights load, 524288 fails at registration. The line is the same with 1 or 2 rows (about 0.7 to 1.0 GB less). The limit is the fixed pool (KV), not the workspace: slots and workspace depend on `prefill_capacity` (512), not on the context. Measured non-weight memory at 32768 is about 3.6 GB (the former preset estimate was 6.7 GB; replaced, see below).
- 8060S: unified memory. The native context 1048576 registers and is admitted (61.3 GB in use, no failure); the machine has 123 GiB. The limit here is what the other tenants of the box leave free, not the device. A full request at that size is not practical: attention makes prefill super-linear (R9700: 32k 22 s, 131k 183 s, 262k 622 s; 8060S: 32k 52 s, 65k 142 s, 131k 443 s, 262k 1520 s), so a 1M-token prefill takes hours (extrapolated well above 6 hours on the 8060S; not run).
- `prefill_capacity`: 512 when the first rows above were measured (the routes kernel limited a stage to 512 tokens). Wave 1 removed that limit: the Q4 presets now use 2048 (see "Wave 1 re-measurement"); the Q8 presets stay at 512. Longer prompts are chunked by the service. At most 256 spans per stage.
- Admission only (`ctx_admit.py` without `--complete`) prints "the server did not cancel the request within 60 s": loom-serve keeps prefilling after the client disconnects. A later case on the same server would stall; `ctx_table.py` starts a new server per case.

## Decisions (lead, applied by `scripts/publish_build.py`, bead llama-cpp-tg3.19)

- Q4 presets `chat` (8060S) and `chat-r9700`: `context_capacity` 131072. Evidence: 25.1 GB used on both GPUs, a full-context request completes (R9700 183 s, 8060S 443 s). 262144 also completes but leaves about 3.9 GB free on the R9700.
- Q8 presets `chat-q8` and `chat-q8-r9700`: stay at 32768 until measured (the Q8 weights are 5.2 GB larger than Q4: 131072 would need about 30.3 GB on the 34.2 GB card).
- `prefill_capacity`: 2048 for the Q4 presets (wave 1, bead llama-cpp-tg3.30; measured faster and token-identical), 512 for the Q8 presets (not measured).
- The card build asserts that the generated presets equal `package/card.json` except `execution.context_capacity`, `execution.prefill_capacity`, `memory.transient_bytes` and `memory.workload` (`DELIBERATE` in `scripts/publish_build.py`).

## Memory model in the card and in llm_config.json

Measured at 4 rows, both GPUs, fixed pool backing: used memory = 19.9 GB + 39.4 KB per context position (fit through 21.21 GB at 32768, 22.50 GB at 65536, 25.09 GB at 131072, 30.25 GB at 262144 on the R9700; 21.27, 22.56, 25.15, 30.32 GB on the 8060S). Non-weight memory is 3.6 GB at 32768 (21.2 GB used minus 17.6 GB weights).

| Field | Value | Source |
|---|---|---|
| `resident_weights_bytes` (catalog memory block) | 17617780736 Q4, 22855473152 Q8 | tensor bytes of the GGUF files (unchanged) |
| `kv_bytes_per_position` | 39400 | the slope of the fit (the global KV alone is 32768; the rest is pool overhead that grows with the context) |
| `workspace_gib` | 2.14 | intercept of the fit minus the weights: 19.9 GB - 17.6 GB = 2.3 GB = 2.14 GiB (slots, buffers, prefill workspace at 512 tokens) |
| preset `memory.weights_bytes` | as `resident_weights_bytes` | unchanged |
| preset `memory.transient_bytes` | 8000000000 (Q4, at 131072 and prefill 2048), 3600000000 (Q8, at 32768 and prefill 512) | Q4: the fit above (19.9 GB + 39.4 KB x 131072 = 25.06 GB) plus the 0.43 GB that lane P1 measured for prefill 2048, minus the weights, rounded up: 7.9 GB, budget 8.0 GB. Measured in this build at 131072 and prefill 2048: 24.86 GB (R9700) and 24.93 GB (8060S), i.e. 7.2 and 7.3 GB of non-weight memory, below the budget. Q8: unchanged, 3.6 GB at 32768 (Q8 itself not measured) |
| preset `memory.workload` | text with the fit | this table |

The old values (transients 6.7 GB, 2.5 GiB workspace, 32768 bytes per position) were estimates; the 6.7 GB figure was high for 32768 (3.6 GB measured) and too low for 131072.

Reproduce (box, one GPU at a time, `flock -s ~/hrx-pr1/gpu.lock`):
```
python3 scripts/ctx_table.py r9700 4x32768 4x65536 4x131072 4x262144            # --complete 1800 is the default
python3 scripts/ctx_table.py r9700 4x327680 4x393216 4x524288 --complete 0      # admission only
```

## Wave 1 re-measurement (bead llama-cpp-tg3.30)

Build: wave 1 package (D1 + D4 + D5 + P1), Q4 presets with `prefill_capacity` 2048, `scripts/ctx_table.py <gpu> 4x131072` (a new server per case, `--complete`), idle box, one case per GPU. Raw lines: `bench/<gpu>/ctx_table_results.jsonl` (field `build`).

| GPU | Rows | context_capacity | Memory in use after the request | Result | Earlier (prefill 512) |
|---|---|---|---|---|---|
| R9700 (chat-r9700) | 4 | 131072 | 24.86 GB VRAM | COMPLETED 130972 tokens in 120 s | 183 s, 25.09 GB |
| 8060S (chat) | 4 | 131072 | 24.93 GB gtt | COMPLETED 130972 tokens in 343 s | 443 s, 25.15 GB |

Full-context prefill is 34 percent (R9700) and 23 percent (8060S) faster. The gates run (`ctx_admit.py` with an admission request first, then `--complete`) reported 130991 tokens in 179 s (R9700) and 626 s (8060S): the first request is accepted and then cancelled by the client, but loom-serve keeps prefilling it (see above), so the second request probably waits for it (not isolated). Use the single `--complete` case for the time of one prefill.

The memory measured here is 0.2 GB below the old fit although the workspace is larger (the +0.43 GB of lane P1 is not visible in this reading). The fit and the preset budget stay conservative.
