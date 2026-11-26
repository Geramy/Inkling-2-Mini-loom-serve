---
base_model: unsloth/Inkling-2-Mini-GGUF
license: mit
tags:
- loom
- loom-serve
- gguf
- strix-halo
- radeon-r9700
---

# Inkling-2-Mini (UD-Q4_K_XL) for AMD Loom

A [Loom](https://github.com/ROCm/hrx-system) package (`.loom` programs and a `model.json` card) for the unmodified GGUF checkpoint [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) (`Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file`), for the AMD Ryzen AI Max+ 395 "Strix Halo" and AMD Radeon AI PRO R9700 GPUs, generated with the [loom-skills](https://github.com/loom-fun/loom-skills) workflow. Learn more on the [Loom ecosystem website](https://loom-fun.github.io/loom-dev/).

## Quick start

Open your GPU and copy its commands.

<details>
<summary>AMD Ryzen AI Max+ 395 "Strix Halo" (gfx1151)</summary>

```bash
pipx install "hrx-loom[hf]==0.3.0"
loom-fetch huggingface:local/unsloth-Inkling-2-Mini-loom:chat --materialize=copy --output=unsloth-Inkling-2-Mini-loom
loom-serve --model=unsloth-Inkling-2-Mini-loom '--device=amdgpu://?completion_spin_ns=4294967295' --pool_backing=fixed
```

- `--materialize=copy` puts copies of the fetched package files in the output directory.
- `completion_spin_ns=4294967295` enables active waiting for GPU completions (uses CPU while GPU work is in flight); the card's performance measurements use this setting.
- `--pool_backing=fixed` commits the preset's memory pool up front instead of growing elastic backing.

On a stock host (8 MiB memlock) start loom-serve as `systemd-run --user --pty --same-dir -p SystemCallFilter=~io_uring_setup -p SystemCallErrorNumber=ENOSYS loom-serve ...` (no root; IREE falls back to its POSIX proactor), or raise `ulimit -l`.

</details>

<details>
<summary>AMD Radeon AI PRO R9700 (gfx1201, 32 GB)</summary>

```bash
pipx install "hrx-loom[hf]==0.3.0"
loom-fetch huggingface:local/unsloth-Inkling-2-Mini-loom:chat-r9700 --materialize=copy --output=unsloth-Inkling-2-Mini-loom
loom-serve --model=unsloth-Inkling-2-Mini-loom '--device=amdgpu://?completion_spin_ns=4294967295' --pool_backing=fixed --parameter_mode=preload
```

- `--materialize=copy` puts copies of the fetched package files in the output directory.
- `completion_spin_ns=4294967295` enables active waiting for GPU completions (uses CPU while GPU work is in flight); the card's performance measurements use this setting.
- `--pool_backing=fixed` commits the preset's memory pool up front instead of growing elastic backing.
- `--parameter_mode=preload` only changes how the weights are loaded on the host, so the performance rows were measured without it unless their job served with it.

</details>

## Send a request

In another terminal (`reasoning_effort` is set to `none`):

```bash
curl http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model": "thinkingmachines/Inkling-2-Mini", "messages": [{"role": "user", "content": "Hello"}], "reasoning_effort": "none"}'
```

## Performance vs llama.cpp

### Strix Halo (gfx1151)

| metric | llama.cpp | Loom | margin vs best llama.cpp | roofline | Loom % of roofline |
|---|---|---|---|---|---|
| decode tok/s, short prompt | 14.9 | 44.2 | +197 % | 57.2 | 77 % |
| decode tok/s, 2k context | 11.3 | 41.6 | +268 % | 54.1 | 77 % |
| decode tok/s, 32k context | — | 34.6 | — | 43.9 | 79 % |
| prefill tok/s, 1k document | 57.6 | 1203.5 | +1989 % | 4792.4 | 25 % |
| prefill tok/s, 2k document | 53.6 | 1331.6 | +2384 % | 4701.2 | 28 % |
| prefill tok/s, 8k document | 48.7 | 1341.2 | +2654 % | 4511.8 | 30 % |
| prefill tok/s, 16k document | 44.6 | 1265.7 | +2738 % | 4329.3 | 29 % |
| prefill tok/s, 32k document | — | 1115.8 | — | 4014.1 | 28 % |
| 4 streams, aggregate decode tok/s | — | 90.1 | — | 217.8 | 41 % |
| speculative decoding, drafted tokens accepted | n/a | n/a | n/a | n/a | n/a |

**Methodology**

- Compared: Loom against llama.cpp ROCm `llama.cpp-cpu/418fe97+PR32704`; the same GGUF, the same machine and session, the same prompts; Loom served with the active wait. Values are the median of 3 sessions (4 streams: median of 3); Loom is timed by the client, llama.cpp by its server.
- Out of the box on both engines: Loom serves the preset shipped in this package; llama.cpp runs its build defaults plus only the server arguments listed below (offload, context and slots). No tuning sweep on either engine.
- Drafting, 1-stream rows (decode, decode_2k, decode_32k, prefill): no drafting on either engine (no MTP head in the GGUF and no drafter published for this exact model). Each engine is loaded once for all these rows.
- 4 streams: both engines loaded with drafting off, 4 parallel requests, aggregate tokens per second.
- Rows: decode = short-context decode; decode_2k / decode_32k = decode after a ~2k / ~32k-token context; prefill_Nk = prompt processing of an N k-token prompt; streams4 = 4-stream aggregate decode. A margin within +-3 % (the measured same-runner noise) is a tie.
- llama.cpp ROCm server arguments: `-ngl 0 -c 33792 -np 1 -t 12 -ctk f32 -ctv f32 --jinja`.
- Not supported / not measured: llama.cpp ROCm: not supported: ROCm and Vulkan llama.cpp builds (owner rule); this baseline is the CPU build, 12 threads, f32 KV cache; llama.cpp ROCm: not measured: 32k rows and 4 streams (baseline server has context 33792 and one slot); llama.cpp ROCm: Loom rows are measured on the server (request duration from /metrics), not by a client; llama.cpp rows are its own server timings.
- Measured on AMD RYZEN AI MAX+ 395 w/ Radeon 8060S, AMD Radeon 8060S Graphics + AMD Radeon AI PRO R9700, Ubuntu 24.04, a named machine (this GPU has no DevLab runner); both engines ran on it.
- Margin vs best llama.cpp: (Loom / max(ROCm, Vulkan) - 1) × 100 %, using the available measured tokens/s for each row on this GPU. With only one valid backend, that backend is used; with neither, the margin is not measured. Raw backend values and their measurement sources are retained below and in the table.
- Measured 2026-11-21; device AMD RYZEN AI MAX+ 395 w/ Radeon 8060S, AMD Radeon 8060S Graphics + AMD Radeon AI PRO R9700, Ubuntu 24.04; hrx-loom `0.3.0` (hrx-system commit unknown); source: named-machine run.
- Roofline: the hardware ceiling for this GGUF on AMD Ryzen AI Max+ 395 "Strix Halo" 8060S (gfx1151), measured read bandwidth at 230 GB/s and 44 TFLOP/s sustained at 85 W (spec 59), from loom-roofline `roofline.py` (loom-skills `618f89e`); the 4-stream ceiling of this mixture-of-experts model assumes the four streams route to the same experts (the bound a run may not exceed).
- — not measured; n/a not applicable.

### Radeon AI PRO R9700 (gfx1201, 32 GB)

| metric | llama.cpp | Loom | margin vs best llama.cpp | roofline | Loom % of roofline |
|---|---|---|---|---|---|
| decode tok/s, short prompt | 14.9 | 104.8 | +603 % | 154.3 | 68 % |
| decode tok/s, 2k context | 11.3 | 96.0 | +750 % | 145.8 | 66 % |
| decode tok/s, 32k context | — | 81.0 | — | 118.4 | 68 % |
| prefill tok/s, 1k document | 57.6 | 3313.0 | +5652 % | 19306.3 | 17 % |
| prefill tok/s, 2k document | 53.6 | 3813.8 | +7015 % | 20543.5 | 19 % |
| prefill tok/s, 8k document | 48.7 | 3869.4 | +7845 % | 19715.8 | 20 % |
| prefill tok/s, 16k document | 44.6 | 3645.6 | +8074 % | 18918.3 | 19 % |
| prefill tok/s, 32k document | — | 3217.6 | — | 17540.9 | 18 % |
| 4 streams, aggregate decode tok/s | — | 220.2 | — | 587.2 | 37 % |
| speculative decoding, drafted tokens accepted | n/a | n/a | n/a | n/a | n/a |

**Methodology**

- Compared: Loom against llama.cpp ROCm `llama.cpp-cpu/418fe97+PR32704`; the same GGUF, the same machine and session, the same prompts; Loom served with the active wait. Values are the median of 3 sessions (4 streams: median of 3); Loom is timed by the client, llama.cpp by its server.
- Out of the box on both engines: Loom serves the preset shipped in this package; llama.cpp runs its build defaults plus only the server arguments listed below (offload, context and slots). No tuning sweep on either engine.
- Drafting, 1-stream rows (decode, decode_2k, decode_32k, prefill): no drafting on either engine (no MTP head in the GGUF and no drafter published for this exact model). Each engine is loaded once for all these rows.
- 4 streams: both engines loaded with drafting off, 4 parallel requests, aggregate tokens per second.
- Rows: decode = short-context decode; decode_2k / decode_32k = decode after a ~2k / ~32k-token context; prefill_Nk = prompt processing of an N k-token prompt; streams4 = 4-stream aggregate decode. A margin within +-3 % (the measured same-runner noise) is a tie.
- llama.cpp ROCm server arguments: `-ngl 0 -c 33792 -np 1 -t 12 -ctk f32 -ctv f32 --jinja`.
- Not supported / not measured: llama.cpp ROCm: not supported: ROCm and Vulkan llama.cpp builds (owner rule); this baseline is the CPU build, 12 threads, f32 KV cache; llama.cpp ROCm: not measured: 32k rows and 4 streams (baseline server has context 33792 and one slot); llama.cpp ROCm: Loom rows are measured on the server (request duration from /metrics), not by a client; llama.cpp rows are its own server timings.
- Measured on AMD RYZEN AI MAX+ 395 w/ Radeon 8060S, AMD Radeon 8060S Graphics + AMD Radeon AI PRO R9700, Ubuntu 24.04, a named machine (this GPU has no DevLab runner); both engines ran on it.
- Margin vs best llama.cpp: (Loom / max(ROCm, Vulkan) - 1) × 100 %, using the available measured tokens/s for each row on this GPU. With only one valid backend, that backend is used; with neither, the margin is not measured. Raw backend values and their measurement sources are retained below and in the table.
- Measured 2026-11-21; device AMD RYZEN AI MAX+ 395 w/ Radeon 8060S, AMD Radeon 8060S Graphics + AMD Radeon AI PRO R9700, Ubuntu 24.04; hrx-loom `0.3.0` (hrx-system commit unknown); source: named-machine run.
- Roofline: the hardware ceiling for this GGUF on AMD Radeon AI PRO R9700 (gfx1201, 32 GB), measured read bandwidth at 620 GB/s and 191 TFLOP/s, from loom-roofline `roofline.py` (loom-skills `618f89e`); the 4-stream ceiling of this mixture-of-experts model assumes the four streams route to the same experts (the bound a run may not exceed).
- — not measured; n/a not applicable.

## Features

| feature | status | detail |
|---|---|---|
| API | ✅ | OpenAI-style `/v1/chat/completions` |
| tool calling | ✅ | OpenAI-style `tools` and `tool_calls` |
| reasoning / thinking | ✅ | always on |
| reasoning effort | ✅ | `reasoning_effort`: none, minimal, low, medium, high, xhigh, max (default high) |
| speculative decoding | — | not applicable |
| image input | ❌ | not supported by Loom (the upstream model has it) |
| audio input / output | — | not applicable |
| sampling | ❌ | greedy decoding only |
| context size | ✅ | 262,144 tokens per request |
| concurrency | ✅ | 4 concurrent requests |
| max output tokens | ✅ | 65,536 (preset `max_tokens`) |

## Correctness

| gate | description | Strix Halo | Radeon AI PRO R9700 |
|---|---|---|---|
| `kernel_tests` | Unit tests of the generated kernels against reference implementations | ✅ pass (34/34) ¹ | ✅ pass (34/34) ¹ |
| `golden` | llama.cpp reference recorded on the same GGUF: greedy tokens and top-5 log-probs, short prompts and 1k-32k documents | ✅ pass (from d08221b, not re-run on this build) ² | ✅ pass (from d08221b, not re-run on this build) ² |
| `compare` | Greedy output vs the llama.cpp reference, token by token; every divergence is a near-tie | ✅ pass (from d08221b, not re-run on this build) ² | ✅ pass (from d08221b, not re-run on this build) ² |
| `vision` | Image input vs llama.cpp with the same projector: identical prompt tokens, replies equal up to near-ties | — not applicable | — not applicable |
| `consistency` | Concurrent requests give the same output as the same requests run alone | ✅ pass ² | ✅ pass ² |
| `spec` | Speculative decoding on gives exactly the output of speculative decoding off | — not applicable | — not applicable |
| `agent` | Multi-turn tool-calling loop: exact prompt rendering, history reuse, valid tool calls, output vs llama.cpp | ✅ pass (tool messages sent with name) ² | ✅ pass (tool messages sent with name) ² |
| `agent_think_off` | The tool-calling loop with thinking off | — not applicable | — not applicable |
| `sampling` | temperature, top_k and seed behave as specified; first-token distribution matches llama.cpp | ❌ fail (default temperature 1.0, check expects greedy) ² | ❌ fail (default temperature 1.0, check expects greedy) ² |
| `logits` | Host logits readback (llm.logits on): at T=0 every returned row's argmax is the device's token, the top-k matches llama.cpp, greedy text is unchanged | — not applicable | — not applicable |
| `decision` | Decision answers (choice, score, noul probabilities) vs llama.cpp on the same prompts | — not applicable | — not applicable |

- ¹ named-machine run, 2026-11-26; device AMD RYZEN AI MAX+ 395 w/ Radeon 8060S, AMD Radeon 8060S Graphics + AMD Radeon AI PRO R9700, Ubuntu 24.04; hrx-loom `0.3.0` (hrx-system commit unknown).
- ² named-machine run, LEVEL=dev, 2026-11-26; device AMD RYZEN AI MAX+ 395 w/ Radeon 8060S, AMD Radeon 8060S Graphics + AMD Radeon AI PRO R9700, Ubuntu 24.04; hrx-loom `0.3.0` (hrx-system commit unknown).
- Gates are the loom-benchmark correctness gates (`gates.sh`); a gate without a recorded run shows no evidence.

## Files

- `model.json` / `model.json.lock`: the card and the pinned source files (sha256 checked by `loom-fetch`).
- `program/*.loom`, `weights.loom`: the generated Loom programs.
- `tokenizer.json`: the tokenizer.
- `llm_config.json`: the configuration the programs were generated from.
- `info.json`: the measurements, gate evidence and versions this card shows; everything else comes from the files above.
- `REPORT.md`: how the values above were measured: setup, commands and the value of every run.

Versions: card schema `loom-card/3`; loom-skills `618f89e`; hrx-loom `0.3.0` (hrx-system commit unknown); card generator `618f89e`.

Source files, fetched by `loom-fetch` from their own repositories:

- checkpoint: [unsloth/Inkling-2-Mini-GGUF](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF) @ `ea0c79b`, `Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf_file`
