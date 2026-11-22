# Running Inkling 2 Mini on loom-serve

The chat server runs on the GPU box (ssh user@gpu-box) from `~/inkling/inkK` (package in `package/`, scripts in `scripts/`). It serves the OpenAI chat API from the model card `package/card.json`.

## Start

```sh
# laptop: copy package/ and scripts/ to the box (once, and after every change)
INKLING_BOX_DIR=inkling/inkK bash scripts/sync.sh

# box: one server per GPU, quantization in the script name (default port 18090, or give a port)
cd ~/inkling/inkK
nohup ./scripts/serve_gfx1201_q4.sh 18090 > serve.log 2>&1 &     # R9700, Q4_K_XL
nohup ./scripts/serve_gfx1201_q8.sh 18090 > serve.log 2>&1 &     # R9700, Q8_K_XL
nohup ./scripts/serve_gfx1151_q4.sh 18091 > serve1151.log 2>&1 & # 8060S, Q4_K_XL
nohup ./scripts/serve_gfx1151_q8.sh 18091 > serve1151.log 2>&1 & # 8060S, Q8_K_XL
```

All four call `scripts/serve_card.sh <arch> <q4|q8> [port] [preset]`. The preset is `interactive-q4` or `interactive-q8` (4 rows, context 32768, prefill 512, 4 checkpoints, max 4096 new tokens; `INKLING_PRESET` or the fourth argument selects another preset of the card, `capture-q8` is the debug route). The GGUF files are read from `~/models/inkling2-mini/Mainline_Support` (`INKLING_MODELS`). Start up takes 2 to 4 s with a warm kernel cache and 1 to 2 minutes for the first compile after code changes; the weights are loaded at the first request (1 to 4 s). From the laptop: `bash scripts/jcard.sh <arch> <port> <q4|q8>` syncs, restarts and waits until the server is ready.

## Use

The server listens on 127.0.0.1 of the box only. Endpoint: `http://127.0.0.1:18090/v1/chat/completions` (OpenAI compatible, streaming and not streaming; `/v1/models`, `/v1/status`, `/tokenize`, `/detokenize` also exist). From the laptop: `ssh -L 18090:127.0.0.1:18090 user@gpu-box`, then use `http://localhost:18090/v1`. Model id `inkling-2-mini`.

```sh
curl -s localhost:18090/v1/chat/completions -H 'Content-Type: application/json' -d '{"model":"inkling-2-mini","max_tokens":256,"temperature":0,"messages":[{"role":"user","content":"Explain in two sentences why the sky is blue."}]}'
```

The model thinks first (default effort `high`): the thinking text is in `reasoning_content`, the answer in `content`; give `max_tokens` enough room. Request fields: `temperature` (default 1.0), `top_p`, `top_k`, `seed`, `max_tokens`, `stream`, `tools`, `tool_choice`, `reasoning_effort` (see docs/CHAT_POLICY.md; unsupported fields are rejected).

## Stop

```sh
kill $(cat ~/inkling/inkK/run-card-<arch>-<q4|q8>/pid)
```

## Limits

- One server per GPU. The Q8_K_XL weights (22.9 GB) plus the transient buffers (about 6.7 GB) fill the R9700; a second model on the same GPU fails with `RESOURCE_EXHAUSTED` and the model stays unavailable until the server restarts.
- `prefill_capacity` must stay at 512 or lower: the expert routing kernel supports at most 512 tokens per stage, and a larger stage hangs the GPU for minutes (the next start then fails with `DATA_LOSS` for about two minutes). Longer prompts are split into 512 token chunks by the service.
- At most 256 spans per stage (attention kernels); rows 4.
- Do not run two jobs for the same GPU and port; use `flock -x ~/hrx-pr1/gpu.lock` around timed runs and `flock -s` around other GPU work (shared box).
- Near ties in the expert routing and the output can flip against the llama.cpp oracle; greedy text is equal for many tokens, then may differ at a tie.
- Numbers (tokens/s) are in docs/INTEGRATION_LOG.md (first measurements, not tuned).

## Publish layout (loom-fetch and loom-serve --model=DIR)

`publish/unsloth-Inkling-2-Mini-loom/` is the published layout (card `model.json`, `model.json.lock`, `llm_config.json`, `program/`, `weights.loom`, `tokenizer.json`). The GGUF weights are not in it: the card names them as `unsloth/Inkling-2-Mini-GGUF` at revision `ea0c79b2ecd73669e774c8519b79bfabfddbc027`. Presets: `chat` (Q4_K_XL, Radeon 8060S), `chat-r9700` (Q4_K_XL, R9700), `chat-q8`, `chat-q8-r9700` (Q8_K_XL). All four use rows 4, prefill 512; context 131072 for the Q4 presets (`chat`, `chat-r9700`, measured in docs/CONTEXT_TABLE.md) and 32768 for the Q8 presets (not yet measured). The old `package/` route above stays unchanged.

```sh
# build the tree (box; generated, never edit): needs cfg.json of the skills' inspect_gguf.py and the skills checkout
python3 scripts/publish_build.py --cfg cfg.json --skills <loom-skills dir> --gguf-dir <dir with the GGUF files>
bash scripts/scrub_check.sh                                   # no host, user or path names in the tree
# local snapshot of the checkpoint origin: <snap>/Mainline_Support/<name>.gguf_file -> the local GGUF file
bash scripts/publish_lock.sh publish/unsloth-Inkling-2-Mini-loom <snap>     # writes model.json.lock

# fetch one preset into an empty directory and serve it (one GPU at a time)
loom-fetch publish/unsloth-Inkling-2-Mini-loom/model.json --preset chat-r9700 --materialize=copy --output <dir> --origin checkpoint=<snap> --no-input
scripts/serve_publish.sh gfx1201 <dir> chat-r9700 18190        # loom-serve --device=<URI> --model=<dir> --preset=chat-r9700 --pool_backing=fixed
python3 scripts/route_check.py 18190 thinkingmachines/Inkling-2-Mini out.json
```

A materialized directory keeps only the selected preset, so fetch with the preset that you serve. Without `--origin` the files come from Hugging Face. The model id of this route is `thinkingmachines/Inkling-2-Mini` (the old route used `inkling-2-mini`). The device URIs are in docs/DEVICES.md.
