# Performance, Q4_K_XL, final build: wave 2 + chat fixes + grouped projection grid (agent integ3/integ4, 2026-11-24, bead llama-cpp-tg3.28)

Build: `package/` = wave 2 (M30) + CHATFIX + CMDPROG (the grouped attention projection GEMV has its own workgroup-per-unit value: 7 on gfx1201, 4 on gfx1151; `docs/OPT_CMDPROG.md`). hrx-loom 0.3.0. Presets `chat-r9700` and `chat`, context 131072, rows 4, prefill 2048, prefix cache 16.

R9700: 3 fresh sessions of `perf_session.sh r9700 <n> spin full` on the final published build (one GPU at a time, nothing else running on the GPUs; the CPU oracle was stopped before). 8060S: the evidence of the previous table below is kept, because the 8060S code path is identical (tune value stays 4); one sanity session of the final build gave decode 42.73 / 39.3 / 32.27 tok/s (short, 2k, 32k), prefill 942 / 1103 / 1124 / 1044 / 878, 4 streams 93.9, decode 8k 37.62: all within the spread of the kept table (4 streams -1.9 percent, noise).

## R9700 (gfx1201, chat-r9700), median of 3 sessions, tok/s

| Row | Final build | Raw (sessions 1, 2, 3) | Previous commit (CHATFIX build) | Change |
|---|---|---|---|---|
| decode, short | 97.8 | 98.08, 97.76, 97.44 | 95.1 | +2.8 % |
| decode, 2k | 90.0 | 90.08, 89.95, 89.36 | 87.7 | +2.6 % |
| decode, 8k (extra) | 86.7 | 86.73, 86.75, 86.02 | 84.5 | +2.7 % |
| decode, 32k | 76.3 | 76.28, 76.47, 76.02 | 74.8 | +2.0 % |
| prefill 1k | 2670.0 | 2735.0, 2670.0, 2666.4 | 2725.0 | -2.0 % |
| prefill 2k | 3132.1 | 3200.9, 3132.1, 3126.0 | 3196.5 | -2.0 % |
| prefill 8k | 3239.3 | 3303.6, 3239.3, 3234.6 | 3297.2 | -1.8 % |
| prefill 16k | 2932.0 | 2981.9, 2932.0, 2928.0 | 2972.2 | -1.4 % |
| prefill 32k | 2342.0 | 2361.9, 2342.0, 2339.3 | 2338.6 | +0.1 % |
| 4 streams, aggregate | 226.8 | 229.7, 226.8, 224.7 | 225.3 | +0.7 % |

Decode gains 2.0 to 2.8 percent (the lane measured +2.3 to +2.6). Prefill reads about 2 percent lower in the median. Session 1 equals the previous table (2735 / 3201 / 3304 / 2982 / 2362) and sessions 2 and 3 are 2 to 3 percent lower: the session-to-session drift of prefill that earlier tables already show (spread up to 3.7 percent). The change touches the decode projection grid only; the prefill difference was not isolated (no ABBA).

Decode short is now 0.63 of the roofline ceiling (154.3). Cold start (3 sessions): ready 2.05 s, first request 3.22 s, second 0.12 s; TTFT at 9k tokens 2.82 s.

Checks of this build: kernel tests 33 of 33 on both GPUs (`bench/integ4/kres`); greedy text (prompts A, B, C, 8k and 32k documents) IDENTICAL to the previous published build on both GPUs; dev gates on both GPUs: consistency, agent, golden, compare pass (R9700 compare 7 full, max gap 0.582; 8060S 9 full, max gap 0.296), sampling fails by design as before; context admission 130,991 tokens completes (175 s R9700, 619 s 8060S).

## 8060S (gfx1151, chat): kept table of the previous build

---

# Previous table (CHATFIX build and M30 build measurements): Performance, Q4_K_XL, wave 2 build on hrx-loom 0.3.0, full skills table (agent integ3, 2026-11-24, bead llama-cpp-tg3.28)

Build: `package/` = wave 1 (`package_T10b`) + D2 (attention front end, grouped projection GEMV) + D6 (experts decode pipelining) + W64 (wave64 down kernel, 8060S only) + D3 (controllers, router), merged by M30 (`docs/OPT_M30.md`, `bench/M30/`). Runtime hrx-loom 0.3.0. Presets `chat-r9700` (gfx1201) and `chat` (gfx1151), context 131072, rows 4, prefill capacity 2048.

Measured with `perf_session.sh <gpu> <n> spin full` (3 fresh servers per GPU, one at a time under `flock -x`, `perf.py` rows, `c4_steady.py`, decode at 8k), the same method and tools as the wave 1 table below. Medians of 3 sessions. The measured programs are the M30 build; the published build adds the chat fixes of lane CHATFIX (prefix cache 16, tolerant tool policy, fork-state fix), whose perf guard showed decode unchanged (94.60 to 94.92 against 94.72 / 94.84 tok/s) and prefill 2k within the session drift (about 2 percent); the full table was not re-measured on the CHATFIX build.

## R9700 (gfx1201, chat-r9700), tok/s

| Row | Median | Raw (sessions 1, 2, 3) | Wave 1 | Change | M30 single session (`ab_session.py`) |
|---|---|---|---|---|---|
| decode, short | 95.1 | 95.43, 95.09, 95.14 | 90.2 | +5.4 % | 94.86 |
| decode, 2k | 87.7 | 87.70, 87.44, 87.75 | 83.2 | +5.4 % | |
| decode, 8k (extra) | 84.5 | 84.46, 84.13, 84.63 | 80.3 | +5.2 % | 84.77 |
| decode, 32k | 74.8 | 74.79, 74.30, 74.81 | 71.3 | +4.9 % | 75.38 |
| prefill 1k (1258 tokens) | 2725.0 | 2725.0, 2717.7, 2725.9 | 2576.3 | +5.8 % | |
| prefill 2k (2454) | 3196.5 | 3196.5, 3190.0, 3200.4 | 2996.0 | +6.7 % | 3704 (other tool) |
| prefill 8k (9083) | 3297.2 | 3297.2, 3295.1, 3307.5 | 3091.5 | +6.7 % | |
| prefill 16k (18436) | 2972.2 | 2972.2, 2969.2, 2984.7 | 2800.9 | +6.1 % | |
| prefill 32k (37007) | 2338.6 | 2338.6, 2337.6, 2345.2 | 2242.0 | +4.3 % | |
| 4 streams, aggregate | 225.3 | 225.0, 227.8, 225.3 | 218.9 | +2.9 % | 213.2 (other tool) |

## 8060S (gfx1151, chat), tok/s

| Row | Median | Raw (sessions 1, 2, 3) | Wave 1 | Change | M30 single session (`ab_session.py`) |
|---|---|---|---|---|---|
| decode, short | 42.6 | 42.82, 42.65, 42.49 | 39.7 | +7.3 % | 42.80 |
| decode, 2k | 39.4 | 39.38, 39.40, 39.00 | 36.7 | +7.4 % | |
| decode, 8k (extra) | 37.6 | 37.57, 37.64, 37.34 | 34.9 | +7.7 % | 37.81 |
| decode, 32k | 32.3 | 32.33, 32.32, 32.05 | 29.7 | +8.8 % | 32.44 |
| prefill 1k (1258 tokens) | 937.1 | 964.1, 937.1, 914.9 | 893.7 | +4.9 % | |
| prefill 2k (2454) | 1094.5 | 1114.6, 1094.5, 1066.3 | 1033.1 | +5.9 % | 1221 (other tool) |
| prefill 8k (9083) | 1122.6 | 1140.6, 1122.6, 1092.0 | 1050.7 | +6.8 % | |
| prefill 16k (18436) | 1040.2 | 1050.6, 1040.2, 1012.2 | 973.1 | +6.9 % | |
| prefill 32k (37007) | 867.2 | 885.5, 867.2, 847.3 | 811.8 | +6.8 % | |
| 4 streams, aggregate | 95.7 | 95.7, 95.8, 94.7 | 89.0 | +7.5 % | 88.06 |

The fresh numbers differ from the M30 single-session numbers: decode within 0.6 percent (R9700 short 95.1 against 94.86; 8060S 42.6 against 42.80), 4 streams +5.7 percent on the R9700 (225.3 against 213.2) and +8.7 percent on the 8060S (95.7 against 88.06). Prefill and 4-stream rows of the M30 table come from `ab_session.py`, which measures differently (see the wave 1 notes), so only the same-tool rows of this table are compared with wave 1. Spreads: decode 0.2 to 0.7 %, prefill up to 3.7 % (8060S), 4 streams up to 1.2 %.

Decode against the ceiling (154.3 and 57.2 tok/s at short context): R9700 0.62, 8060S 0.74 (wave 1: 0.58, 0.69).

Cold start (`coldstart.jsonl`, 3 sessions): server start to ready 2.05 to 2.54 s (R9700) and 2.55 s (8060S); first request (weights load) 3.22 s and 0.77 to 0.80 s; second request 0.12 s and 0.27 s. Time to first token at 8k (9083 tokens): 2.82 s (R9700), 8.1 to 8.3 s (8060S). Kernel cache warm.

Greedy text of the M30 build equals wave 1 (A, B, C, 8k, 32k, both GPUs). Kernel tests 33 of 33. Dev gates (M30 build): consistency, agent, golden and compare pass; sampling fails by design. Gates of the CHATFIX build: `docs/OPT_CHATFIX.md`.

---

# Wave 1 baseline (kept): Performance baseline, Q4_K_XL, skills method: wave 1 build (re-baseline, agent integ2, 2026-11-22, bead llama-cpp-tg3.30)

Build: `package/` = wave 1 (`package_T10b` = base + D1 decode residual-chain fusion + D4 long-context decode attention + D5 lm_head and sampler path + P1 larger-n experts pass), published tree rebuilt with `prefill_capacity` 2048 for the Q4 presets (`programs_sha256` 160c0325...). Presets `chat-r9700` (gfx1201) and `chat` (gfx1151), context 131072, rows 4. The method is the one of the baseline below, unchanged: `scripts/perf_session.sh <gpu> <n> spin full` under `flock -x ~/hrx-pr1/gpu.lock`, 3 fresh serve sessions per GPU (`loom-serve --device='amdgpu://<URI>?completion_spin_ns=4294967295' --preset=<P> --pool_backing=fixed --workers=8`), one server at a time, idle box (nothing else ran: the CPU llama.cpp server was stopped before, kernel tests and the context table ran before the sessions), weights hard-linked, kernel cache warm, label `loom-baseline`, `--thinking off`, medians of 3 session values. The CPU llama.cpp rows are the ones of the baseline below (the build and the oracle did not change: they were not run again). The roofline ceilings were recomputed for the Q4 prefill chunk of 2048 tokens (`docs/ROOFLINE.md`), so the prefill efficiencies below use new ceilings; decode ceilings are unchanged.

Result in one line: decode +5 percent on the R9700 (85.9 to 90.2) and +3 percent on the 8060S (38.6 to 39.7) at short context, +6 and +8 percent at 32k; prefill +28 to +63 percent on the R9700 and +12 to +37 percent on the 8060S (the largest gains at 8k to 32k); 4 streams +2 and +3 percent.

Compared with the end-to-end ABBA of `docs/OPT_WAVE1.md` (decode short +5.3 / +2.3 percent, prefill 2k +61 / +33 percent): the decode numbers agree. The prefill 2k rows here are smaller (+44 / +26 percent) because `perf.py` measures a 2454 token request that includes decode steps and a short last chunk (the baseline already noted that perf.py prefill rows are 6 to 7 percent below a `max_tokens` 1 rate); the ABBA used 2040 token prompts.

## R9700 (gfx1201, chat-r9700), median of 3 sessions, tok/s

| Row | Median (new) | Raw runs (sessions 1, 2, 3) | Spread | Before wave 1 | Change | Ceiling | Efficiency | Of target |
|---|---|---|---|---|---|---|---|---|
| decode, short prompt | 90.2 | 90.45, 89.89, 90.18 | 0.6 % | 85.9 | +5.0 % | 154.3 | 0.58 | 0.75 |
| decode, 2k context | 83.2 | 83.61, 82.77, 83.24 | 1.0 % | 79.3 | +4.9 % | 145.8 | 0.57 | 0.72 |
| decode, 8k context (extra) | 80.3 | 80.32, 79.86, 80.37 | 0.6 % | 76.3 | +5.2 % | 139.4 | 0.58 | 0.72 |
| decode, 32k context | 71.3 | 71.34, 71.09, 71.36 | 0.4 % | 67.0 | +6.4 % | 118.4 | 0.60 | 0.74 |
| prefill 1k (1258 tokens) | 2576.3 | 2647.8, 2576.3, 2572.4 | 2.9 % | 2014.2 | +27.9 % | 19306.3 | 0.13 | 0.40 |
| prefill 2k (2454) | 2996.0 | 3071.2, 2996.0, 2989.6 | 2.7 % | 2074.7 | +44.4 % | 20543.5 | 0.15 | 0.44 |
| prefill 8k (9083) | 3091.5 | 3151.9, 3091.5, 3087.2 | 2.1 % | 1919.8 | +61.0 % | 19715.8 | 0.16 | 0.48 |
| prefill 16k (18436) | 2800.9 | 2846.8, 2800.9, 2798.0 | 1.7 % | 1714.0 | +63.4 % | 18918.3 | 0.15 | 0.45 |
| prefill 32k (37007) | 2242.0 | 2255.6, 2239.7, 2242.0 | 0.7 % | 1400.8 | +60.1 % | 17540.9 | 0.13 | 0.39 |
| 4 streams, aggregate decode | 218.9 | 218.9, 218.9, 219.7 | 0.4 % | 214.2 | +2.2 % | 587.2 | 0.37 | 0.44 |

## 8060S (gfx1151, chat), median of 3 sessions, tok/s

| Row | Median (new) | Raw runs (sessions 1, 2, 3) | Spread | Before wave 1 | Change | Ceiling | Efficiency | Of target |
|---|---|---|---|---|---|---|---|---|
| decode, short prompt | 39.7 | 39.7, 39.65, 39.66 | 0.1 % | 38.6 | +2.8 % | 57.2 | 0.69 | 0.89 |
| decode, 2k context | 36.7 | 36.56, 36.72, 36.72 | 0.4 % | 35.5 | +3.4 % | 54.1 | 0.68 | 0.86 |
| decode, 8k context (extra) | 34.9 | 34.94, 34.89, 34.93 | 0.1 % | 33.2 | +5.1 % | 51.7 | 0.68 | 0.85 |
| decode, 32k context | 29.7 | 29.67, 29.66, 29.64 | 0.1 % | 27.5 | +8.0 % | 43.9 | 0.68 | 0.83 |
| prefill 1k (1258 tokens) | 893.7 | 899.8, 891.9, 893.7 | 0.9 % | 798.9 | +11.9 % | 4792.4 | 0.19 | 0.42 |
| prefill 2k (2454) | 1033.1 | 1046.5, 1033.1, 1021.1 | 2.5 % | 822.6 | +25.6 % | 4701.2 | 0.22 | 0.49 |
| prefill 8k (9083) | 1050.7 | 1059.1, 1050.7, 1032.2 | 2.6 % | 785.7 | +33.7 % | 4511.8 | 0.23 | 0.52 |
| prefill 16k (18436) | 973.1 | 979.0, 973.1, 952.9 | 2.7 % | 717.7 | +35.6 % | 4329.3 | 0.22 | 0.51 |
| prefill 32k (37007) | 811.8 | 817.9, 811.8, 795.4 | 2.8 % | 594.6 | +36.5 % | 4014.1 | 0.20 | 0.45 |
| 4 streams, aggregate decode | 89.0 | 88.9, 89.0, 92.4 | 3.9 % | 86.6 | +2.8 % | 217.8 | 0.41 | 0.48 |

"Before wave 1" is the median of the baseline below (same method, 512 token chunks). Efficiency = measured / ceiling of `bench/<gpu>/roofline.json` (new Q4 prefill ceilings for 2048 token chunks; the old prefill ceiling at 512 was 9.8k tok/s on the R9700, so the old efficiencies read 0.14 to 0.21; the new ones read 0.13 to 0.16 against the compute-bound ceiling of about 20k). "Of target" = measured / (ceiling x skills default efficiency). `perf.py` still prints REVIEW on every row (no optimization history between sessions, 1 h default budget); `targets.jsonl` was logged against the campaign view of the earlier ceilings, `roofline_campaign_view.json` is the new one.

Noise: decode rows 0.1 to 1.0 % between sessions, prefill 0.7 to 2.9 %, 4 streams 0.4 % (R9700) and 3.9 % (8060S, one session 3.9 percent above the other two). The slow drift of prefill over the sessions seen in the baseline is mostly absent: R9700 prefill 1k 2648, 2576, 2572 (a 2.7 percent step after session 1), 8060S 900, 892, 894.

## Time to first token and cold start

TTFT of the prefill rows (prompt tokens / rate), seconds, before in brackets:

| Prompt tokens | 1258 | 2454 | 9083 | 18436 | 37007 |
|---|---|---|---|---|---|
| R9700 | 0.49 (0.62) | 0.82 (1.18) | 2.94 (4.73) | 6.58 (10.76) | 16.5 (26.4) |
| 8060S | 1.41 (1.57) | 2.38 (2.98) | 8.64 (11.56) | 18.9 (25.7) | 45.6 (62.2) |

Full context (131072): a request of 130972 tokens completes in 120 s on the R9700 (183 s before) and 343 s on the 8060S (443 s before), `docs/CONTEXT_TABLE.md`.

Start-up per serve session (`coldstart.jsonl`, 3 spin sessions): server start to registration ready 1.5 to 2.1 s (R9700) and 2.6 to 3.1 s (8060S); first request (weights load) 3.22 s and 0.81 to 0.83 s; second request 0.13 s and 0.30 s: the same as before. The kernel cache was warm (the programs had been compiled by the gate run before), so the cold JIT compile of the new programs was not measured. Lane P1 measured the stage compile on an empty cache growing with the capacity (41 to 75 s on the R9700 and 75 to 135 s on the 8060S, summed over the workers), a cost of the first start after a program change only.

## Evidence files

`bench/<gpu>/ev_perf.json` (made by `scripts/perf_evidence.sh`: `make_info.py perf --source-kind machine --baseline-version "llama.cpp-cpu/418fe97+PR32704"`, CPU rows reused from the baseline), `perf_log.jsonl`, `c4_log.jsonl`, `targets.jsonl`, `extra_log.jsonl` (8k decode), `coldstart.jsonl`, `roofline.json`, `ev_roofline_*.json`, `roofline_campaign_view.json`, `perf_raw/spin-{1,2,3}/` (the other folders of `perf_raw` are from the baseline). The files of the baseline are kept with the suffix `_prewave1` (`ev_perf_prewave1.json`, `perf_log_prewave1.jsonl`, ...). The bare, blocking and ctx32k arms were not repeated (wait mode and context capacity did not matter before).

Command: `flock -x ~/hrx-pr1/gpu.lock bash scripts/perf_session.sh <r9700|strix> <1|2|3> spin full` (box dir with `bench-<gpu>/`, `scripts/perf_decode_ctx.py`, `mat-<gpu>` = the fetched preset), then `bash scripts/perf_evidence.sh <gpu> bench-<gpu> mat-<gpu> $PWD/bench-<gpu>/ev_perf.json`.

---

# Previous baseline before wave 1: performance baseline, Q4_K_XL, skills method (T6, agent inkU, 2026-11-21, bead llama-cpp-tg3.20)

Tree: `publish/unsloth-Inkling-2-Mini-loom/` (refreshed by inkS), presets `chat-r9700` (gfx1201) and `chat` (gfx1151), context 131072, rows 4, prefill 512. Same programs as the gates (`programs_sha256` 51c67d50...). Box idle at the start of each session (only my own processes; no other server; load average about 1 is the tail of my previous session). Every timed session ran under `flock -x ~/hrx-pr1/gpu.lock`, one server at a time, stopped by PID. Weights hard-linked (`loom-fetch --materialize=share`), no copy. Kernel cache warm (see Cold start).

## Method

- Server: `loom-serve --device='amdgpu://<URI>?completion_spin_ns=4294967295' --model=<preset dir> --preset=<P> --pool_backing=fixed --listen=127.0.0.1:<port> --workers=8` (the skills' `serve.sh` flags).
- Per serve session (3 sessions per GPU, one fresh server each): first request (cold start), `perf.py` (decode 300 tokens, prefill 1k and 2k documents, decode 256 after 2k), `perf.py --long` (prefill 8k, 16k, 32k documents, decode 128 after 32k), `c4_steady.py 4`, decode at 8k (my `scripts/perf_decode_ctx.py`, same rules as perf.py, not a skills metric). Label `loom-baseline`; `{"session_end"}` markers separate the sessions in `perf_log.jsonl` and `c4_log.jsonl`, so `make_info.py perf` takes the median of 3 session values.
- `--thinking` was left at the default `off`: every request carries `chat_template_kwargs.enable_thinking=false`, which our policy maps to reasoning effort `none`. All decode requests produced the full max_tokens (300/300, 256/256, 128/128, 400 per stream in c4_steady; `perf.py` rejects a run below half), so no decode row is a short-request artefact. `--repeat-rows` stays 1 (decode is far below 100 tok/s).
- Prompt sizes: the skills' "1k, 2k, 8k, 16k, 32k" documents are 1258, 2454, 9083, 18436 and 37007 tokens with our tokenizer (all inside 131072). Prefill is chunked in 512 token stages by the service.
- Campaign view: `perf.py` (without `--no-campaign`) needs a skills-schema `roofline.json` next to the log. Ours (`inkling-roofline/1`) is not that schema, so `scripts/roofline_campaign_view.py` writes a view (`bench/<gpu>/roofline_campaign_view.json`, metric names of the skills) from `bench/<gpu>/roofline.json`; perf.py then logs `targets.jsonl` and prints a verdict. Verdict on every row: REVIEW ("ETA to target exceeds the 44 min left at this rate": there is no optimization history between the sessions, the 1 h default budget is meaningless here). No row is DONE or BUDGET. The efficiency columns below are what matters.
- Efficiency = measured / ceiling of `roofline.json` (`docs/ROOFLINE.md`); "of target" = measured / (ceiling x skills default efficiency). The ceilings are for 1000/2000/8000/32000 token contexts and the measured contexts are about 10 to 25 percent longer (e.g. 37007 tokens at "32k"), so the longer rows read about 1 to 3 percent worse than the exact ceiling would give. Prefill ceilings are memory bound (31 GB of weights per 512 token chunk).

## R9700 (gfx1201, chat-r9700), median of 3 sessions, tok/s

| Row | Median | Raw runs (session 1, 2, 3) | Spread | Ceiling | Efficiency | Of target |
|---|---|---|---|---|---|---|
| decode, short prompt | 85.9 | 85.88, 86.04, 85.69 | 0.4 % | 154.3 | 0.56 | 0.71 |
| decode, 2k context | 79.3 | 79.74, 79.33, 79.22 | 0.7 % | 145.8 | 0.54 | 0.69 |
| decode, 8k context (extra) | 76.3 | 76.39, 76.33, 76.30 | 0.1 % | 139.4 | 0.55 | 0.69 |
| decode, 32k context | 67.0 | 67.19, 66.95, 67.03 | 0.4 % | 118.4 | 0.57 | 0.69 |
| prefill 1k (1258 tokens) | 2014 | 2057.6, 2014.2, 1993.9 | 3.2 % | 9772 | 0.21 | 0.62 |
| prefill 2k (2454) | 2075 | 2128.9, 2074.7, 2066.6 | 3.0 % | 9786 | 0.21 | 0.64 |
| prefill 8k (9083) | 1920 | 1961.3, 1919.8, 1919.4 | 2.2 % | 9769 | 0.20 | 0.60 |
| prefill 16k (18436) | 1714 | 1739.6, 1714.0, 1713.4 | 1.5 % | 9730 | 0.18 | 0.53 |
| prefill 32k (37007) | 1401 | 1410.5, 1399.6, 1400.8 | 0.8 % | 9805 | 0.14 | 0.43 |
| 4 streams, aggregate decode | 214.2 | 214.2, 214.9, 209.8 | 2.4 % | 587.2 | 0.36 | 0.43 |

Per row each session value is one run (perf.py median of 5 for the 1k and 2k prefill samples inside the run, min/max within a run 0.1 to 0.3 %).

## 8060S (gfx1151, chat), median of 3 sessions, tok/s

| Row | Median | Raw runs (session 1, 2, 3) | Spread | Ceiling | Efficiency | Of target |
|---|---|---|---|---|---|---|
| decode, short prompt | 38.6 | 38.78, 38.53, 38.65 | 0.6 % | 57.2 | 0.68 | 0.87 |
| decode, 2k context | 35.5 | 35.51, 35.48, 35.67 | 0.5 % | 54.1 | 0.66 | 0.83 |
| decode, 8k context (extra) | 33.2 | 33.11, 33.23, 33.49 | 1.1 % | 51.7 | 0.64 | 0.81 |
| decode, 32k context | 27.5 | 27.08, 27.54, 27.66 | 2.1 % | 43.9 | 0.63 | 0.77 |
| prefill 1k (1258 tokens) | 799 | 818.8, 798.9, 779.4 | 4.9 % | 3625 | 0.22 | 0.50 |
| prefill 2k (2454) | 823 | 836.8, 822.6, 802.6 | 4.2 % | 3630 | 0.23 | 0.51 |
| prefill 8k (9083) | 786 | 794.3, 785.7, 765.7 | 3.6 % | 3624 | 0.22 | 0.49 |
| prefill 16k (18436) | 718 | 722.2, 717.7, 700.1 | 3.1 % | 3610 | 0.20 | 0.45 |
| prefill 32k (37007) | 595 | 594.6, 597.4, 583.6 | 2.3 % | 3631 | 0.16 | 0.37 |
| 4 streams, aggregate decode | 86.6 | 86.6, 85.6, 87.0 | 1.6 % | 217.8 | 0.40 | 0.47 |

Noise: decode rows 0.1 to 2 % between sessions (the 8060S 32k row is the widest, 2.1 %); prefill 1 to 5 %, larger on the 8060S, and the 1k and 2k prefill values fall from session 1 to session 3 on both GPUs (R9700 2058, 2014, 1994; 8060S 819, 799, 779; a drift of 3 to 5 percent over about 30 minutes, cause not found: thermal or clock state is possible); 4 streams 1.6 to 2.4 %. The skills' own noise band is 2 %.

## Time to first token and cold start

TTFT of the prefill rows (prompt tokens / rate, from the medians above), seconds:

| Prompt tokens | 1258 | 2454 | 9083 | 18436 | 37007 |
|---|---|---|---|---|---|
| R9700 | 0.62 | 1.18 | 4.73 | 10.76 | 26.4 |
| 8060S | 1.57 | 2.98 | 11.56 | 25.7 | 62.2 |

Short prompts (extra check, `scripts/perf_prefill_n.py`, median of 5): 145 tokens 0.155 s (R9700) and 0.381 s (8060S); 491 tokens 0.231 s and 0.601 s.

Start-up per serve session (`coldstart.jsonl`, 3 spin sessions each, 5 to 8 with the other arms):

| | Server start to registration ready | First request ("Say hi.", 8 tokens; weights activation) | Second request |
|---|---|---|---|
| R9700 | 1.0 to 1.5 s | 3.23 s (3.226 to 3.232) | 0.13 s |
| 8060S | 2.0 s | 0.82 to 0.85 s | 0.30 s |

The kernel cache was warm in all sessions (every program had been compiled by earlier agents), so the cold JIT compile was not measured; INTEGRATION_LOG gives 60 to 100 s for a first compile of the whole package with an empty cache. The weights (17.6 GB) are loaded at the first request. The first request costs the same in every session, so it is the load time, not noise.

## Wait mode (R9700, cheap A/B; 8060S bare arm for one session)

| Arm | Sessions | decode | decode 2k | prefill 1k | prefill 2k | 4 streams |
|---|---|---|---|---|---|---|
| spin (`completion_spin_ns=4294967295`, baseline) | 3 | 85.9 | 79.3 | 2014 | 2075 | 214.2 |
| bare URI (SDK default, active 1 s window) | 3 | 86.0 | 79.8 | 1996 | 2067 | 210.8 |
| `--wait_mode=blocking` (skills' no-active-wait arm) | 1 | 85.2 | 78.9 | 1987 | 2059 | 211.0 |
| 8060S spin | 3 | 38.6 | 35.5 | 799 | 823 | 86.6 |
| 8060S bare URI | 1 | 38.5 | 35.5 | 778 | 804 | 85.0 |

Active waiting does not matter for us on the R9700: decode equal, prefill and 4 streams within 1 percent, blocking costs 1 to 2 percent at most. On the 8060S the bare arm is within noise for decode (0.3 %) and about 2.6 % lower for prefill and 1.9 % for 4 streams (one session; arrange more sessions before treating it as real). The spin value stays in the baseline as the skills do it. Files: `perf_log_bare.jsonl`, `perf_log_blocking.jsonl` (labels `loom-baseline-bare`, `loom-baseline-blocking`, `--no-campaign`).

## Consistency with INTEGRATION_LOG (first measurements)

| | This baseline | INTEGRATION_LOG | Difference |
|---|---|---|---|
| R9700 decode, short | 85.9 | 86.6 (ctx 16) | -0.8 % (noise) |
| R9700 decode, 8k | 76.3 | 74.9 (ctx 8005) | +1.9 % |
| R9700 prefill, 491 tokens | 2124 | 2202 (520) | -3.5 % |
| R9700 prefill, 2454 / 2053 tokens | 2075 | 2257 (2053) | -8 %, but 20 percent more tokens |
| 8060S decode, short | 38.6 | 38.9 | -0.8 % (noise) |
| 8060S decode, 8k | 33.2 | 33.3 | -0.3 % |
| 8060S prefill, 491 tokens | 817 | 900 (520) | -9 % |
| 8060S prefill, 2454 / 2053 tokens | 823 | 909 (2053) | -9 % |

Decode agrees within noise. Prefill is 3.5 percent (R9700) and 9 percent (8060S) lower than the first measurements, outside the 1 to 5 percent noise. Checked and ruled out: context capacity 131072 against 32768 (a copy of the preset with `context_capacity` 32768, one session per GPU: 8060S prefill 814 / 830, decode 38.7, 4 streams 87.7; R9700 2053 / 2128, decode 86.0, 4 streams 215.8: equal within noise; `perf_log_ctx32k.jsonl`), the wait mode (table above), pool backing (8060S elastic `--memory_bytes=40gib`, 491 tokens: 819 against 817 fixed), host load (idle box; the readings are stable session to session). Not checked: the old `package/` route of inkK in its own directory (not touched), and the state of the 8060S power mode (not recorded). The first measurements were taken on an earlier program state (before inkQ's chat policy change and the publish rebuild) and with a different prompt (prompt A/B plus chat template); a program-level cause is possible but not shown. The prefill ratio to the roofline (0.20 to 0.23) is therefore the current figure; the figure in ROOFLINE.md (0.31 R9700 and 0.34 8060S at 520 tokens) used the older rates and a ceiling at 520 tokens that includes the 8-token second chunk.

## CPU reference row (not a competitor)

`llama.cpp` mainline 418fe97 + PR 32704, CPU build (no ROCm, no Vulkan; owner rule), 12 threads, `llama-server -ngl 0 -c 33792 -np 1 -t 12 -ctk f32 -ctv f32 --jinja`, Q4_K_XL, `perf.py --kind llama --repeats 3` (server-side `timings`), 3 runs on one server session plus one `--long` run, under the GPU lock (nothing ran on the GPUs). Label `llama-cpu-baseline`. This is a reference for the numerics (the oracle), not the card's competitor: there is no GPU llama.cpp baseline.

| Row | Median | Runs |
|---|---|---|
| decode, short | 14.9 | 15.21, 14.86, 14.26 |
| decode, 2k | 11.3 | 11.18, 11.28, 11.26 |
| prefill 1k (1260 tokens) | 57.6 | 57.6, 57.6, 55.6 |
| prefill 2k (2456) | 53.6 | 55.3, 53.6, 53.4 |
| prefill 8k (9085) | 48.7 | 48.7 (one run) |
| prefill 16k (18438) | 44.6 | 44.6 (one run) |
| prefill 32k, decode 32k | not measured | refused as over the server context (33792; the 32k document is 37k tokens) |
| 4 streams | not measured | one slot (`-np 1`) |

Loom (R9700) is 5.8 times (decode), 35 times (prefill 1k) and 38 times (prefill 16k) the CPU build; 8060S 2.6, 14 and 16 times. These ratios compare a GPU with a CPU and are not a margin.

## Evidence files

`bench/<gpu>/ev_perf.json` (gpu = radeon-r9700, strix-halo), made by `make_info.py perf --perf-log perf_log.jsonl --loom-label loom-baseline --llama-label llama-cpu-baseline --c4-loom c4_log.jsonl --source-kind machine --gpu <id> --package <fetched preset dir> --baseline-version "llama.cpp-cpu/418fe97+PR32704" --baseline-args "<server args>" --differs ... --command ... -o ev_perf.json`. The free-text version is accepted as given (version string `llama.cpp-cpu/418fe97+PR32704`). The first `--differs` text was refused later by `make_info.py build` ("a caveat is one short factual line", at most 240 characters, one sentence, no history such as the PR number); the stored text is `not supported: ROCm and Vulkan llama.cpp builds (owner rule); this baseline is the CPU build, 12 threads, f32 KV cache` and `not measured: 32k rows and 4 streams (baseline server has context 33792 and one slot)`. Validation: a trial `make_info.py build <fetched dir> --repo local/unsloth-Inkling-2-Mini-loom --upstream none --report generated --gpus strix-halo,radeon-r9700 --evidence bench/radeon-r9700/ev_perf.json bench/strix-halo/ev_perf.json -o <tmp>/info.json` wrote a valid `loom-card/3` info.json (to /tmp, deleted; not part of the tree). `programs_sha256` 51c67d50845729c5... equals the gates evidence. Unverified: the evidence `source.device` names both GPUs (auto-detected on the box) and the 8060S power mode is not recorded; the Loom rows are named-machine evidence (`--source-kind machine`), not DevLab.

Other files per GPU: `perf_log.jsonl` (Loom spin sessions and, in the same file, the CPU llama rows), `c4_log.jsonl`, `targets.jsonl`, `roofline_campaign_view.json`, `extra_log.jsonl` (8k decode, prefill by size), `coldstart.jsonl`, `perf_log_{bare,blocking,ctx32k}.jsonl`, `c4_log_*.jsonl`, `perf_raw/<arm>-<n>/{short,long,c4,d8k}.log, serve.log, preflight.txt` and `perf_raw/cpu/`.

## Commands

```sh
# box, ~/inkling/inkU: fetch the preset (hard links to the GGUF) once per GPU
loom-fetch <publish tree>/model.json --preset chat-r9700 --materialize=share --output mat-r9700 --origin checkpoint=<snap> --no-input
# one serve session with all timed runs (spin | bare | blocking, full | quick); run under the GPU lock
flock -x ~/hrx-pr1/gpu.lock bash scripts/perf_session.sh r9700 1 spin full      # scripts/perf_run_all.sh = the whole order
python3 scripts/perf_summary.py bench-r9700                                      # medians, raw runs, efficiency
flock -x ~/hrx-pr1/gpu.lock bash scripts/perf_prefill_check.sh r9700             # prefill at 145 / 491 / 1251 tokens
# CPU reference, strictly after the GPU work
flock -x ~/hrx-pr1/gpu.lock bash scripts/perf_cpu.sh
```

Scripts (laptop `scripts/`): `perf_session.sh`, `perf_run_all.sh`, `perf_cpu.sh`, `perf_summary.py`, `perf_decode_ctx.py`, `perf_prefill_n.py`, `perf_prefill_check.sh`, `roofline_campaign_view.py`. The skills tree was not edited (scripts run by path with `$HOME/loom/.tools/bin/python`, `PYTHONDONTWRITEBYTECODE=1`).

## Notes for the optimize loop

- Decode is 0.54 to 0.57 of the ceiling on the R9700 and 0.63 to 0.68 on the 8060S; the target (skills efficiency) is 0.78: 29 to 31 percent short on the R9700 and 13 to 23 percent on the 8060S. The 8060S has the smaller gap because fixed launch costs (about 480 launches per token) matter less when each token takes 26 ms instead of 11.6 ms.
- Prefill is the weakest row (0.14 to 0.23 of the memory ceiling, 0.37 to 0.64 of target) and falls with length on both GPUs (R9700 2075 at 2k to 1401 at 37k): attention over a longer context costs more than the model weights traffic accounts for.
- 4 streams reach 0.36 (R9700) and 0.40 (8060S) of the shared-experts ceiling; against the independent-routing bound (272.5 on the R9700) the R9700 is at 0.79.
