# Per-arch tune route (decision, inkH, 2026-11-20)

## Decision

Arch-dependent choices (tile shapes, thresholds, split counts, matrix-op providers) use route (a): `template.def` providers selected by the device target. Route (b), config values from the card, stays as the override path for values that are not arch data (checkpoint dense type, debug capture) and for experiments.

Both routes were tested in a real serve stage on both GPUs (`dispatch/debug_probes.loom`, capture ids 101 and 102, `scripts/peek_capture.py`).

| Probe | gfx1151 (8060S) | gfx1201 (R9700) |
|---|---|---|
| id 101, template providers `target(@inkling_rdna3)` returns 11, `target(@inkling_rdna4)` returns 12, default returns the config value | 11 | 12 |
| id 102, config value only (`INKLING_TUNE_PROBE=5`, a `config.def` override) | 5 | 5 |
| card route: variant `config` `inkling.dense_type` and program `config` `inkling.capture_id` (`card.json`, preset `capture-q8`) | works | works |

Why (a): the program never asks for the device and never compares an arch string. The server binds the live device to the command target, and the provider whose `amdgpu.target<...>` family matches wins (priority 20 for the arch provider, -1 for the portable default). One package serves both GPUs and any future arch with only the default row. Route (b) would need one preset per arch and a host script to pick it.

## Mechanism for kernel areas

1. Target families come from `dispatch/targets.loom`: `amdgpu.target<gfx11-generic> @inkling_rdna3`, `amdgpu.target<gfx12-generic> @inkling_rdna4`. Libraries with their own families (`lib/inkling/attn_targets.loom`, `dense_targets.loom`) define their aliases in their own files. Link those files and define nothing twice. A file that uses an alias needs `target.decl @alias` (or the definition) in the same file.
2. Declare one template per tune value: `template.decl @inkling.tune.<area>.<key>(<context scalars>) -> (index)`.
3. Provide a portable default with `priority(-1)` and one provider per tuned arch with `target(@inkling_rdna3)` or `target(@inkling_rdna4)`. Context scalars (token capacity, context length) may select among values inside the provider.
4. Apply the template inside a `command.def target(@inkling_execution)` (the visit and hook commands are planned under this target) and pass the result as an exact scalar to the kernel launch. Kernel code does not see the arch.
5. Values that come from the shape bucket (live row count, split counts) are set in `prepare` with `config_i64(stage, key, value)` per stage.
6. Optional override: define `config.def @inkling.tune.<area>.<key> = <default>` and let the default provider return `config.get` of it. A card variant or preset `config` then overrides one value without code changes (tested, id 102).
7. Document each row in the area's `docs/KERNELS_<area>.md` with columns gfx1201, gfx1151, portable default.

Limits found:
- `config.get` keys are stage-global. PR1 kernels read `ggml.mul_mat.weight_type` from config, so one stage can use PR1 `mul_mat` for only one weight type. Use template or launch-scalar parameters (as the inkF and inkD kernels do) for per-weight choices.
- Config keys set by `config_i64` must be declared (`config.decl`) in a linked file, or the key is rejected.
- Target-selected providers are resolved while the command is planned. A wrong or missing alias shows as the portable default, not as an error, so test the selected value on both GPUs (the probe pattern above).

## Values still open for the kernel areas

`split_blocks` of the attention decode (local 1, global 2/4/8 as context grows), `inkling.dense.gemv.cols` per bucket (1 for decode), the experts matrix-path minimum n (48 on both), mix `ctrl_chunk`. Each goes through steps 2 to 5 when the area is integrated.
