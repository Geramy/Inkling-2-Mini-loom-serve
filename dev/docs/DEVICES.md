# GPU devices on the box (gpu-box)

`loom-run --list_devices=amdgpu` prints two URIs. They map to the GPUs as follows.

| URI | GPU | Target | Memory | Notes |
|---|---|---|---|---|
| `amdgpu://id/000000e2000000808e270f2b0bbb4e00` | Radeon 8060S (Strix Halo iGPU) | gfx1151, JIT target `amdgpu-rdna3-5` | Unified: 123 GiB host RAM, 2 GiB VRAM carve-out | Selected by device id. KFD `unique_id` is 0, so no `GPU-` UUID exists. |
| `amdgpu://GPU-e0880c05003c324d` | Radeon AI PRO R9700 (Thunderbolt 5 eGPU) | gfx1201 (RDNA4) | 34.2 GB VRAM (`rocm-smi`: 34,208,743,424 B) | Selected by KFD UUID. |

In `rocm-smi` the order is reversed: `GPU[0]` is the R9700, `GPU[1]` is the 8060S. Do not map by ordinal. Always use the URIs.

## How this was established

1. `rocminfo`: agent `gfx1201`, `Marketing Name: AMD Radeon AI PRO R9700`, `Uuid: GPU-e0880c05003c324d`. The agent `gfx1151` (`Radeon 8060S Graphics`) has `Uuid: GPU-XX` (no stable UUID).
2. `/sys/class/kfd/kfd/topology/nodes/*/properties`: node 1 has `gfx_target_version 110501`, `unique_id 0`. Node 2 has `gfx_target_version 120001` and `unique_id 16179194876946821709` (= 0xe0880c05003c324d), which is the `GPU-` URI.
3. `loom-serve --device=<URI>` then `GET /v1/status`: `server.resources.devices[0].name` is `AMD Radeon 8060S Graphics` for the `id/` URI and `AMD Radeon AI Pro R9700` for the `GPU-` URI.
4. The SDK saxpy example (`loom-test saxpy.loom --device=<URI>`) passes on both. `loom-run ... --compile-report=text-summary` shows different artifacts per URI (code bytes 140 vs 172, 21 vs 22 target capability rows). The gfx1151 JIT error text names target `amdgpu-rdna3-5`. Both report `subgroup_size=32` (wave32).

Reproduce:

```sh
. ~/loom-serve/.venv-inkC/bin/activate
loom-run --list_devices=amdgpu
rocminfo | grep -E "Marketing|Uuid|Name:.*gfx"
for n in /sys/class/kfd/kfd/topology/nodes/*; do grep -E "gfx_target_version|unique_id" $n/properties; done
loom-test ~/inkling/inkC/saxpy.loom --device=amdgpu://GPU-e0880c05003c324d
```

## Machine-local device files

Placement is not part of the package. Two files select the device and the pool (`package/devices/`).

```json
{"devices": {"gpu": {"uri": "amdgpu://id/000000e2000000808e270f2b0bbb4e00"}},
 "pools": {"main": {"devices": ["gpu"], "pool_backing": "elastic", "memory_bytes": "48gib"}}}
```

`gfx1201.json` uses `amdgpu://GPU-e0880c05003c324d` and `"memory_bytes": "28gib"`. The pool budget on the 8060S must leave memory for the other agents. The 8060S pool limit that loom-serve reports by default is 132,578,619,392 B (all of unified RAM). The deployment document names `"device": "gpu"` and `"pool": "main"`.

## Fit of the checkpoints

| File | Tensor bytes | R9700 (34.2 GB) | 8060S |
|---|---|---|---|
| Q4_K_XL | 17,617,780,736 | fits, about 16 GB left for KV and workspace | fits |
| Q8_K_XL | 22,855,473,152 | fits, about 11 GB left | fits |

Measured: the whole Q4_K_XL checkpoint (744 tensors, 17.6 GB) loads as one parameter bank in about 10 s on the 8060S and about 12 s on the R9700, and every tensor is readable (see `LOOM_SERVE_NOTES.md`, experiment B). On the R9700 `rocm-smi` showed 19.9 GB VRAM in use with that bank resident.
