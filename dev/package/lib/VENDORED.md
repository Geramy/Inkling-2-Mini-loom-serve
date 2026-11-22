# Vendored files

Every file below is copied unchanged except for the listed edits. Each source is MIT licensed (see the notice at the end).

## hrx-loom-library (ROCm/hrx-loom-library), tag v0.2.0, commit 1fd6bd23279e1f77b3d6b8149ef6d03b27158e8a

The tag matches the installed SDK (hrx-loom 0.2.0). The library main branch (cb9ae9e, 2026-11-20) splits these files differently and its `publish.cxx` needs a newer `loomcxx` than the SDK ships, so main is not used.

| Copy | Source path in the library | Edits |
|---|---|---|
| `lib/vendor/epoch.loom` | `program/text/qwen3_8/epoch.loom` | `qwen38` renamed to `inkling`; contains encode_state, encode_epoch, publish_epoch, step, epoch |
| `lib/vendor/storage.loom` | `program/text/qwen3_8/storage.loom` | `qwen38` renamed to `inkling` |
| `lib/vendor/packet.loom` | `motif/sequence/qwen3_8/packet.loom` | `qwen38` renamed to `inkling` |
| `lib/vendor/feedback.loom` | `motif/sequence/qwen3_8/feedback.loom` | `qwen38` renamed to `inkling` |
| `lib/vendor/span.loom` | `motif/sequence/qwen3_8/span.loom` | `qwen38` renamed to `inkling` |
| `lib/vendor/input.loom` | `motif/sequence/qwen3_8/input.loom` | `qwen38` renamed to `inkling` (not linked; media inputs are out of scope) |
| `lib/vendor/q8_0.loom` | `motif/format/ggml/q8_0.loom` | none |
| `lib/vendor/sampling.loom` | `motif/tensor/sampling.loom` | none |
| `lib/vendor/random.loom` | `motif/tensor/random.loom` | none |

Not vendored: the library `encode.loom`, `publish.cxx`, `epoch_target.loom` and `speculation.loom` exist only on main. `lib/media_stub.loom` is our own stub of the two media functions that `epoch.loom` declares (`encode_media_plan`, `publish_media_positions`).

`attn_matrix.loom` (`lib/inkling/`) is adapted from `hrx-loom-library/motif/tensor/matrix.loom`.

## llama.cpp branch hrx-llama-pr1, commit 0c0b59d51e3971c872fbd5b38cfe5b123156955f (`ggml/src/ggml-hrx/kernels/`)

| Copy | Source path | Edits |
|---|---|---|
| `lib/pr1/common.loom` | `motifs/common.loom` | `func.def` made `public` |
| `lib/pr1/float.loom` | `motifs/float.loom` | same |
| `lib/pr1/kq.loom` | `motifs/kq.loom` | same |
| `lib/pr1/q6_k.loom` | `motifs/q6_k.loom` | same |
| `lib/pr1/q8_0.loom` | `motifs/q8_0.loom` | same |
| `lib/pr1/mm_core.loom` | `matmul/lib/mm_core.loom` | same |
| `lib/pr1/mm_vec.loom` | `matmul/lib/mm_vec.loom` | same |
| `lib/pr1/mul_mat.loom` | `matmul/mul_mat.loom` | same |

The edit is needed because the serve linker keeps `func.def` private to its file unless it is `public`. These files are used only by the debug probe (`dispatch/debug_probes.loom`, capture id 100) until the dense area takes over.

## MIT notice

hrx-loom-library: Copyright (c) 2026 AMD ROCm Software. llama.cpp: Copyright (c) 2023-2026 The ggml authors.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## Notes from the kernel areas

- `lib/inkling/dense_*.loom` motifs derive from llama.cpp ggml-hrx PR1 (MIT). Nothing in the dense area comes from hrx-loom-library.
- Pending: the chat sampler needs `sampling.loom` and `random.loom` from `hrx-loom-library` (tag v0.2.0, `motif/tensor/`, MIT). Vendor them verbatim into `lib/vendor/` when the head is integrated, and add them to the table above.
