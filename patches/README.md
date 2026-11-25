# loom-serve patches

These patches change loom-serve (ROCm/hrx-system, `loom/src/loom/serve/service/text/chat.c`) so tool calling works with common OpenAI-compatible clients. The package itself already works around the gaps; the patches remove the need for the workarounds. They are open as pull requests:

- `01-tool-args-object.patch`: accept tool call `arguments` as a JSON object. https://github.com/ROCm/hrx-system/pull/3551
- `02-tool-call-type-default.patch`: a missing tool call `type` means `function`. https://github.com/ROCm/hrx-system/pull/3552
- `03-tool-name-from-call-id.patch`: give tool messages the function name of their call. https://github.com/ROCm/hrx-system/pull/3554

Apply to a checkout of ROCm/hrx-system `main` with `git apply`.
