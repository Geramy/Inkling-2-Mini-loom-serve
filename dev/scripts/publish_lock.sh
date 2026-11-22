#!/usr/bin/env bash
# usage: publish_lock.sh <publish dir> <snapshot dir>   (runs on the GPU box)
# Writes model.json.lock for all variants. The snapshot dir has Mainline_Support/<name>.gguf_file links to the local GGUF files (see docs/RUNNING.md).
set -euo pipefail
cd "${1:?publish dir}"
~/loom/.venv/bin/loom-fetch ./model.json --update-lock --all-variants --no-input --origin checkpoint="${2:?snapshot dir}"
rm -f .loom-fetch.lock
