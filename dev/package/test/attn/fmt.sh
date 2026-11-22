#!/bin/sh
# Format the attn sources on the box and copy them back.
set -e
cd "$(dirname "$0")/../../.."
sh package/test/attn/sync.sh
ssh user@gpu-box '. ~/loom-serve/.venv-inkE/bin/activate && cd ~/inkling/inkE && loom-format --in-place package/kernel/attn/*.loom package/lib/inkling/attn_*.loom'
ssh user@gpu-box 'cd ~/inkling/inkE && tar cf - package/kernel/attn package/lib/inkling/attn_*.loom' | tar xmf -
