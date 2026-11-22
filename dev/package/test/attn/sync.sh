#!/bin/sh
# Copy the attn area to the box work dir.
set -e
cd "$(dirname "$0")/../../.."
files="package/kernel/attn package/test/attn $(ls package/lib/inkling/attn_*.loom 2>/dev/null)"
tar cf - $files | ssh user@gpu-box 'mkdir -p ~/inkling/inkE && cd ~/inkling/inkE && tar xmf -'
