#!/bin/sh
# Copy the mix area to the box work dir.
set -e
cd "$(dirname "$0")/../../.."
files="package/kernel/mix package/test/mix $(ls package/lib/inkling/mix_*.loom 2>/dev/null || true)"
tar cf - $files | ssh user@gpu-box 'mkdir -p ~/inkling/inkG && cd ~/inkling/inkG && tar xmf -'
