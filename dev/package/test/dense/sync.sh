#!/bin/sh
# Copies the dense kernels and tests to the box. Usage: sync.sh
set -e
cd "$(dirname "$0")/../.."
tar cf - lib/inkling/dense_*.loom kernel/dense test/dense | ssh user@gpu-box 'mkdir -p ~/inkling/inkF/package && tar xmf - -C ~/inkling/inkF/package'
