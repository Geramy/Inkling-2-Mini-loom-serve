#!/usr/bin/env bash
# Copies package/ and scripts/ from the laptop to the GPU box (default ~/inkling/inkH).
set -euo pipefail
cd "$(dirname "$0")/.."
BOX=${BOX:-user@gpu-box}
DEST=${INKLING_BOX_DIR:-inkling/inkH}
tar cf - package scripts | ssh "$BOX" "mkdir -p ~/$DEST && cd ~/$DEST && tar xmf - && chmod +x scripts/*.sh"
