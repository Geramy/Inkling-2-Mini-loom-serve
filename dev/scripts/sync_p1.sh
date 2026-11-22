#!/usr/bin/env bash
# Copies package_P1/ (as package/) and scripts/ to ~/inkling/optP1 on the GPU box.
set -euo pipefail
cd "$(dirname "$0")/.."
BOX=${BOX:-user@gpu-box}
tar cf - --transform "s#^package_P1#package#;s#^scripts_P1#scripts#" package_P1 scripts scripts_P1 | ssh "$BOX" "mkdir -p ~/inkling/optP1 && cd ~/inkling/optP1 && tar xmf - && chmod +x scripts/*.sh"
