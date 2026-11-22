#!/bin/sh
# Compile one kernel file on the box: cc.sh file.loom [target]
. ~/inkling/inkG/.venv/bin/activate
cd ~/inkling/inkG
loom-compile "$1" --target="${2:-amdgpu:gfx11-generic}" --output=/tmp/inkG.out 2>&1 | head -${LINES_MAX:-30}
