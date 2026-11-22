#!/usr/bin/env bash
# Usage: ROOTS="a b" build.sh <gfx1201|gfx1151> files...   Links the sources and compiles the kernels with a compile report.
set -euo pipefail
arch=$1; shift
out=${OUT:-build}
mkdir -p "$out"
loom-link "$@" --output="$out/m.loom"
roots=()
for r in $ROOTS; do roots+=(--root=@$r); done
loom-compile "$out/m.loom" --target=amdgpu:$arch --format=amdgpu-hsaco "${roots[@]}" --output="$out/m-$arch.hsaco" --compile-report=${REPORT:-text-summary}
