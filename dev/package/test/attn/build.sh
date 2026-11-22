#!/bin/sh
# On the box: merge attn sources into build/<name>.loom and compile for a target.
# usage: build.sh <name> <root> <target> files...
set -e
. ~/loom-serve/.venv-inkE/bin/activate
cd ~/inkling/inkE
name=$1; root=$2; target=$3; shift 3
mkdir -p build
loom-link "$@" --mode=merge --to=text --output=build/$name.loom
loom-compile build/$name.loom --root=@$root --target=amdgpu:$target --format=amdgpu-hsaco --output=build/$name.$target.hsaco --compile-report=text-summary 2>&1 | grep -E "error|status=|COMPILE-REPORT: (allocation|static_instruction_mix|memory|target_resources)" | sed -E 's/(artifact_bytes|materialized_spill_storage=|assignments=).*//' | cut -c1-330
