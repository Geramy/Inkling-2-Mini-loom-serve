#!/usr/bin/env bash
# usage: serve_tiny.sh <gfx1151|gfx1201> <port> [extra loom-serve flags]
set -euo pipefail
case "$1" in
  gfx1151) URI=amdgpu://id/000000e2000000808e270f2b0bbb4e00 ;;
  gfx1201) URI=amdgpu://GPU-e0880c05003c324d ;;
  *) echo "unknown arch $1" >&2; exit 2 ;;
esac
PORT=$2; shift 2
. ~/loom-serve/.venv-inkC/bin/activate
cd ~/inkling/inkC
exec loom-serve --device="$URI" --model=tiny-text/model.json --listen=127.0.0.1:$PORT --workers=2 "$@"
