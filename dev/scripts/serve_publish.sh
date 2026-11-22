#!/usr/bin/env bash
# usage: serve_publish.sh <gfx1151|gfx1201> <materialized dir> <preset> [port]   (runs on the GPU box)
# Starts loom-serve from a directory made by `loom-fetch <publish>/model.json --preset <preset> --materialize=copy --output <dir>`. Stop it with: kill $(cat <dir>/serve.pid)
set -euo pipefail
ARCH=${1:?gfx1151|gfx1201}; DIR=${2:?materialized dir}; PRESET=${3:?preset}; PORT=${4:-18090}
VENV=${LOOM_VENV:-$HOME/loom/.venv}
case "$ARCH" in
  gfx1151) URI=amdgpu://id/000000e2000000808e270f2b0bbb4e00 ;;
  gfx1201) URI=amdgpu://GPU-e0880c05003c324d ;;
  *) echo "arch must be gfx1151 or gfx1201"; exit 1 ;;
esac
. "$VENV/bin/activate"
echo $$ > "$DIR/serve.pid"
exec loom-serve --device="$URI" --model="$DIR" --preset="$PRESET" --pool_backing=fixed --listen=127.0.0.1:$PORT --workers=4
