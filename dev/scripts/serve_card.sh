#!/usr/bin/env bash
# usage: serve_card.sh <gfx1151|gfx1201> <q4|q8> [port] [preset]   (runs on the GPU box)
# Starts the chat server (OpenAI compatible, http://127.0.0.1:<port>/v1/chat/completions) from package/card.json. Stop it with: kill $(cat <run dir>/pid)
# env: INKLING_ROOT (default ~/inkling/inkK), LOOM_VENV, INKLING_MODELS (directory of the GGUF files), INKLING_PRESET, INKLING_SERVICE (chat or capture)
set -euo pipefail
ARCH=${1:?gfx1151|gfx1201}; Q=${2:?q4|q8}; PORT=${3:-18090}
ROOT=${INKLING_ROOT:-$HOME/inkling/inkK}
VENV=${LOOM_VENV:-$HOME/loom-serve/.venv-inkK}
MODELS=${INKLING_MODELS:-$HOME/models/inkling2-mini/Mainline_Support}
PRESET=${4:-${INKLING_PRESET:-interactive-$Q}}
case "$Q" in q4) GGUF=Inkling-2-Mini-UD-Q4_K_XL.gguf ;; q8) GGUF=Inkling-2-Mini-UD-Q8_K_XL.gguf ;; *) echo "quant must be q4 or q8"; exit 1 ;; esac
RUN=$ROOT/run-card-$ARCH-$Q
mkdir -p "$RUN"
ln -sfn "$ROOT/package" "$RUN/pkg"
ln -sf "$MODELS/$GGUF" "$ROOT/package/$GGUF"
KIND=text
case "$PRESET" in capture*) KIND=sequence ;; esac
cat > "$RUN/model.json" <<JSON
{"name": "inkling-2-mini", "device": "gpu", "pool": "main", "model": {"path": "pkg/card.json", "preset": "$PRESET"}, "definition": {"kind": "$KIND"}}
JSON
. "$VENV/bin/activate"
echo $$ > "$RUN/pid"
exec loom-serve --devices="$ROOT/package/devices/$ARCH.json" --model="$RUN/model.json" --listen=127.0.0.1:$PORT --workers=4
