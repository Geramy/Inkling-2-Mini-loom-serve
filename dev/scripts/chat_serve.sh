#!/usr/bin/env bash
# usage: chat_serve.sh <package dir> <gfx1151|gfx1201> <port>   (runs on the GPU box, starts loom-serve in the background, pid in <package dir>/run-<arch>/pid)
set -euo pipefail
PKG=$(readlink -f "$1"); ARCH=$2; PORT=$3
GGUF=${INKLING_GGUF:-$HOME/models/inkling2-mini/Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf}
VENV=${LOOM_VENV:-$HOME/loom-serve/.venv-inkI}
RUN=$PKG/run-$ARCH
rm -rf "$RUN"; mkdir -p "$RUN"
for f in "$PKG"/*; do case "$f" in */run-*) ;; *) ln -s "$f" "$RUN/$(basename "$f")";; esac; done
ln -sf "$GGUF" "$RUN/weights.gguf"
ln -sf "${INKLING_TOKENIZER:-$HOME/inkling/inkC/tokenizer.json}" "$RUN/tokenizer.json"
. "$VENV/bin/activate"
cd "$RUN"
nohup loom-serve --devices="$PKG/devices/$ARCH.json" --model="$RUN/model.json" --listen=127.0.0.1:$PORT --workers=2 > "$RUN/serve.log" 2>&1 &
echo $! > "$RUN/pid"
for i in $(seq 1 120); do
  if curl -s "localhost:$PORT/v1/models" | grep -q '"registration":"ready"'; then echo "ready on $PORT (pid $(cat "$RUN/pid"))"; exit 0; fi
  if curl -s "localhost:$PORT/v1/status" | grep -q "\"registration\":\"failed\""; then echo "registration failed: $(curl -s localhost:$PORT/v1/status | grep -o "\"message\":\"[^\"]*\"")"; exit 2; fi
  if ! kill -0 "$(cat "$RUN/pid")" 2>/dev/null; then echo "server exited"; tail -20 "$RUN/serve.log"; exit 1; fi
  sleep 1
done
echo "timeout"; tail -20 "$RUN/serve.log"; exit 1
