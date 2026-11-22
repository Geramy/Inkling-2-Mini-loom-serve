#!/usr/bin/env bash
# usage: chat_restart.sh <gfx1151|gfx1201> <port> [pkg dir]   (GPU box) rebuilds the test package from package/control and restarts its server
set -uo pipefail
ARCH=$1; PORT=$2; PKG=${3:-$HOME/inkling/inkI/pkgT}; ROOT=$HOME/inkling/inkI
"$ROOT/scripts/chat_stop.sh" "$PKG" "$ARCH" >/dev/null 2>&1
"$ROOT/scripts/chat_build_test_pkg.sh" "$PKG" "$ROOT/package/control" >/dev/null
INKLING_TOKENIZER=${INKLING_TOKENIZER:-$ROOT/tokenizer_nonspecial.json} timeout 100 "$ROOT/scripts/chat_serve.sh" "$PKG" "$ARCH" "$PORT" 2>&1 | tail -3
curl -s "localhost:$PORT/v1/models" | grep -o '"registration":"[a-z]*"\|"message":"[^"]*"' | head -3
