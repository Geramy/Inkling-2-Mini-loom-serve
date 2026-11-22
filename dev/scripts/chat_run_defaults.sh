#!/usr/bin/env bash
# usage: chat_run_defaults.sh <gfx1151|gfx1201>   (GPU box, from ~/inkling/inkI) deployment sampling defaults: a valid options object sets the defaults, invalid ones fail the registration.
set -uo pipefail
ROOT=${INKLING_CHAT_ROOT:-$HOME/inkling/inkI}
cd "$ROOT"
ARCH=$1
PY=${INKLING_CHAT_PY:-$HOME/loom-serve/.venv-inkI/bin/python}
PORT=${INKLING_CHAT_PORT:-$([ "$ARCH" = gfx1151 ] && echo 18450 || echo 18460)}
fail=0
try() { # label options-json expectation(ready|failed)
  pkg=$ROOT/pkg_defaults
  scripts/chat_build_test_pkg.sh "$pkg" "$ROOT/package/control" > /dev/null
  "$PY" scripts/chat_policy_kernel.py "$pkg"
  sed -i 's/%valid_options = scalar.andi %object, %empty_options : i1/%valid_options = scalar.andi %object, %object : i1/' "$pkg/prepare.loom"
  sed -i "s|\"kind\": \"text\",|\"kind\": \"text\", \"options\": $2,|" "$pkg/model.json"
  out=$(INKLING_TOKENIZER=$ROOT/tokenizer_nonspecial.json scripts/chat_serve.sh "$pkg" "$ARCH" $PORT 2>&1 | tail -1)
  if [ "$3" = ready ]; then
    case $out in ready*) echo "PASS $1 registers";; *) echo "FAIL $1 registers: $out"; fail=1;; esac
    [ "$1" = valid ] && { flock -s "$HOME/hrx-pr1/gpu.lock" "$PY" package/test/chat/test_options.py $PORT --defaults || fail=1; }
  else
    case $out in "registration failed"*) echo "PASS $1 rejected: ${out:0:140}";; *) echo "FAIL $1 should fail registration: $out"; fail=1;; esac
  fi
  scripts/chat_stop.sh "$pkg" "$ARCH" > /dev/null
}
try valid '{"sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20}}' ready
try negative_temperature '{"sampling": {"temperature": -1}}' failed
try unknown_key '{"sampling": {"temperature": 0.6, "min_p": 0.1}}' failed
try unknown_option '{"other": 1}' failed
try null_value '{"sampling": {"temperature": null}}' failed
exit $fail
