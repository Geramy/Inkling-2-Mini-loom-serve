#!/usr/bin/env bash
# usage: chat_run_all.sh [gfx1151 gfx1201 ...]   (GPU box, from ~/inkling/inkI) builds the four chat test packages from package/control and runs every chat test per GPU.
# Each test server is started and stopped here by pid. Correctness runs hold the shared GPU lock.
set -uo pipefail
ROOT=${INKLING_CHAT_ROOT:-$HOME/inkling/inkI}
cd "$ROOT"
PY=${INKLING_CHAT_PY:-$HOME/loom-serve/.venv-inkI/bin/python}
TOKENIZER=$ROOT/tokenizer_nonspecial.json
[ -f "$TOKENIZER" ] || "$PY" scripts/chat_tokenizer.py "$HOME/inkling/inkC/tokenizer.json" "$TOKENIZER"
ARCHS=${*:-gfx1151 gfx1201}
declare -A PORT=([gfx1151]=${INKLING_CHAT_PORT_1151:-18430} [gfx1201]=${INKLING_CHAT_PORT_1201:-18440})
fail=0
"$PY" package/test/chat/scenarios.py > "$ROOT/scenarios.json"
build() { # name
  scripts/chat_build_test_pkg.sh "$ROOT/pkg_$1" "$ROOT/package/control" > /dev/null
  case $1 in
    echo) "$PY" scripts/chat_echo_kernel.py "$ROOT/pkg_$1/kernel.loom" "$ROOT/pkg_$1/kernel.loom.new" && mv "$ROOT/pkg_$1/kernel.loom.new" "$ROOT/pkg_$1/kernel.loom";;
    script) "$PY" scripts/chat_script_kernel.py "$ROOT/pkg_$1/kernel.loom" "$ROOT/pkg_$1/kernel.loom.new" "$ROOT/scenarios.json" && mv "$ROOT/pkg_$1/kernel.loom.new" "$ROOT/pkg_$1/kernel.loom";;
    policy) "$PY" scripts/chat_policy_kernel.py "$ROOT/pkg_$1";;
  esac
}
for arch in $ARCHS; do
  port=${PORT[$arch]}
  for variant in plain echo script policy; do
    build $variant
    pkg=$ROOT/pkg_$variant
    INKLING_TOKENIZER=$TOKENIZER scripts/chat_serve.sh "$pkg" "$arch" $port > "$ROOT/serve_$variant.log" 2>&1
    if ! grep -q ready "$ROOT/serve_$variant.log"; then echo "FAIL $arch $variant server did not start"; cat "$ROOT/serve_$variant.log" | tail -3; scripts/chat_stop.sh "$pkg" "$arch"; fail=1; continue; fi
    case $variant in
      plain) cmd=("$PY" package/test/chat/test_template.py $port);;
      echo) cmd=("$PY" package/test/chat/test_template.py $port --echo);;
      script) cmd=("$PY" package/test/chat/test_output.py $port);;
      policy) cmd=("$PY" package/test/chat/test_options.py $port);;
    esac
    echo "== $arch $variant: ${cmd[*]##*/}"
    flock -s "$HOME/hrx-pr1/gpu.lock" "${cmd[@]}" > "$ROOT/result_${arch}_$variant.txt" 2>&1 || fail=1
    grep -c "^PASS" "$ROOT/result_${arch}_$variant.txt" | sed 's/^/pass lines: /'; grep "^FAIL" "$ROOT/result_${arch}_$variant.txt" | head -5
    scripts/chat_stop.sh "$pkg" "$arch" > /dev/null
  done
done
exit $fail
