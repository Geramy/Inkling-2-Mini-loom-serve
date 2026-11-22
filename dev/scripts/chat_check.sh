#!/usr/bin/env bash
# usage: chat_check.sh   (GPU box) links package/control/chat*.loom and compiles the exports for the VM target
. "$HOME/loom-serve/.venv-inkI/bin/activate"
cd "$HOME/inkling/inkI/package/control" || exit 1
loom-link --root-library=chat.loom --library=chat_json.loom --to=text --output=/tmp/inkI_linked.loom || exit 1
loom-compile /tmp/inkI_linked.loom --target=vm:core --output=/tmp/inkI_linked.vm 2>&1 | grep -E "error" | sed -E 's/^.*(chat[a-z_]*\.loom|inkI_linked.loom)/\1/' | head -${1:-12}
