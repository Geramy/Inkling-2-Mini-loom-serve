#!/usr/bin/env bash
# usage: chat_build_test_pkg.sh <out dir> <chat.loom dir>   (runs on the GPU box)
# Builds a direct-definition test package: the inkC skeleton with its chat functions replaced by package/control/chat*.loom.
set -euo pipefail
OUT=$1; CHAT=$2
rm -rf "$OUT"; mkdir -p "$OUT"
cp -r "$HOME/inkling/inkC/package/." "$OUT/"
rm -rf "$OUT"/run-*
awk '
BEGIN { n = split("request_defaults request_options bind_request chat_reasoning chat_image_limits chat_begin chat_message chat_end render_tool parse_tools prepare_input output_next complete_text", names, " "); for (i = 1; i <= n; i++) drop["func.def public @" names[i] "("] = 1; drop["func.def @effort_ordinal("] = 1; drop["func.def @effort_header("] = 1 }
{ if (!skipping) { for (key in drop) if (index($0, key) == 1) { skipping = 1; break } }
  if (skipping) { if ($0 == "}") skipping = 0; next }
  print }' "$HOME/inkling/inkC/package/control.loom" > "$OUT/control.loom"
sed -i 's/%state_bytes = index.constant 128 : offset/%state_bytes = index.constant 4096 : offset/' "$OUT/prepare.loom"
for f in chat chat_json; do sed -E 's/@text_([a-z_]+)_v1/@\1/g' "$CHAT/$f.loom" > "$OUT/$f.loom"; done
# The inkC skeleton keeps the option table at byte 128; the card layout (package/control) keeps it at byte 40.
sed -i 's/%table = index.constant 40 : offset/%table = index.constant 128 : offset/' "$OUT/chat.loom"
sed -i 's/"sources": \[/"sources": ["chat.loom", "chat_json.loom", /' "$OUT/model.json"
echo built "$OUT"
