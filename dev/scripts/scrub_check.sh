#!/usr/bin/env bash
# usage: scrub_check.sh [publish dir]   Fails if the publish tree names the GPU box, a user, a home path, a Windows path or an agent name (inkA..inkZ).
set -u
export LC_ALL=C
DIR=${1:-$(dirname "$0")/../publish/unsloth-Inkling-2-Mini-loom}
PATTERN='10\.0\.0\.|user|user|/home/|~/inkling|\bink[A-Z]\b|C:\\|C:/|/Users/|/mnt/c/'
if grep -rInE "$PATTERN" "$DIR" --exclude=tokenizer.json | cut -c1-200 | grep .; then
  echo "scrub_check: FAIL"; exit 1
fi
if grep -c -E "$PATTERN" "$DIR/tokenizer.json" | grep -qv '^0$'; then echo "scrub_check: FAIL (tokenizer.json)"; exit 1; fi
for f in model.json LICENSE NOTICE .gitattributes weights.loom tokenizer.json llm_config.json; do [ -f "$DIR/$f" ] || { echo "scrub_check: missing $f"; exit 1; }; done
echo "scrub_check: PASS"
