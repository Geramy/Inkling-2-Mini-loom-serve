#!/usr/bin/env bash
# Starts the chat server for gfx1151 with the q4 checkpoint on the GPU box (port 18090 unless given). See docs/RUNNING.md.
exec "$(dirname "$0")/serve_card.sh" gfx1151 q4 "${1:-18090}"
