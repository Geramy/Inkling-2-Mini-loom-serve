#!/usr/bin/env bash
# Starts the chat server for gfx1201 with the q8 checkpoint on the GPU box (port 18090 unless given). See docs/RUNNING.md.
exec "$(dirname "$0")/serve_card.sh" gfx1201 q8 "${1:-18090}"
