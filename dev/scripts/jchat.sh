#!/usr/bin/env bash
# usage: jchat.sh <arch> <port> <q4|q8> [preset]   (local) restarts the card server and sends the prompt C chat request (not streaming and streaming); prints the reply text and the timing
cd "$(dirname "$0")/.."
bash scripts/jcard.sh "$1" "$2" "$3" "${4:-interactive-$3}" 900 2>&1 | head -1
ssh user@gpu-box "cd ~/inkling/inkK; scripts/chatreq.sh $2 | python3 -c 'import sys,json; d=json.load(sys.stdin); m=d[\"choices\"][0][\"message\"]; print(d[\"usage\"], d[\"choices\"][0][\"finish_reason\"]); print((m.get(\"reasoning_content\") or \"\")+(m.get(\"content\") or \"\"))'; scripts/chatreq.sh $2 stream | tail -c 200"
