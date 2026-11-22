#!/usr/bin/env bash
# usage: chat_stop.sh <package dir> <arch>   stops the loom-serve started by chat_serve.sh (by pid file only)
RUN=$(readlink -f "$1")/run-$2
[ -f "$RUN/pid" ] && kill "$(cat "$RUN/pid")" 2>/dev/null && rm -f "$RUN/pid" && echo stopped
