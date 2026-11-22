#!/bin/bash
# CPU reference llama-server (oracle build: mainline 418fe97 + PR 32704). Read-only use of the build tree.
# usage: oracle_server.sh start|stop|status [q4|q8]     (default q4)
#   env: ORACLE_BIN (build/bin dir), ORACLE_CTX (default 32768), ORACLE_THREADS (default 12), ORACLE_PORT, ORACLE_STATE, ORACLE_EXTRA
set -u
bin=${ORACLE_BIN:-$HOME/inkling/inkB/build/bin}
md=${ORACLE_MODELS:-$HOME/models/inkling2-mini/Mainline_Support}
state=${ORACLE_STATE:-$HOME/inkling/inkO/run}
ctx=${ORACLE_CTX:-32768}
threads=${ORACLE_THREADS:-12}
cmd=${1:-status}; q=${2:-q4}
case $q in
  q4) gguf=$md/Inkling-2-Mini-UD-Q4_K_XL.gguf; port=${ORACLE_PORT:-18400};;
  q8) gguf=$md/Inkling-2-Mini-UD-Q8_K_XL.gguf; port=${ORACLE_PORT:-18401};;
  *) echo "variant must be q4 or q8"; exit 2;;
esac
mkdir -p "$state"; pidf=$state/oracle-$q.pid; log=$state/oracle-$q.log
alive() { [ -f "$pidf" ] && kill -0 "$(cat "$pidf")" 2>/dev/null && tr '\0' ' ' < /proc/"$(cat "$pidf")"/cmdline | grep -q llama-server; }
case $cmd in
  start)
    alive && { echo "already running: pid $(cat "$pidf") port $port"; exit 0; }
    ss -ltn | grep -q ":$port " && { echo "port $port is busy"; exit 1; }
    cd "$bin" || exit 1
    # shellcheck disable=SC2086
    nohup ./llama-server -m "$gguf" --host 127.0.0.1 --port "$port" -ngl 0 -c $((ctx + 1024)) -np 1 -t "$threads" -ctk f32 -ctv f32 --jinja ${ORACLE_EXTRA:-} > "$log" 2>&1 &
    echo $! > "$pidf"
    for _ in $(seq 600); do
      curl -sf -m 2 "localhost:$port/health" > /dev/null && { echo "ready: pid $(cat "$pidf") port $port ($q, ctx $((ctx + 1024)), $threads threads)"; exit 0; }
      alive || { echo "server exited; see $log"; rm -f "$pidf"; exit 1; }
      sleep 1
    done
    echo "timeout waiting for health; see $log"; exit 1;;
  stop)
    alive || { rm -f "$pidf"; echo "not running"; exit 0; }
    pid=$(cat "$pidf"); kill "$pid"
    for _ in $(seq 60); do kill -0 "$pid" 2>/dev/null || break; sleep 0.5; done
    kill -0 "$pid" 2>/dev/null && kill -9 "$pid"
    rm -f "$pidf"; echo "stopped";;
  status)
    if alive; then echo "running: pid $(cat "$pidf") port $port"; curl -s -m 2 "localhost:$port/health"; echo; else echo "not running"; exit 1; fi;;
  *) echo "usage: $0 start|stop|status [q4|q8]"; exit 2;;
esac
