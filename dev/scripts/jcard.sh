#!/usr/bin/env bash
# usage: jcard.sh <arch> <port> <q4|q8> [preset] [max wait s]   (local) syncs the package to the box, restarts the card server, waits until ready or failed, prints diagnostics
cd "$(dirname "$0")/.."
export INKLING_BOX_DIR=inkling/inkK
bash scripts/sync.sh 2>&1 | grep -v future
ssh user@gpu-box "cd ~/inkling/inkK; R=run-card-$1-$3; if [ -f \$R/pid ]; then P=\$(cat \$R/pid); kill \$P 2>/dev/null; for i in 1 2 3 4 5 6 7 8; do kill -0 \$P 2>/dev/null || break; sleep 1; done; kill -9 \$P 2>/dev/null; fi; sleep 1; nohup ./scripts/serve_card.sh $1 $3 $2 $4 > serve-card-$1-$3.log 2>&1 &
sleep 2; for i in \$(seq 1 \$(( ${5:-900} / 5 ))); do s=\$(curl -s -m 5 localhost:$2/v1/status | grep -o '\"registration\":\"[a-z_]*\"'); case \"\$s\" in *ready*|*failed*) break;; esac; sleep 5; done; echo \$s; curl -s localhost:$2/v1/status | head -c 1500; tail -c 600 serve-card-$1-$3.log"
