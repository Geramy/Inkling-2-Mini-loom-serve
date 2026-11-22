#!/usr/bin/env bash
# usage: card_test.sh <arch> <port> <gguf file name> [preset]   (local) serves package/card.json through a deployment document on the box
set -euo pipefail
cd "$(dirname "$0")/.."
ARCH=${1:?arch}; PORT=${2:?port}; GGUF=${3:?gguf file name}; PRESET=${4:-capture-q8}
BOX=${BOX:-user@gpu-box}; DEST=${INKLING_BOX_DIR:-inkling/inkH}
bash scripts/sync.sh 2>&1 | grep -v "in the future" || true
ssh "$BOX" "bash -s" <<REMOTE
set -e
cd /home/user/$DEST
R=run-card-$ARCH
[ -f \$R/pid ] && kill \$(cat \$R/pid) 2>/dev/null || true
sleep 1
mkdir -p \$R
ln -sfn /home/user/$DEST/package \$R/pkg
ln -sf /home/user/models/inkling2-mini/Mainline_Support/$GGUF package/$GGUF
cat > \$R/model.json <<JSON
{"name": "inkling-2-mini", "device": "gpu", "pool": "main", "model": {"path": "pkg/card.json", "preset": "$PRESET"}, "definition": {"kind": "sequence"}}
JSON
. /home/user/loom-serve/.venv-inkC/bin/activate
nohup loom-serve --devices=/home/user/$DEST/package/devices/$ARCH.json --model=\$R/model.json --listen=127.0.0.1:$PORT --workers=4 > serve-card-$ARCH.log 2>&1 &
echo \$! > \$R/pid
REMOTE
sleep 25
bash scripts/diag.sh "$PORT"
