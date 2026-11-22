#!/usr/bin/env bash
# usage: serve.sh <gfx1151|gfx1201> [port] [extra loom-serve flags]   (runs on the GPU box)
# env: INKLING_MODE=sequence|text (default sequence), INKLING_CAPTURE=id,visit,occurrence,width (sequence mode),
#      INKLING_GGUF, INKLING_ROOT, INKLING_CONTEXT, INKLING_PREFILL, INKLING_ROWS, INKLING_POOL_BYTES
set -euo pipefail
ARCH=${1:?arch}
PORT=${2:-18090}
shift $(( $# < 2 ? $# : 2 ))
ROOT=${INKLING_ROOT:-$HOME/inkling/inkH}
PKG=$ROOT/package
MODE=${INKLING_MODE:-sequence}
GGUF=${INKLING_GGUF:-$HOME/models/inkling2-mini/Mainline_Support/Inkling-2-Mini-UD-Q4_K_XL.gguf}
VENV=${LOOM_VENV:-$HOME/loom-serve/.venv-inkC}
RUN=$ROOT/run-$MODE-$ARCH
CONTEXT=${INKLING_CONTEXT:-4096}
PREFILL=${INKLING_PREFILL:-512}
ROWS=${INKLING_ROWS:-2}
DENSE=0
case "$GGUF" in *Q8_K_XL*) DENSE=1 ;; esac
mkdir -p "$RUN"
ln -sfn "$PKG" "$RUN/pkg"
ln -sf "$PKG/weights.loom" "$RUN/weights.loom"
ln -sf "$GGUF" "$RUN/weights.gguf"
PROBES=${INKLING_PROBES:-0}
IFS=, read -r CID CVISIT COCC CWIDTH <<< "${INKLING_CAPTURE:-0,0,0,2304}"
cat > "$RUN/run_config.loom" <<EOF
config.def @inkling.capture_id = $CID : index

config.def @inkling.capture_visit = $CVISIT : index

config.def @inkling.capture_occurrence = $COCC : index

config.def @inkling.capture_width = $CWIDTH : index

config.def @inkling.dense_type = $DENSE : index

config.def @inkling.probes = $PROBES : index

config.def @inkling.tune_probe = ${INKLING_TUNE_PROBE:-0} : index

config.def @inkling.stop_after = ${INKLING_STOP:-99} : index
EOF
SOURCES='"run_config.loom","pkg/config.loom","pkg/prepare.loom","pkg/storage.loom","pkg/lib/vendor/epoch.loom","pkg/lib/vendor/storage.loom","pkg/lib/vendor/packet.loom","pkg/lib/vendor/feedback.loom","pkg/lib/vendor/span.loom","pkg/lib/vendor/q8_0.loom","pkg/lib/vendor/sampling.loom","pkg/lib/vendor/random.loom","pkg/control/chat_sampler.loom","pkg/lib/media_stub.loom","pkg/dispatch/serve_direct.loom","pkg/dispatch/spans.loom","pkg/dispatch/capture.loom","pkg/dispatch/targets.loom","pkg/dispatch/visit.loom","pkg/dispatch/layer.loom","pkg/dispatch/model.loom","pkg/lib/inkling/dense_common.loom","pkg/lib/inkling/dense_float.loom","pkg/lib/inkling/dense_q8_0.loom","pkg/lib/inkling/dense_abi.loom","pkg/lib/inkling/dense_vec.loom","pkg/lib/inkling/dense_mm_wmma.loom","pkg/lib/inkling/dense_wmma.loom","pkg/lib/inkling/dense_targets.loom","pkg/lib/inkling/attn_common.loom","pkg/lib/inkling/attn_matrix.loom","pkg/lib/inkling/attn_targets.loom","pkg/lib/inkling/experts_common.loom","pkg/lib/inkling/experts_stage.loom","pkg/lib/inkling/experts_wmma.loom","pkg/lib/inkling/nvfp4.loom","pkg/kernel/dense/norm.loom","pkg/kernel/dense/embed.loom","pkg/kernel/dense/gemv.loom","pkg/kernel/dense/gemv_split.loom","pkg/kernel/dense/gemm_q8_0.loom","pkg/kernel/dense/gemm_f32.loom","pkg/kernel/dense/gemm_bf16_wmma.loom","pkg/kernel/dense/head.loom","pkg/kernel/attn/attn_prep.loom","pkg/kernel/attn/attn_tile.loom","pkg/kernel/attn/attn_merge.loom","pkg/kernel/attn/attn_conv_commit.loom","pkg/kernel/attn/attn_kv_append.loom","pkg/kernel/experts/router.loom","pkg/kernel/experts/routes.loom","pkg/kernel/experts/gate_up.loom","pkg/kernel/experts/down.loom","pkg/kernel/experts/combine.loom","pkg/kernel/experts/gate_up_wmma.loom","pkg/kernel/experts/down_wmma.loom","pkg/kernel/mix/mix_states.loom","pkg/kernel/mix/ctrl.loom","pkg/kernel/mix/conv_res.loom","pkg/dispatch/hooks_dense.loom","pkg/dispatch/hooks_attn.loom","pkg/dispatch/hooks_mix.loom","pkg/dispatch/hooks_experts.loom"'
if [ "$PROBES" = 1 ]; then
  SOURCES="$SOURCES"',"pkg/dispatch/debug_probes.loom","pkg/lib/pr1/common.loom","pkg/lib/pr1/float.loom","pkg/lib/pr1/kq.loom","pkg/lib/pr1/q6_k.loom","pkg/lib/pr1/q8_0.loom","pkg/lib/pr1/mm_core.loom","pkg/lib/pr1/mm_vec.loom","pkg/lib/pr1/mul_mat.loom"'
else
  SOURCES="$SOURCES"',"pkg/dispatch/stub_debug.loom"'
fi
if [ "$MODE" = sequence ]; then
  CONTROL=pkg/dispatch/sequence.loom
  KIND=sequence
  EXTRA=
  LIMIT=
else
  [ -f "$ROOT/tokenizer.json" ] || "$VENV/bin/python" "$ROOT/scripts/gen_tokenizer.py" "$GGUF" "$ROOT/tokenizer.json"
  ln -sf "$ROOT/tokenizer.json" "$RUN/tokenizer.json"
  CONTROL=pkg/dispatch/chat_skeleton.loom
  KIND=text
  EXTRA='"tokenizer": "tokenizer.json",'
  LIMIT='"max_tokens": 64,'
fi
cat > "$RUN/model.json" <<EOF
{
  "name": "inkling-2-mini",
  "device": "gpu",
  "pool": "main",
  "definition": {
    "kind": "$KIND",
    "prepare": "pkg/prepare.loom",
    "control": "$CONTROL",
    "sources": [$SOURCES],
    "weights": "weights.gguf",
    $EXTRA
    "rows": $ROWS,
    "context_capacity": $CONTEXT,
    $LIMIT
    "checkpoint_capacity": ${INKLING_CHECKPOINTS:-0},
    "prefill_capacity": $PREFILL
  }
}
EOF
. "$VENV/bin/activate"
echo $$ > "$RUN/pid"
exec loom-serve --devices="$PKG/devices/$ARCH.json" --model="$RUN/model.json" --listen=127.0.0.1:$PORT --workers=4 "$@"
