#!/usr/bin/env bash
set -euo pipefail

# Evaluate the trained CDs_and_Vinyl Qwen3 embedding checkpoint on committed
# RRec-style valid/test JSONL artifacts.

ROOT=${ROOT:-/mnt/tidal-sh01/usr/xiayu6/xiayu/aaai_pro}
VENV=${VENV:-/root/miniconda3/envs/swift}
PYTHON_BIN=${PYTHON_BIN:-$VENV/bin/python}

CATEGORY=${CATEGORY:-CDs_and_Vinyl}
SPLIT=${SPLIT:-test}
MAX_EXAMPLES=${MAX_EXAMPLES:-0}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

EMBEDDER_OUT=${EMBEDDER_OUT:-$ROOT/checkpoints/rrec_amazon_CDs_and_Vinyl/qwen3_embedding_cds_tidal}
QWEN3_EMBEDDING_MODEL=${QWEN3_EMBEDDING_MODEL:-}
RREC_EVAL_DIR=${RREC_EVAL_DIR:-$ROOT/github_artifacts/CDs_and_Vinyl/rrec_eval}
EVAL_EXAMPLES=${EVAL_EXAMPLES:-$RREC_EVAL_DIR/${SPLIT}.jsonl}
ITEM_INFO=${ITEM_INFO:-$RREC_EVAL_DIR/item_info.jsonl}

EMBEDDING_MAX_LENGTH=${EMBEDDING_MAX_LENGTH:-2048}
EMBEDDING_BATCH_SIZE=${EMBEDDING_BATCH_SIZE:-8}
EMBEDDING_TORCH_DTYPE=${EMBEDDING_TORCH_DTYPE:-bfloat16}
EMBEDDING_DEVICE=${EMBEDDING_DEVICE:-cuda:0}
MAX_ITEM_CHARS=${MAX_ITEM_CHARS:-0}
EVAL_QUERY_MODE=${EVAL_QUERY_MODE:-rebuild_history}
EVAL_COT_TEXT_MODE=${EVAL_COT_TEXT_MODE:-tagged}
EVAL_CANDIDATE_INDEX=${EVAL_CANDIDATE_INDEX:-0}
EVAL_REQUIRE_COT=${EVAL_REQUIRE_COT:-0}
EVAL_MASK_HISTORY_ITEMS=${EVAL_MASK_HISTORY_ITEMS:-1}
EVAL_MASK_PAD_ITEM=${EVAL_MASK_PAD_ITEM:-1}
EVAL_KEEP_TARGET_UNMASKED=${EVAL_KEEP_TARGET_UNMASKED:-0}
KS=${KS:-5,10,20}
EVAL_OUT=${EVAL_OUT:-}

if [[ "${SMOKE:-0}" == "1" && "$MAX_EXAMPLES" == "0" ]]; then
  MAX_EXAMPLES=100
fi
mask_suffix=""
if [[ "$EVAL_MASK_HISTORY_ITEMS" == "1" || "$EVAL_MASK_HISTORY_ITEMS" == "true" ]]; then
  mask_suffix="_masked_seen"
fi
EVAL_OUT=${EVAL_OUT:-$ROOT/outputs/rrec_amazon/eval/$CATEGORY/CDs_and_Vinyl_embedding_trained_${SPLIT}${mask_suffix}_max${MAX_EXAMPLES}.json}

require_path() {
  local label="$1"
  local path="$2"
  if [[ ! -e "$path" ]]; then
    echo "Missing $label: $path" >&2
    exit 1
  fi
}

require_path "project root" "$ROOT"
require_path "python" "$PYTHON_BIN"

if [[ -z "$QWEN3_EMBEDDING_MODEL" ]]; then
  require_path "trained embedding output dir" "$EMBEDDER_OUT"
  QWEN3_EMBEDDING_MODEL=$(find "$EMBEDDER_OUT" -type d -name 'checkpoint-*' -print 2>/dev/null | sort -V | tail -n 1)
fi

require_path "trained embedding checkpoint" "$QWEN3_EMBEDDING_MODEL"
require_path "RRec eval examples" "$EVAL_EXAMPLES"
require_path "RRec item_info" "$ITEM_INFO"

cd "$ROOT"
mkdir -p "$(dirname "$EVAL_OUT")"

export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES

echo "ROOT=$ROOT"
echo "EVAL_EXAMPLES=$EVAL_EXAMPLES"
echo "ITEM_INFO=$ITEM_INFO"
echo "QWEN3_EMBEDDING_MODEL=$QWEN3_EMBEDDING_MODEL"
echo "SPLIT=$SPLIT"
echo "MAX_EXAMPLES=$MAX_EXAMPLES"
echo "EVAL_OUT=$EVAL_OUT"
echo "EMBEDDING_DEVICE=$EMBEDDING_DEVICE"
echo "MAX_ITEM_CHARS=$MAX_ITEM_CHARS"
echo "EVAL_QUERY_MODE=$EVAL_QUERY_MODE"
echo "EVAL_COT_TEXT_MODE=$EVAL_COT_TEXT_MODE"
echo "EVAL_CANDIDATE_INDEX=$EVAL_CANDIDATE_INDEX"
echo "EVAL_REQUIRE_COT=$EVAL_REQUIRE_COT"
echo "EVAL_MASK_HISTORY_ITEMS=$EVAL_MASK_HISTORY_ITEMS"
echo "EVAL_MASK_PAD_ITEM=$EVAL_MASK_PAD_ITEM"
echo "EVAL_KEEP_TARGET_UNMASKED=$EVAL_KEEP_TARGET_UNMASKED"

require_cot_arg="--no-require-cot"
if [[ "$EVAL_REQUIRE_COT" == "1" || "$EVAL_REQUIRE_COT" == "true" ]]; then
  require_cot_arg="--require-cot"
fi
mask_history_arg="--no-mask-history-items"
if [[ "$EVAL_MASK_HISTORY_ITEMS" == "1" || "$EVAL_MASK_HISTORY_ITEMS" == "true" ]]; then
  mask_history_arg="--mask-history-items"
fi
mask_pad_arg="--no-mask-pad-item"
if [[ "$EVAL_MASK_PAD_ITEM" == "1" || "$EVAL_MASK_PAD_ITEM" == "true" ]]; then
  mask_pad_arg="--mask-pad-item"
fi
keep_target_arg=""
if [[ "$EVAL_KEEP_TARGET_UNMASKED" == "1" || "$EVAL_KEEP_TARGET_UNMASKED" == "true" ]]; then
  keep_target_arg="--keep-target-unmasked"
fi

"$PYTHON_BIN" scripts/eval/evaluate_rrec_jsonl_fullset.py \
  --examples "$EVAL_EXAMPLES" \
  --item-info "$ITEM_INFO" \
  --category "$CATEGORY" \
  --split "$SPLIT" \
  --max-examples "$MAX_EXAMPLES" \
  --query-mode "$EVAL_QUERY_MODE" \
  --cot-text-mode "$EVAL_COT_TEXT_MODE" \
  --candidate-index "$EVAL_CANDIDATE_INDEX" \
  "$require_cot_arg" \
  "$mask_history_arg" \
  "$mask_pad_arg" \
  $keep_target_arg \
  --ks "$KS" \
  --scorer qwen3_embedding \
  --embedding-model "$QWEN3_EMBEDDING_MODEL" \
  --embedding-max-length "$EMBEDDING_MAX_LENGTH" \
  --embedding-batch-size "$EMBEDDING_BATCH_SIZE" \
  --max-item-chars "$MAX_ITEM_CHARS" \
  --torch-dtype "$EMBEDDING_TORCH_DTYPE" \
  --device "$EMBEDDING_DEVICE" \
  --output "$EVAL_OUT"

echo "Wrote metrics: $EVAL_OUT"
