#!/usr/bin/env bash
set -euo pipefail

# Evaluate every embedding checkpoint under CHECKPOINT_ROOT on RRec-style
# CDs_and_Vinyl valid/test JSONL artifacts. Existing metric files are skipped.

ROOT=${ROOT:-/mnt/tidal-sh01/usr/xiayu6/xiayu/aaai_pro}
VENV=${VENV:-/root/miniconda3/envs/swift}

CATEGORY=${CATEGORY:-CDs_and_Vinyl}
SPLIT=${SPLIT:-test}
MAX_EXAMPLES=${MAX_EXAMPLES:-0}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-${EMBEDDER_OUT:-$ROOT/checkpoints/rrec_amazon_CDs_and_Vinyl/qwen3_embedding_cds_tidal}}
CHECKPOINT_PATTERN=${CHECKPOINT_PATTERN:-checkpoint-*}
EVAL_DIR=${EVAL_DIR:-$ROOT/outputs/rrec_amazon/eval/$(basename "$CHECKPOINT_ROOT")_${SPLIT}_all_ckpts}

RREC_EVAL_DIR=${RREC_EVAL_DIR:-$ROOT/github_artifacts/CDs_and_Vinyl/rrec_eval}
EVAL_EXAMPLES=${EVAL_EXAMPLES:-$RREC_EVAL_DIR/${SPLIT}.jsonl}
ITEM_INFO=${ITEM_INFO:-$RREC_EVAL_DIR/item_info.jsonl}

EMBEDDING_MAX_LENGTH=${EMBEDDING_MAX_LENGTH:-4096}
EMBEDDING_BATCH_SIZE=${EMBEDDING_BATCH_SIZE:-256}
EMBEDDING_TORCH_DTYPE=${EMBEDDING_TORCH_DTYPE:-bfloat16}
EMBEDDING_DEVICE=${EMBEDDING_DEVICE:-cuda:0}
KS=${KS:-5,10,20}
FORCE_EVAL=${FORCE_EVAL:-0}

require_path() {
  local label="$1"
  local path="$2"
  if [[ ! -e "$path" ]]; then
    echo "Missing $label: $path" >&2
    exit 1
  fi
}

find_checkpoints() {
  local root="$1"
  if [[ "$(basename "$root")" == checkpoint-* ]]; then
    echo "$root"
    return
  fi
  find "$root" -maxdepth 1 -type d -name "$CHECKPOINT_PATTERN" -print 2>/dev/null | sort -V
}

require_path "project root" "$ROOT"
require_path "checkpoint root" "$CHECKPOINT_ROOT"
require_path "RRec eval examples" "$EVAL_EXAMPLES"
require_path "RRec item_info" "$ITEM_INFO"

mapfile -t CHECKPOINTS < <(find_checkpoints "$CHECKPOINT_ROOT")
if (( ${#CHECKPOINTS[@]} == 0 )); then
  echo "No checkpoints found under $CHECKPOINT_ROOT with pattern $CHECKPOINT_PATTERN" >&2
  exit 1
fi

mkdir -p "$EVAL_DIR"

echo "ROOT=$ROOT"
echo "CHECKPOINT_ROOT=$CHECKPOINT_ROOT"
echo "CHECKPOINT_PATTERN=$CHECKPOINT_PATTERN"
echo "EVAL_DIR=$EVAL_DIR"
echo "SPLIT=$SPLIT"
echo "MAX_EXAMPLES=$MAX_EXAMPLES"
echo "EVAL_EXAMPLES=$EVAL_EXAMPLES"
echo "ITEM_INFO=$ITEM_INFO"
echo "EMBEDDING_BATCH_SIZE=$EMBEDDING_BATCH_SIZE"
echo "EMBEDDING_MAX_LENGTH=$EMBEDDING_MAX_LENGTH"
echo "KS=$KS"
echo "Found ${#CHECKPOINTS[@]} checkpoint(s)"

for checkpoint in "${CHECKPOINTS[@]}"; do
  require_path "checkpoint" "$checkpoint"
  step="$(basename "$checkpoint")"
  out="$EVAL_DIR/${step}_${SPLIT}.json"

  if [[ "$FORCE_EVAL" != "1" && "$FORCE_EVAL" != "true" && -s "$out" ]]; then
    echo "Skip existing: $out"
    continue
  fi

  echo "Evaluating $checkpoint -> $out"
  CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  ROOT="$ROOT" \
  VENV="$VENV" \
  CATEGORY="$CATEGORY" \
  SPLIT="$SPLIT" \
  MAX_EXAMPLES="$MAX_EXAMPLES" \
  QWEN3_EMBEDDING_MODEL="$checkpoint" \
  RREC_EVAL_DIR="$RREC_EVAL_DIR" \
  EVAL_EXAMPLES="$EVAL_EXAMPLES" \
  ITEM_INFO="$ITEM_INFO" \
  EMBEDDING_MAX_LENGTH="$EMBEDDING_MAX_LENGTH" \
  EMBEDDING_BATCH_SIZE="$EMBEDDING_BATCH_SIZE" \
  EMBEDDING_TORCH_DTYPE="$EMBEDDING_TORCH_DTYPE" \
  EMBEDDING_DEVICE="$EMBEDDING_DEVICE" \
  KS="$KS" \
  EVAL_OUT="$out" \
  bash scripts/eval_cds_embedding_trained_tidal.sh
done

echo "Wrote metrics under: $EVAL_DIR"
