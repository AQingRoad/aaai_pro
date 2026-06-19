#!/usr/bin/env bash
set -euo pipefail

# Standalone CDs_and_Vinyl embedding training on the Tidal server.
# It trains Qwen3-Embedding-0.6B on the committed phase0 JSONL pairs.

ROOT=${ROOT:-/mnt/tidal-sh01/usr/xiayu6/xiayu/aaai_pro}
VENV=${VENV:-/root/miniconda3/envs/swift}
PYTHON_BIN=${PYTHON_BIN:-$VENV/bin/python}

BASE_EMBEDDING_MODEL=${BASE_EMBEDDING_MODEL:-/mnt/tidal-sh01/usr/xiayu6/xiayu/checkpoint/Qwen3_embedding/0.6B}
EMBEDDER_DATASET=${EMBEDDER_DATASET:-$ROOT/github_artifacts/CDs_and_Vinyl/phase0/phase0_embedder_rrec_amazon_cds_and_vinyl_train.jsonl}
EMBEDDER_OUT=${EMBEDDER_OUT:-$ROOT/checkpoints/rrec_amazon_CDs_and_Vinyl/qwen3_embedding_cds_tidal}

CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
EMBEDDER_BATCH_SIZE=${EMBEDDER_BATCH_SIZE:-128}
EMBEDDER_GRAD_ACCUM=${EMBEDDER_GRAD_ACCUM:-1}
EMBEDDER_MAX_LENGTH=${EMBEDDER_MAX_LENGTH:-2048}
EMBEDDER_EPOCHS=${EMBEDDER_EPOCHS:-3}
EMBEDDER_MAX_STEPS=${EMBEDDER_MAX_STEPS:--1}
EMBEDDER_LR=${EMBEDDER_LR:-6e-6}
EMBEDDER_SAVE_STEPS=${EMBEDDER_SAVE_STEPS:-auto}
EMBEDDER_TORCH_DTYPE=${EMBEDDER_TORCH_DTYPE:-bfloat16}
SEED=${SEED:-42}

if [[ "$EMBEDDER_MAX_STEPS" == "smoke" || "${SMOKE:-0}" == "1" ]]; then
  EMBEDDER_MAX_STEPS=1
  EMBEDDER_SAVE_STEPS=1
fi

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
require_path "base embedding model" "$BASE_EMBEDDING_MODEL"
require_path "phase0 embedder dataset" "$EMBEDDER_DATASET"

if [[ "$EMBEDDER_SAVE_STEPS" == "auto" ]]; then
  row_count=$(wc -l < "$EMBEDDER_DATASET" | tr -d ' ')
  full_batches=$((row_count / EMBEDDER_BATCH_SIZE))
  if ((full_batches < 1)); then
    echo "Need at least one full batch: rows=$row_count batch_size=$EMBEDDER_BATCH_SIZE" >&2
    exit 1
  fi
  EMBEDDER_SAVE_STEPS=$(((full_batches + EMBEDDER_GRAD_ACCUM - 1) / EMBEDDER_GRAD_ACCUM))
fi

mkdir -p "$EMBEDDER_OUT"
cd "$ROOT"

export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export MODELSCOPE_CACHE=${MODELSCOPE_CACHE:-/mnt/tidal-sh01/usr/xiayu6/xiayu/modelscope_cache}
export CUDA_VISIBLE_DEVICES

echo "ROOT=$ROOT"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "BASE_EMBEDDING_MODEL=$BASE_EMBEDDING_MODEL"
echo "EMBEDDER_DATASET=$EMBEDDER_DATASET"
echo "EMBEDDER_OUT=$EMBEDDER_OUT"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

"$PYTHON_BIN" scripts/train_phase0_embedder.py \
  --model "$BASE_EMBEDDING_MODEL" \
  --dataset "$EMBEDDER_DATASET" \
  --output-dir "$EMBEDDER_OUT" \
  --max-length "$EMBEDDER_MAX_LENGTH" \
  --batch-size "$EMBEDDER_BATCH_SIZE" \
  --grad-accum "$EMBEDDER_GRAD_ACCUM" \
  --epochs "$EMBEDDER_EPOCHS" \
  --max-steps "$EMBEDDER_MAX_STEPS" \
  --learning-rate "$EMBEDDER_LR" \
  --torch-dtype "$EMBEDDER_TORCH_DTYPE" \
  --save-steps "$EMBEDDER_SAVE_STEPS" \
  --seed "$SEED"

latest_checkpoint=$(find "$EMBEDDER_OUT" -type d -name 'checkpoint-*' -print 2>/dev/null | sort -V | tail -n 1)
if [[ -n "$latest_checkpoint" ]]; then
  echo "QWEN3_EMBEDDING_MODEL=$latest_checkpoint"
fi
