#!/usr/bin/env bash
set -euo pipefail

# Build and train a CoT-aware CDs_and_Vinyl embedding model.
# The generated dataset contains query/positive pairs for:
#   1) history -> target item
#   2) history + generated CoT -> target item

ROOT=${ROOT:-/mnt/tidal-sh01/usr/xiayu6/xiayu/aaai_pro}
VENV=${VENV:-/root/miniconda3/envs/swift}
PYTHON_BIN=${PYTHON_BIN:-$VENV/bin/python}

CATEGORY=${CATEGORY:-CDs_and_Vinyl}
OUT_DIR=${OUT_DIR:-$ROOT/outputs/rrec_amazon/$CATEGORY}
RREC_EVAL_DIR=${RREC_EVAL_DIR:-$ROOT/github_artifacts/CDs_and_Vinyl/rrec_eval}
ITEM_INFO=${ITEM_INFO:-$RREC_EVAL_DIR/item_info.jsonl}

COT_CANDIDATE_LISTS=${COT_CANDIDATE_LISTS:-$OUT_DIR/cot_candidate_one_lists_deepseek_v4_pro_low.jsonl}
COT_TEXT_MODE=${COT_TEXT_MODE:-answer}
COT_EMBEDDER_DATASET=${COT_EMBEDDER_DATASET:-$OUT_DIR/phase0_embedder_cds_with_cot_${COT_TEXT_MODE}.jsonl}
FORCE_REBUILD_DATASET=${FORCE_REBUILD_DATASET:-0}
INCLUDE_HISTORY=${INCLUDE_HISTORY:-1}
INCLUDE_COT=${INCLUDE_COT:-1}
MAX_COT_CHARS=${MAX_COT_CHARS:-1200}
MAX_ITEM_CHARS=${MAX_ITEM_CHARS:-1400}
NEGATIVE_SAMPLING=${NEGATIVE_SAMPLING:-none}
NUM_NEGATIVES=${NUM_NEGATIVES:-0}
NEGATIVE_SEED=${NEGATIVE_SEED:-42}

BASE_EMBEDDING_MODEL=${BASE_EMBEDDING_MODEL:-/mnt/tidal-sh01/usr/xiayu6/xiayu/checkpoint/Qwen3_embedding/0.6B}
EMBEDDER_OUT=${EMBEDDER_OUT:-$ROOT/checkpoints/rrec_amazon_CDs_and_Vinyl/qwen3_embedding_cds_cot_tidal}

EMBEDDER_CUDA_VISIBLE_DEVICES=${EMBEDDER_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}
EMBEDDER_NPROC_PER_NODE=${EMBEDDER_NPROC_PER_NODE:-auto}
EMBEDDER_MASTER_PORT=${EMBEDDER_MASTER_PORT:-29522}
EMBEDDER_BATCH_SIZE=${EMBEDDER_BATCH_SIZE:-128}
EMBEDDER_GRAD_ACCUM=${EMBEDDER_GRAD_ACCUM:-1}
EMBEDDER_MAX_LENGTH=${EMBEDDER_MAX_LENGTH:-2048}
EMBEDDER_EPOCHS=${EMBEDDER_EPOCHS:-1}
EMBEDDER_MAX_STEPS=${EMBEDDER_MAX_STEPS:--1}
EMBEDDER_LR=${EMBEDDER_LR:-3e-6}
EMBEDDER_SAVE_STEPS=${EMBEDDER_SAVE_STEPS:-auto}
EMBEDDER_TORCH_DTYPE=${EMBEDDER_TORCH_DTYPE:-bfloat16}
EMBEDDER_GRADIENT_CHECKPOINTING=${EMBEDDER_GRADIENT_CHECKPOINTING:-auto}
EMBEDDER_CROSS_GPU_NEGATIVES=${EMBEDDER_CROSS_GPU_NEGATIVES:-0}
SEED=${SEED:-42}

require_path() {
  local label="$1"
  local path="$2"
  if [[ ! -e "$path" ]]; then
    echo "Missing $label: $path" >&2
    exit 1
  fi
}

require_file() {
  local label="$1"
  local path="$2"
  if [[ ! -s "$path" ]]; then
    echo "Missing or empty $label: $path" >&2
    exit 1
  fi
}

count_devices() {
  local raw="$1"
  if [[ -z "$raw" ]]; then
    echo 1
    return
  fi
  local IFS=','
  local devices=()
  read -r -a devices <<< "$raw"
  echo "${#devices[@]}"
}

resolve_nproc() {
  if [[ "$EMBEDDER_NPROC_PER_NODE" == "auto" ]]; then
    count_devices "$EMBEDDER_CUDA_VISIBLE_DEVICES"
  else
    echo "$EMBEDDER_NPROC_PER_NODE"
  fi
}

require_path "project root" "$ROOT"
require_path "python" "$PYTHON_BIN"
require_file "CoT candidate lists" "$COT_CANDIDATE_LISTS"
require_file "item info" "$ITEM_INFO"
require_path "base embedding model" "$BASE_EMBEDDING_MODEL"

mkdir -p "$OUT_DIR" "$EMBEDDER_OUT"
cd "$ROOT"

export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export MODELSCOPE_CACHE=${MODELSCOPE_CACHE:-/mnt/tidal-sh01/usr/xiayu6/xiayu/modelscope_cache}

build_args=()
if [[ "$INCLUDE_HISTORY" == "1" || "$INCLUDE_HISTORY" == "true" ]]; then
  build_args+=(--include-history)
else
  build_args+=(--no-include-history)
fi
if [[ "$INCLUDE_COT" == "1" || "$INCLUDE_COT" == "true" ]]; then
  build_args+=(--include-cot)
else
  build_args+=(--no-include-cot)
fi

if [[ "$NEGATIVE_SAMPLING" != "none" && "$NUM_NEGATIVES" -gt 0 && -s "$COT_EMBEDDER_DATASET" ]]; then
  if ! grep -m 1 -q '"negatives"' "$COT_EMBEDDER_DATASET"; then
    echo "Existing dataset has no explicit negatives; rebuilding: $COT_EMBEDDER_DATASET"
    FORCE_REBUILD_DATASET=1
  fi
fi

if [[ "$FORCE_REBUILD_DATASET" == "1" || ! -s "$COT_EMBEDDER_DATASET" ]]; then
  "$PYTHON_BIN" scripts/make_cot_embedder_dataset.py \
    --candidate-lists "$COT_CANDIDATE_LISTS" \
    --item-info "$ITEM_INFO" \
    --output "$COT_EMBEDDER_DATASET" \
    --cot-text-mode "$COT_TEXT_MODE" \
    --max-cot-chars "$MAX_COT_CHARS" \
    --max-item-chars "$MAX_ITEM_CHARS" \
    --negative-sampling "$NEGATIVE_SAMPLING" \
    --num-negatives "$NUM_NEGATIVES" \
    --negative-seed "$NEGATIVE_SEED" \
    "${build_args[@]}"
else
  echo "Using existing CoT embedder dataset: $COT_EMBEDDER_DATASET"
fi

require_file "CoT embedder dataset" "$COT_EMBEDDER_DATASET"

EMBEDDER_NPROC=$(resolve_nproc)
if ((EMBEDDER_NPROC < 1)); then
  echo "EMBEDDER_NPROC_PER_NODE must be >= 1" >&2
  exit 1
fi

if [[ "$EMBEDDER_SAVE_STEPS" == "auto" ]]; then
  row_count=$(wc -l < "$COT_EMBEDDER_DATASET" | tr -d ' ')
  global_train_batch=$((EMBEDDER_BATCH_SIZE * EMBEDDER_NPROC))
  full_batches=$((row_count / global_train_batch))
  if ((full_batches < 1)); then
    echo "Need at least one full global batch: rows=$row_count per_device_batch_size=$EMBEDDER_BATCH_SIZE nproc=$EMBEDDER_NPROC" >&2
    exit 1
  fi
  EMBEDDER_SAVE_STEPS=$(((full_batches + EMBEDDER_GRAD_ACCUM - 1) / EMBEDDER_GRAD_ACCUM))
fi

echo "ROOT=$ROOT"
echo "COT_CANDIDATE_LISTS=$COT_CANDIDATE_LISTS"
echo "COT_EMBEDDER_DATASET=$COT_EMBEDDER_DATASET"
echo "COT_TEXT_MODE=$COT_TEXT_MODE"
echo "NEGATIVE_SAMPLING=$NEGATIVE_SAMPLING"
echo "NUM_NEGATIVES=$NUM_NEGATIVES"
echo "BASE_EMBEDDING_MODEL=$BASE_EMBEDDING_MODEL"
echo "EMBEDDER_OUT=$EMBEDDER_OUT"
echo "EMBEDDER_LR=$EMBEDDER_LR"
echo "EMBEDDER_EPOCHS=$EMBEDDER_EPOCHS"
echo "EMBEDDER_SAVE_STEPS=$EMBEDDER_SAVE_STEPS"
echo "EMBEDDER_CUDA_VISIBLE_DEVICES=$EMBEDDER_CUDA_VISIBLE_DEVICES"
echo "EMBEDDER_NPROC=$EMBEDDER_NPROC"
echo "EMBEDDER_BATCH_SIZE=$EMBEDDER_BATCH_SIZE"
echo "EMBEDDER_GLOBAL_BATCH_SIZE=$((EMBEDDER_BATCH_SIZE * EMBEDDER_NPROC * EMBEDDER_GRAD_ACCUM))"
echo "EMBEDDER_GRADIENT_CHECKPOINTING=$EMBEDDER_GRADIENT_CHECKPOINTING"
echo "EMBEDDER_CROSS_GPU_NEGATIVES=$EMBEDDER_CROSS_GPU_NEGATIVES"

train_args=(
  scripts/train_phase0_embedder.py
  --model "$BASE_EMBEDDING_MODEL" \
  --dataset "$COT_EMBEDDER_DATASET" \
  --output-dir "$EMBEDDER_OUT" \
  --max-length "$EMBEDDER_MAX_LENGTH" \
  --batch-size "$EMBEDDER_BATCH_SIZE" \
  --grad-accum "$EMBEDDER_GRAD_ACCUM" \
  --epochs "$EMBEDDER_EPOCHS" \
  --max-steps "$EMBEDDER_MAX_STEPS" \
  --learning-rate "$EMBEDDER_LR" \
  --torch-dtype "$EMBEDDER_TORCH_DTYPE" \
  --save-steps "$EMBEDDER_SAVE_STEPS" \
  --gradient-checkpointing "$EMBEDDER_GRADIENT_CHECKPOINTING" \
  --seed "$SEED"
)
if [[ "$EMBEDDER_CROSS_GPU_NEGATIVES" == "1" || "$EMBEDDER_CROSS_GPU_NEGATIVES" == "true" ]]; then
  train_args+=(--cross-gpu-negatives)
else
  train_args+=(--no-cross-gpu-negatives)
fi

if ((EMBEDDER_NPROC > 1)); then
  CUDA_VISIBLE_DEVICES="$EMBEDDER_CUDA_VISIBLE_DEVICES" "$PYTHON_BIN" -m torch.distributed.run \
    --nproc_per_node "$EMBEDDER_NPROC" \
    --master_port "$EMBEDDER_MASTER_PORT" \
    "${train_args[@]}"
else
  CUDA_VISIBLE_DEVICES="$EMBEDDER_CUDA_VISIBLE_DEVICES" "$PYTHON_BIN" "${train_args[@]}"
fi

latest_checkpoint=$(find "$EMBEDDER_OUT" -type d -name 'checkpoint-*' -print 2>/dev/null | sort -V | tail -n 1)
if [[ -n "$latest_checkpoint" ]]; then
  echo "QWEN3_EMBEDDING_MODEL=$latest_checkpoint"
fi
