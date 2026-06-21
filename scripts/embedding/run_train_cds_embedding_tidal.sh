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

EMBEDDER_CUDA_VISIBLE_DEVICES=${EMBEDDER_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}
EMBEDDER_NPROC_PER_NODE=${EMBEDDER_NPROC_PER_NODE:-auto}
EMBEDDER_MASTER_PORT=${EMBEDDER_MASTER_PORT:-29521}
EMBEDDER_BATCH_SIZE=${EMBEDDER_BATCH_SIZE:-128}
EMBEDDER_GRAD_ACCUM=${EMBEDDER_GRAD_ACCUM:-1}
EMBEDDER_MAX_LENGTH=${EMBEDDER_MAX_LENGTH:-2048}
EMBEDDER_EPOCHS=${EMBEDDER_EPOCHS:-3}
EMBEDDER_MAX_STEPS=${EMBEDDER_MAX_STEPS:--1}
EMBEDDER_LR=${EMBEDDER_LR:-6e-6}
EMBEDDER_SAVE_STEPS=${EMBEDDER_SAVE_STEPS:-auto}
EMBEDDER_TORCH_DTYPE=${EMBEDDER_TORCH_DTYPE:-bfloat16}
EMBEDDER_GRADIENT_CHECKPOINTING=${EMBEDDER_GRADIENT_CHECKPOINTING:-auto}
EMBEDDER_CROSS_GPU_NEGATIVES=${EMBEDDER_CROSS_GPU_NEGATIVES:-0}
EMBEDDER_SYNC_BARRIERS=${EMBEDDER_SYNC_BARRIERS:-0}

RUN_HARD_NEGATIVES=${RUN_HARD_NEGATIVES:-0}
HARD_NEGATIVE_DATASET=${HARD_NEGATIVE_DATASET:-${EMBEDDER_DATASET%.jsonl}.hardneg.jsonl}
HARD_NEGATIVE_EMBEDDING_MODEL=${HARD_NEGATIVE_EMBEDDING_MODEL:-$BASE_EMBEDDING_MODEL}
HARD_NEGATIVE_ITEM_INFO=${HARD_NEGATIVE_ITEM_INFO:-$ROOT/github_artifacts/CDs_and_Vinyl/rrec_eval/item_info.jsonl}
HARD_NEGATIVE_ITEM_SUMMARY=${HARD_NEGATIVE_ITEM_SUMMARY:-}
HARD_NEGATIVE_NUM_NEGATIVES=${HARD_NEGATIVE_NUM_NEGATIVES:-8}
HARD_NEGATIVE_CANDIDATE_POOL=${HARD_NEGATIVE_CANDIDATE_POOL:-200}
HARD_NEGATIVE_QUERY_BATCH_SIZE=${HARD_NEGATIVE_QUERY_BATCH_SIZE:-32}
HARD_NEGATIVE_EMBEDDING_BATCH_SIZE=${HARD_NEGATIVE_EMBEDDING_BATCH_SIZE:-128}
HARD_NEGATIVE_MAX_LENGTH=${HARD_NEGATIVE_MAX_LENGTH:-$EMBEDDER_MAX_LENGTH}
HARD_NEGATIVE_MAX_ITEM_CHARS=${HARD_NEGATIVE_MAX_ITEM_CHARS:-1800}
HARD_NEGATIVE_CUDA_VISIBLE_DEVICES=${HARD_NEGATIVE_CUDA_VISIBLE_DEVICES:-${EMBEDDER_CUDA_VISIBLE_DEVICES%%,*}}
HARD_NEGATIVE_DEVICE=${HARD_NEGATIVE_DEVICE:-cuda:0}
HARD_NEGATIVE_ONLY_ABOVE_TARGET=${HARD_NEGATIVE_ONLY_ABOVE_TARGET:-1}
HARD_NEGATIVE_FORCE=${HARD_NEGATIVE_FORCE:-0}
NCCL_NET=${NCCL_NET:-Socket}
NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
NCCL_MNNVL_ENABLE=${NCCL_MNNVL_ENABLE:-0}
NCCL_COLLNET_ENABLE=${NCCL_COLLNET_ENABLE:-0}
NCCL_DEBUG=${NCCL_DEBUG:-WARN}
TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
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
require_path "base embedding model" "$BASE_EMBEDDING_MODEL"
require_path "phase0 embedder dataset" "$EMBEDDER_DATASET"

if [[ "$RUN_HARD_NEGATIVES" == "1" || "$RUN_HARD_NEGATIVES" == "true" ]]; then
  require_path "hard negative mining embedding model" "$HARD_NEGATIVE_EMBEDDING_MODEL"
  require_path "hard negative item_info" "$HARD_NEGATIVE_ITEM_INFO"
fi

if [[ "$RUN_HARD_NEGATIVES" == "1" || "$RUN_HARD_NEGATIVES" == "true" ]]; then
  if [[ "$HARD_NEGATIVE_FORCE" == "1" || "$HARD_NEGATIVE_FORCE" == "true" || ! -s "$HARD_NEGATIVE_DATASET" ]]; then
    echo "Building hard-negative embedder dataset: $HARD_NEGATIVE_DATASET"
    hard_negative_args=(
      scripts/data/add_hard_negatives_to_embedder_dataset.py
      --input "$EMBEDDER_DATASET"
      --output "$HARD_NEGATIVE_DATASET"
      --item-info "$HARD_NEGATIVE_ITEM_INFO"
      --embedding-model "$HARD_NEGATIVE_EMBEDDING_MODEL"
      --num-negatives "$HARD_NEGATIVE_NUM_NEGATIVES"
      --candidate-pool "$HARD_NEGATIVE_CANDIDATE_POOL"
      --query-batch-size "$HARD_NEGATIVE_QUERY_BATCH_SIZE"
      --embedding-batch-size "$HARD_NEGATIVE_EMBEDDING_BATCH_SIZE"
      --embedding-max-length "$HARD_NEGATIVE_MAX_LENGTH"
      --max-item-chars "$HARD_NEGATIVE_MAX_ITEM_CHARS"
      --torch-dtype "$EMBEDDER_TORCH_DTYPE"
      --device "$HARD_NEGATIVE_DEVICE"
    )
    if [[ -n "$HARD_NEGATIVE_ITEM_SUMMARY" ]]; then
      hard_negative_args+=(--item-summary "$HARD_NEGATIVE_ITEM_SUMMARY")
    fi
    if [[ "$HARD_NEGATIVE_ONLY_ABOVE_TARGET" == "1" || "$HARD_NEGATIVE_ONLY_ABOVE_TARGET" == "true" ]]; then
      hard_negative_args+=(--only-above-target)
    else
      hard_negative_args+=(--no-only-above-target)
    fi
    CUDA_VISIBLE_DEVICES="$HARD_NEGATIVE_CUDA_VISIBLE_DEVICES" "$PYTHON_BIN" "${hard_negative_args[@]}"
  else
    echo "Using existing hard-negative dataset: $HARD_NEGATIVE_DATASET"
  fi
  EMBEDDER_DATASET="$HARD_NEGATIVE_DATASET"
  require_path "hard-negative embedder dataset" "$EMBEDDER_DATASET"
fi

EMBEDDER_NPROC=$(resolve_nproc)
if ((EMBEDDER_NPROC < 1)); then
  echo "EMBEDDER_NPROC_PER_NODE must be >= 1" >&2
  exit 1
fi

if [[ "$EMBEDDER_SAVE_STEPS" == "auto" ]]; then
  row_count=$(wc -l < "$EMBEDDER_DATASET" | tr -d ' ')
  global_train_batch=$((EMBEDDER_BATCH_SIZE * EMBEDDER_NPROC))
  full_batches=$((row_count / global_train_batch))
  if ((full_batches < 1)); then
    echo "Need at least one full global batch: rows=$row_count per_device_batch_size=$EMBEDDER_BATCH_SIZE nproc=$EMBEDDER_NPROC" >&2
    exit 1
  fi
  EMBEDDER_SAVE_STEPS=$(((full_batches + EMBEDDER_GRAD_ACCUM - 1) / EMBEDDER_GRAD_ACCUM))
fi

mkdir -p "$EMBEDDER_OUT"
cd "$ROOT"

export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export MODELSCOPE_CACHE=${MODELSCOPE_CACHE:-/mnt/tidal-sh01/usr/xiayu6/xiayu/modelscope_cache}
export NCCL_NET
export NCCL_IB_DISABLE
export NCCL_P2P_DISABLE
export NCCL_NVLS_ENABLE
export NCCL_MNNVL_ENABLE
export NCCL_COLLNET_ENABLE
export NCCL_DEBUG
export TORCH_NCCL_ASYNC_ERROR_HANDLING

echo "ROOT=$ROOT"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "BASE_EMBEDDING_MODEL=$BASE_EMBEDDING_MODEL"
echo "EMBEDDER_DATASET=$EMBEDDER_DATASET"
echo "EMBEDDER_OUT=$EMBEDDER_OUT"
echo "EMBEDDER_CUDA_VISIBLE_DEVICES=$EMBEDDER_CUDA_VISIBLE_DEVICES"
echo "EMBEDDER_NPROC=$EMBEDDER_NPROC"
echo "EMBEDDER_BATCH_SIZE=$EMBEDDER_BATCH_SIZE"
echo "EMBEDDER_GLOBAL_BATCH_SIZE=$((EMBEDDER_BATCH_SIZE * EMBEDDER_NPROC * EMBEDDER_GRAD_ACCUM))"
echo "EMBEDDER_SAVE_STEPS=$EMBEDDER_SAVE_STEPS"
echo "EMBEDDER_GRADIENT_CHECKPOINTING=$EMBEDDER_GRADIENT_CHECKPOINTING"
echo "EMBEDDER_CROSS_GPU_NEGATIVES=$EMBEDDER_CROSS_GPU_NEGATIVES"
echo "EMBEDDER_SYNC_BARRIERS=$EMBEDDER_SYNC_BARRIERS"
echo "RUN_HARD_NEGATIVES=$RUN_HARD_NEGATIVES"
if [[ "$RUN_HARD_NEGATIVES" == "1" || "$RUN_HARD_NEGATIVES" == "true" ]]; then
  echo "HARD_NEGATIVE_DATASET=$HARD_NEGATIVE_DATASET"
  echo "HARD_NEGATIVE_EMBEDDING_MODEL=$HARD_NEGATIVE_EMBEDDING_MODEL"
  echo "HARD_NEGATIVE_NUM_NEGATIVES=$HARD_NEGATIVE_NUM_NEGATIVES"
  echo "HARD_NEGATIVE_CANDIDATE_POOL=$HARD_NEGATIVE_CANDIDATE_POOL"
fi

train_args=(
  scripts/train_phase0_embedder.py
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
  --gradient-checkpointing "$EMBEDDER_GRADIENT_CHECKPOINTING" \
  --seed "$SEED"
)
if [[ "$EMBEDDER_CROSS_GPU_NEGATIVES" == "1" || "$EMBEDDER_CROSS_GPU_NEGATIVES" == "true" ]]; then
  train_args+=(--cross-gpu-negatives)
else
  train_args+=(--no-cross-gpu-negatives)
fi
if [[ "$EMBEDDER_SYNC_BARRIERS" == "1" || "$EMBEDDER_SYNC_BARRIERS" == "true" ]]; then
  train_args+=(--sync-barriers)
else
  train_args+=(--no-sync-barriers)
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
