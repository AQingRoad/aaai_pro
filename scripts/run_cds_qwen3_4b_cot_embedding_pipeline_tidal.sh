#!/usr/bin/env bash
set -euo pipefail

# Generate one CoT per CDs_and_Vinyl training example with local Qwen3-4B,
# then train a CoT-aware Qwen3 embedding model.
#
# Stages can be controlled with:
#   RUN_GENERATE=1 RUN_BUILD_DATASET=1 RUN_TRAIN_EMBEDDER=1

ROOT=${ROOT:-/mnt/tidal-sh01/usr/xiayu6/xiayu/aaai_pro}
VENV=${VENV:-/root/miniconda3/envs/swift}
PYTHON_BIN=${PYTHON_BIN:-$VENV/bin/python}

CATEGORY=${CATEGORY:-CDs_and_Vinyl}
OUT_DIR=${OUT_DIR:-$ROOT/outputs/rrec_amazon/$CATEGORY}
TRAIN_EXAMPLES=${TRAIN_EXAMPLES:-$ROOT/data/rrec_amazon/$CATEGORY/examples.jsonl}
ITEM_INFO=${ITEM_INFO:-$ROOT/github_artifacts/CDs_and_Vinyl/rrec_eval/item_info.jsonl}

LLM_MODEL=${LLM_MODEL:-/mnt/tidal-sh01/usr/xiayu6/xiayu/checkpoint/Qwen3/4B}
COT_OUTPUT=${COT_OUTPUT:-$OUT_DIR/cot_candidate_one_lists_qwen3_4b_local.jsonl}
COT_SHARD_DIR=${COT_SHARD_DIR:-$COT_OUTPUT.shards}
RUN_NAME=${RUN_NAME:-qwen3_4b_local_cot}

DEVICES=${DEVICES:-0,1,2,3,4,5,6,7}
NUM_SHARDS=${NUM_SHARDS:-}
GENERATION_BATCH_SIZE=${GENERATION_BATCH_SIZE:-8}
MAX_EXAMPLES=${MAX_EXAMPLES:-0}
MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS:-2048}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
TEMPERATURE=${TEMPERATURE:-0.2}
TOP_P=${TOP_P:-0.9}
TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
MODEL_DEVICE=${MODEL_DEVICE:-cuda:0}
RESUME=${RESUME:-1}
GEN_AGGREGATE_EVERY=${GEN_AGGREGATE_EVERY:-20}

COT_TEXT_MODE=${COT_TEXT_MODE:-answer}
COT_EMBEDDER_DATASET=${COT_EMBEDDER_DATASET:-$OUT_DIR/phase0_embedder_cds_with_${RUN_NAME}_${COT_TEXT_MODE}.jsonl}
INCLUDE_HISTORY=${INCLUDE_HISTORY:-1}
INCLUDE_COT=${INCLUDE_COT:-1}
MAX_COT_CHARS=${MAX_COT_CHARS:-1200}
MAX_ITEM_CHARS=${MAX_ITEM_CHARS:-1400}

BASE_EMBEDDING_MODEL=${BASE_EMBEDDING_MODEL:-/mnt/tidal-sh01/usr/xiayu6/xiayu/checkpoint/Qwen3_embedding/0.6B}
EMBEDDER_OUT=${EMBEDDER_OUT:-$ROOT/checkpoints/rrec_amazon_CDs_and_Vinyl/qwen3_embedding_cds_cot_${RUN_NAME}}
EMBEDDER_CUDA_VISIBLE_DEVICES=${EMBEDDER_CUDA_VISIBLE_DEVICES:-$DEVICES}
EMBEDDER_NPROC_PER_NODE=${EMBEDDER_NPROC_PER_NODE:-auto}
EMBEDDER_MASTER_PORT=${EMBEDDER_MASTER_PORT:-29523}
EMBEDDER_BATCH_SIZE=${EMBEDDER_BATCH_SIZE:-128}
EMBEDDER_GRAD_ACCUM=${EMBEDDER_GRAD_ACCUM:-1}
EMBEDDER_MAX_LENGTH=${EMBEDDER_MAX_LENGTH:-2048}
EMBEDDER_EPOCHS=${EMBEDDER_EPOCHS:-1}
EMBEDDER_MAX_STEPS=${EMBEDDER_MAX_STEPS:--1}
EMBEDDER_LR=${EMBEDDER_LR:-3e-6}
EMBEDDER_SAVE_STEPS=${EMBEDDER_SAVE_STEPS:-auto}
EMBEDDER_TORCH_DTYPE=${EMBEDDER_TORCH_DTYPE:-bfloat16}
SEED=${SEED:-42}

RUN_GENERATE=${RUN_GENERATE:-1}
RUN_BUILD_DATASET=${RUN_BUILD_DATASET:-1}
RUN_TRAIN_EMBEDDER=${RUN_TRAIN_EMBEDDER:-1}
FORCE_REBUILD_DATASET=${FORCE_REBUILD_DATASET:-0}

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

resolve_embedder_nproc() {
  if [[ "$EMBEDDER_NPROC_PER_NODE" == "auto" ]]; then
    count_devices "$EMBEDDER_CUDA_VISIBLE_DEVICES"
  else
    echo "$EMBEDDER_NPROC_PER_NODE"
  fi
}

require_path "project root" "$ROOT"
require_path "python" "$PYTHON_BIN"
require_file "train examples" "$TRAIN_EXAMPLES"
require_file "item info" "$ITEM_INFO"
require_path "Qwen3-4B LLM model" "$LLM_MODEL"
require_path "base embedding model" "$BASE_EMBEDDING_MODEL"

IFS=',' read -r -a DEVICE_LIST <<< "$DEVICES"
if [[ "${#DEVICE_LIST[@]}" -eq 0 ]]; then
  echo "DEVICES is empty" >&2
  exit 1
fi
if [[ -z "$NUM_SHARDS" ]]; then
  NUM_SHARDS=${#DEVICE_LIST[@]}
fi
if (( NUM_SHARDS < 1 )); then
  echo "NUM_SHARDS must be >= 1" >&2
  exit 1
fi
if (( NUM_SHARDS > ${#DEVICE_LIST[@]} )); then
  echo "NUM_SHARDS=$NUM_SHARDS is larger than number of DEVICES=${#DEVICE_LIST[@]}" >&2
  exit 1
fi

cd "$ROOT"
mkdir -p "$OUT_DIR" "$COT_SHARD_DIR" "$EMBEDDER_OUT"

export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export MODELSCOPE_CACHE=${MODELSCOPE_CACHE:-/mnt/tidal-sh01/usr/xiayu6/xiayu/modelscope_cache}

echo "ROOT=$ROOT"
echo "TRAIN_EXAMPLES=$TRAIN_EXAMPLES"
echo "ITEM_INFO=$ITEM_INFO"
echo "LLM_MODEL=$LLM_MODEL"
echo "COT_OUTPUT=$COT_OUTPUT"
echo "DEVICES=$DEVICES"
echo "NUM_SHARDS=$NUM_SHARDS"
echo "GENERATION_BATCH_SIZE=$GENERATION_BATCH_SIZE"
echo "MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
echo "TEMPERATURE=$TEMPERATURE"
echo "COT_EMBEDDER_DATASET=$COT_EMBEDDER_DATASET"
echo "BASE_EMBEDDING_MODEL=$BASE_EMBEDDING_MODEL"
echo "EMBEDDER_OUT=$EMBEDDER_OUT"
echo "EMBEDDER_CUDA_VISIBLE_DEVICES=$EMBEDDER_CUDA_VISIBLE_DEVICES"

if [[ "$RUN_GENERATE" == "1" ]]; then
  pids=()
  shard_files=()
  for shard in $(seq 0 $((NUM_SHARDS - 1))); do
    device="${DEVICE_LIST[$shard]}"
    shard_output="$COT_SHARD_DIR/shard${shard}.jsonl"
    shard_log="$COT_SHARD_DIR/shard${shard}.log"
    shard_files+=("$shard_output")
    echo "Launching generation shard $shard/$NUM_SHARDS on GPU $device -> $shard_log"
    (
      CUDA_VISIBLE_DEVICES="$device" \
      COT_MODEL_DEVICE="$MODEL_DEVICE" \
      "$PYTHON_BIN" scripts/generate_cot_candidate_lists_local.py \
        --input "$TRAIN_EXAMPLES" \
        --output "$shard_output" \
        --model "$LLM_MODEL" \
        --num-shards "$NUM_SHARDS" \
        --shard-index "$shard" \
        --generation-batch-size "$GENERATION_BATCH_SIZE" \
        --max-examples "$MAX_EXAMPLES" \
        --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        --temperature "$TEMPERATURE" \
        --top-p "$TOP_P" \
        --torch-dtype "$TORCH_DTYPE" \
        --device "$MODEL_DEVICE" \
        --aggregate-every "$GEN_AGGREGATE_EVERY" \
        --seed "$SEED" \
        $( [[ "$RESUME" == "1" || "$RESUME" == "true" ]] && printf '%s' '--resume' || printf '%s' '--no-resume' )
    ) >"$shard_log" 2>&1 &
    pids+=("$!")
  done

  failed=0
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
  if [[ "$failed" != "0" ]]; then
    echo "At least one generation shard failed. Check logs under $COT_SHARD_DIR" >&2
    exit 1
  fi

  "$PYTHON_BIN" scripts/aggregate_cot_candidate_list_shards.py \
    --input "$TRAIN_EXAMPLES" \
    --shards "${shard_files[@]}" \
    --output "$COT_OUTPUT" \
    --max-examples "$MAX_EXAMPLES"
else
  echo "Skipping generation: $COT_OUTPUT"
fi

if [[ "$RUN_BUILD_DATASET" == "1" ]]; then
  require_file "CoT output" "$COT_OUTPUT"
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
  if [[ "$FORCE_REBUILD_DATASET" == "1" || ! -s "$COT_EMBEDDER_DATASET" ]]; then
    "$PYTHON_BIN" scripts/make_cot_embedder_dataset.py \
      --candidate-lists "$COT_OUTPUT" \
      --item-info "$ITEM_INFO" \
      --output "$COT_EMBEDDER_DATASET" \
      --cot-text-mode "$COT_TEXT_MODE" \
      --max-cot-chars "$MAX_COT_CHARS" \
      --max-item-chars "$MAX_ITEM_CHARS" \
      "${build_args[@]}"
  else
    echo "Using existing CoT embedder dataset: $COT_EMBEDDER_DATASET"
  fi
else
  echo "Skipping CoT embedder dataset build: $COT_EMBEDDER_DATASET"
fi

if [[ "$RUN_TRAIN_EMBEDDER" == "1" ]]; then
  require_file "CoT embedder dataset" "$COT_EMBEDDER_DATASET"
  EMBEDDER_NPROC=$(resolve_embedder_nproc)
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

  echo "EMBEDDER_NPROC=$EMBEDDER_NPROC"
  echo "EMBEDDER_BATCH_SIZE=$EMBEDDER_BATCH_SIZE"
  echo "EMBEDDER_GLOBAL_BATCH_SIZE=$((EMBEDDER_BATCH_SIZE * EMBEDDER_NPROC * EMBEDDER_GRAD_ACCUM))"
  echo "EMBEDDER_SAVE_STEPS=$EMBEDDER_SAVE_STEPS"

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
    --seed "$SEED"
  )

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
else
  echo "Skipping embedding training: $EMBEDDER_OUT"
fi

echo "Done."
echo "  cot_output:       $COT_OUTPUT"
echo "  cot_shards:       $COT_SHARD_DIR"
echo "  embedder_dataset: $COT_EMBEDDER_DATASET"
echo "  embedder_out:     $EMBEDDER_OUT"
