#!/usr/bin/env bash
set -euo pipefail

# CDs pipeline:
#   1. Generate/fill item description summaries with local Qwen/vLLM.
#   2. Rebuild train examples so user_history includes item metadata.
#   3. Generate one CoT per train example from the metadata-enhanced prompt.
#
# Stages:
#   RUN_SUMMARY=1 RUN_BUILD_META_EXAMPLES=1 RUN_GENERATE_COT=1

ROOT=${ROOT:-/mnt/tidal-sh01/usr/xiayu6/xiayu/aaai_pro}
VENV=${VENV:-/root/miniconda3/envs/swift}
PYTHON_BIN=${PYTHON_BIN:-$VENV/bin/python}

CATEGORY=${CATEGORY:-CDs_and_Vinyl}
OUT_DIR=${OUT_DIR:-$ROOT/outputs/rrec_amazon/$CATEGORY}
RREC_EVAL_DIR=${RREC_EVAL_DIR:-$ROOT/github_artifacts/$CATEGORY/rrec_eval}
ITEM_INFO=${ITEM_INFO:-$RREC_EVAL_DIR/item_info.jsonl}
TRAIN_EXAMPLES=${TRAIN_EXAMPLES:-$ROOT/data/rrec_amazon/$CATEGORY/examples.jsonl}

SUMMARY_RUN_NAME=${SUMMARY_RUN_NAME:-qwen3_32b_desc60}
SUMMARY_MODEL=${SUMMARY_MODEL:-/mnt/tidal-sh01/usr/xiayu6/xiayu/checkpoint/Qwen3/32B}
ITEM_METADATA_SUMMARY=${ITEM_METADATA_SUMMARY:-$RREC_EVAL_DIR/item_metadata_summary_${SUMMARY_RUN_NAME}.jsonl}
SUMMARY_MAX_WORDS=${SUMMARY_MAX_WORDS:-60}
SUMMARY_MAX_NEW_TOKENS=${SUMMARY_MAX_NEW_TOKENS:-2048}

RUN_NAME=${RUN_NAME:-qwen3_32b_meta_desc60}
META_EXAMPLES=${META_EXAMPLES:-$ROOT/data/rrec_amazon/$CATEGORY/examples_${RUN_NAME}.jsonl}
COT_MODEL=${COT_MODEL:-$SUMMARY_MODEL}
COT_OUTPUT=${COT_OUTPUT:-$OUT_DIR/cot_candidate_one_lists_${RUN_NAME}.jsonl}
TOKENIZER=${TOKENIZER:-}

DEVICES=${DEVICES:-0,1,2,3,4,5,6,7}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-auto}
GENERATION_BATCH_SIZE=${GENERATION_BATCH_SIZE:-32}
MAX_EXAMPLES=${MAX_EXAMPLES:-0}
MAX_HISTORY_ITEMS=${MAX_HISTORY_ITEMS:-20}
HISTORY_METADATA_MODE=${HISTORY_METADATA_MODE:-summary}
HISTORY_MAX_ITEM_CHARS=${HISTORY_MAX_ITEM_CHARS:-420}
MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS:-2048}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
TEMPERATURE=${TEMPERATURE:-0.2}
TOP_P=${TOP_P:-0.9}
VLLM_DTYPE=${VLLM_DTYPE:-bfloat16}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-4096}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-$GENERATION_BATCH_SIZE}
VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-0}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.85}
VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-0}
VLLM_SAFE_NCCL_DEFAULTS=${VLLM_SAFE_NCCL_DEFAULTS:-1}
VLLM_DISABLE_CUSTOM_ALL_REDUCE=${VLLM_DISABLE_CUSTOM_ALL_REDUCE:-1}
VLLM_DISTRIBUTED_EXECUTOR_BACKEND=${VLLM_DISTRIBUTED_EXECUTOR_BACKEND:-mp}
VLLM_WORKER_MULTIPROC_METHOD=${VLLM_WORKER_MULTIPROC_METHOD:-spawn}
NCCL_NET=${NCCL_NET:-Socket}
NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
NCCL_MNNVL_ENABLE=${NCCL_MNNVL_ENABLE:-0}
NCCL_COLLNET_ENABLE=${NCCL_COLLNET_ENABLE:-0}
NCCL_DEBUG=${NCCL_DEBUG:-WARN}
TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
SAVE_EVERY=${SAVE_EVERY:-100}
SEED=${SEED:-42}
RESUME=${RESUME:-1}

RUN_SUMMARY=${RUN_SUMMARY:-1}
RUN_BUILD_META_EXAMPLES=${RUN_BUILD_META_EXAMPLES:-1}
RUN_GENERATE_COT=${RUN_GENERATE_COT:-1}
FORCE_BUILD_META_EXAMPLES=${FORCE_BUILD_META_EXAMPLES:-0}

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

require_path "project root" "$ROOT"
require_path "python" "$PYTHON_BIN"
require_file "item_info" "$ITEM_INFO"
require_file "train examples" "$TRAIN_EXAMPLES"
if [[ "$RUN_SUMMARY" == "1" || "$RUN_SUMMARY" == "true" ]]; then
  require_path "summary model" "$SUMMARY_MODEL"
else
  require_file "item metadata summary" "$ITEM_METADATA_SUMMARY"
fi
if [[ "$RUN_GENERATE_COT" == "1" || "$RUN_GENERATE_COT" == "true" ]]; then
  require_path "CoT model" "$COT_MODEL"
fi

if [[ "$TENSOR_PARALLEL_SIZE" == "auto" ]]; then
  TENSOR_PARALLEL_SIZE=$(count_devices "$DEVICES")
fi

cd "$ROOT"
mkdir -p "$OUT_DIR" "$(dirname "$META_EXAMPLES")" "$(dirname "$ITEM_METADATA_SUMMARY")"

export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export VLLM_SAFE_NCCL_DEFAULTS
export VLLM_DISABLE_CUSTOM_ALL_REDUCE
export VLLM_DISTRIBUTED_EXECUTOR_BACKEND
export VLLM_WORKER_MULTIPROC_METHOD
export NCCL_NET
export NCCL_IB_DISABLE
export NCCL_P2P_DISABLE
export NCCL_NVLS_ENABLE
export NCCL_MNNVL_ENABLE
export NCCL_COLLNET_ENABLE
export NCCL_DEBUG
export TORCH_NCCL_ASYNC_ERROR_HANDLING

echo "ROOT=$ROOT"
echo "ITEM_INFO=$ITEM_INFO"
echo "TRAIN_EXAMPLES=$TRAIN_EXAMPLES"
echo "ITEM_METADATA_SUMMARY=$ITEM_METADATA_SUMMARY"
echo "META_EXAMPLES=$META_EXAMPLES"
echo "COT_OUTPUT=$COT_OUTPUT"
echo "SUMMARY_MODEL=$SUMMARY_MODEL"
echo "COT_MODEL=$COT_MODEL"
echo "DEVICES=$DEVICES"
echo "TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE"
echo "GENERATION_BATCH_SIZE=$GENERATION_BATCH_SIZE"
echo "MAX_EXAMPLES=$MAX_EXAMPLES"
echo "MAX_PROMPT_TOKENS=$MAX_PROMPT_TOKENS"
echo "MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
echo "HISTORY_METADATA_MODE=$HISTORY_METADATA_MODE"
echo "HISTORY_MAX_ITEM_CHARS=$HISTORY_MAX_ITEM_CHARS"

if [[ "$RUN_SUMMARY" == "1" || "$RUN_SUMMARY" == "true" ]]; then
  echo "Stage 1/3: item description summary"
  ROOT="$ROOT" \
  VENV="$VENV" \
  PYTHON_BIN="$PYTHON_BIN" \
  CATEGORY="$CATEGORY" \
  RREC_EVAL_DIR="$RREC_EVAL_DIR" \
  ITEM_INFO="$ITEM_INFO" \
  MODEL="$SUMMARY_MODEL" \
  TOKENIZER="$TOKENIZER" \
  OUTPUT="$ITEM_METADATA_SUMMARY" \
  RUN_NAME="$SUMMARY_RUN_NAME" \
  DEVICES="$DEVICES" \
  TENSOR_PARALLEL_SIZE="$TENSOR_PARALLEL_SIZE" \
  MAX_EXAMPLES="$MAX_EXAMPLES" \
  GENERATION_BATCH_SIZE="$GENERATION_BATCH_SIZE" \
  SUMMARY_MAX_WORDS="$SUMMARY_MAX_WORDS" \
  MAX_PROMPT_TOKENS="$MAX_PROMPT_TOKENS" \
  MAX_NEW_TOKENS="$SUMMARY_MAX_NEW_TOKENS" \
  TEMPERATURE=0.0 \
  TOP_P="$TOP_P" \
  VLLM_DTYPE="$VLLM_DTYPE" \
  VLLM_MAX_MODEL_LEN="$VLLM_MAX_MODEL_LEN" \
  VLLM_MAX_NUM_SEQS="$VLLM_MAX_NUM_SEQS" \
  VLLM_MAX_NUM_BATCHED_TOKENS="$VLLM_MAX_NUM_BATCHED_TOKENS" \
  VLLM_GPU_MEMORY_UTILIZATION="$VLLM_GPU_MEMORY_UTILIZATION" \
  VLLM_ENFORCE_EAGER="$VLLM_ENFORCE_EAGER" \
  SAVE_EVERY="$SAVE_EVERY" \
  RESUME="$RESUME" \
  SEED="$SEED" \
  bash scripts/inference/run_summarize_cds_item_descriptions_vllm_tidal.sh
else
  echo "Skipping Stage 1/3: $ITEM_METADATA_SUMMARY"
fi

require_file "item metadata summary" "$ITEM_METADATA_SUMMARY"

if [[ "$RUN_BUILD_META_EXAMPLES" == "1" || "$RUN_BUILD_META_EXAMPLES" == "true" ]]; then
  echo "Stage 2/3: rebuild examples with metadata user_history"
  if [[ "$FORCE_BUILD_META_EXAMPLES" == "1" || "$FORCE_BUILD_META_EXAMPLES" == "true" || ! -s "$META_EXAMPLES" ]]; then
    "$PYTHON_BIN" scripts/data/rewrite_examples_with_item_metadata.py \
      --input "$TRAIN_EXAMPLES" \
      --output "$META_EXAMPLES" \
      --item-info "$ITEM_INFO" \
      --item-summary "$ITEM_METADATA_SUMMARY" \
      --category "$CATEGORY" \
      --max-examples "$MAX_EXAMPLES" \
      --max-history-items "$MAX_HISTORY_ITEMS" \
      --history-metadata-mode "$HISTORY_METADATA_MODE" \
      --history-max-item-chars "$HISTORY_MAX_ITEM_CHARS"
  else
    echo "Using existing metadata examples: $META_EXAMPLES"
  fi
else
  echo "Skipping Stage 2/3: $META_EXAMPLES"
fi

require_file "metadata examples" "$META_EXAMPLES"

if [[ "$RUN_GENERATE_COT" == "1" || "$RUN_GENERATE_COT" == "true" ]]; then
  echo "Stage 3/3: generate one CoT per metadata example"
  tokenizer_args=()
  if [[ -n "$TOKENIZER" ]]; then
    tokenizer_args+=(--tokenizer "$TOKENIZER")
  fi
  eager_args=()
  if [[ "$VLLM_ENFORCE_EAGER" == "1" || "$VLLM_ENFORCE_EAGER" == "true" ]]; then
    eager_args+=(--enforce-eager)
  fi
  custom_all_reduce_args=()
  if [[ "$VLLM_DISABLE_CUSTOM_ALL_REDUCE" == "1" || "$VLLM_DISABLE_CUSTOM_ALL_REDUCE" == "true" ]]; then
    custom_all_reduce_args+=(--disable-custom-all-reduce)
  else
    custom_all_reduce_args+=(--no-disable-custom-all-reduce)
  fi
  executor_args=()
  if [[ -n "$VLLM_DISTRIBUTED_EXECUTOR_BACKEND" ]]; then
    executor_args+=(--distributed-executor-backend "$VLLM_DISTRIBUTED_EXECUTOR_BACKEND")
  fi
  resume_args=()
  if [[ "$RESUME" == "1" || "$RESUME" == "true" ]]; then
    resume_args+=(--resume)
  else
    resume_args+=(--no-resume)
  fi

  CUDA_VISIBLE_DEVICES="$DEVICES" \
  "$PYTHON_BIN" scripts/inference/vllm_batch_infer_jsonl.py \
    --task cot_generation \
    --input "$META_EXAMPLES" \
    --output "$COT_OUTPUT" \
    --model "$COT_MODEL" \
    "${tokenizer_args[@]}" \
    --max-examples "$MAX_EXAMPLES" \
    --generation-batch-size "$GENERATION_BATCH_SIZE" \
    --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top-p "$TOP_P" \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --vllm-dtype "$VLLM_DTYPE" \
    --vllm-max-model-len "$VLLM_MAX_MODEL_LEN" \
    --vllm-max-num-seqs "$VLLM_MAX_NUM_SEQS" \
    --max-num-batched-tokens "$VLLM_MAX_NUM_BATCHED_TOKENS" \
    --gpu-memory-utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    "${eager_args[@]}" \
    "${custom_all_reduce_args[@]}" \
    "${executor_args[@]}" \
    "${resume_args[@]}" \
    --save-every "$SAVE_EVERY" \
    --seed "$SEED"
else
  echo "Skipping Stage 3/3: $COT_OUTPUT"
fi

echo "Done."
echo "ITEM_METADATA_SUMMARY=$ITEM_METADATA_SUMMARY"
echo "META_EXAMPLES=$META_EXAMPLES"
echo "COT_OUTPUT=$COT_OUTPUT"
