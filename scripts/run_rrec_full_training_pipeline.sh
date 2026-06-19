#!/usr/bin/env bash
set -euo pipefail

# End-to-end RRec/Amazon training pipeline.
#
# Typical use on a new server:
#   PIPELINE_ENV_FILE=configs/rrec_full_pipeline.example.env bash scripts/run_rrec_full_training_pipeline.sh
#
# If SFT/GRPO datasets were already built elsewhere:
#   PIPELINE_ENV_FILE=configs/rrec_train_only.example.env bash scripts/run_rrec_full_training_pipeline.sh
#
# For a cheap sanity run:
#   SMOKE=1 PIPELINE_ENV_FILE=configs/rrec_full_pipeline.example.env bash scripts/run_rrec_full_training_pipeline.sh

PIPELINE_ENV_FILE=${PIPELINE_ENV_FILE:-}
if [[ -n "$PIPELINE_ENV_FILE" ]]; then
  if [[ ! -f "$PIPELINE_ENV_FILE" ]]; then
    echo "PIPELINE_ENV_FILE does not exist: $PIPELINE_ENV_FILE" >&2
    exit 1
  fi
  set -a
  # shellcheck source=/dev/null
  source "$PIPELINE_ENV_FILE"
  set +a
fi

ROOT=${ROOT:-/root/autodl-tmp/rec/aaai_pro}
RREC_DATA_ROOT=${RREC_DATA_ROOT:-/root/autodl-tmp/rec/RRec_official/data}
VENV=${VENV:-/root/autodl-tmp/rec/ms-swift-312-cu124-venv}
PYTHON_BIN=${PYTHON_BIN:-python}
GLM_CODEPLAN_ENV=${GLM_CODEPLAN_ENV:-$ROOT/configs/glm_codeplan.env}
if [[ -f "$GLM_CODEPLAN_ENV" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$GLM_CODEPLAN_ENV"
  set +a
fi

CATEGORY=${CATEGORY:-CDs_and_Vinyl}
RUN_NAME=${RUN_NAME:-full_pipeline}
SPLIT=${SPLIT:-train}
SEED=${SEED:-20260619}

MODEL=${MODEL:-/root/autodl-tmp/modelscope_cache/models/Qwen/Qwen3-4B}
MODEL_TYPE=${MODEL_TYPE:-qwen3}
BASE_EMBEDDING_MODEL=${BASE_EMBEDDING_MODEL:-/root/autodl-tmp/modelscope_cache/models/Qwen/Qwen3-Embedding-0.6B}
QWEN3_EMBEDDING_MODEL=${QWEN3_EMBEDDING_MODEL:-$BASE_EMBEDDING_MODEL}

DATA_DIR=${DATA_DIR:-$ROOT/data/rrec_amazon/$CATEGORY}
OUT_DIR=${OUT_DIR:-$ROOT/outputs/rrec_amazon/$CATEGORY}
LOG_DIR=${LOG_DIR:-$ROOT/logs}
CKPT_ROOT=${CKPT_ROOT:-$ROOT/checkpoints/rrec_amazon_$CATEGORY}

EXAMPLES_FILE=${EXAMPLES_FILE:-$DATA_DIR/examples_${RUN_NAME}.jsonl}
EMBEDDER_DATASET=${EMBEDDER_DATASET:-$DATA_DIR/phase0_embedder_${RUN_NAME}.jsonl}
EMBEDDER_OUT=${EMBEDDER_OUT:-$CKPT_ROOT/qwen3_embedding_${RUN_NAME}}

CANDIDATE_LISTS=${CANDIDATE_LISTS:-$OUT_DIR/cot_candidate_lists_${RUN_NAME}.jsonl}
RUBRIC_SCORES=${RUBRIC_SCORES:-$OUT_DIR/cot_candidate_lists_${RUN_NAME}.rubric.jsonl}
COT_JUDGED=${COT_JUDGED:-$OUT_DIR/cot_judged_${RUN_NAME}.jsonl}
COT_SCORED=${COT_SCORED:-$OUT_DIR/cot_scored_${RUN_NAME}.jsonl}
SCORED_EXAMPLES=${SCORED_EXAMPLES:-$OUT_DIR/examples_${RUN_NAME}.jsonl}
FILTERED_COT=${FILTERED_COT:-$OUT_DIR/filtered_high_quality_cot_${RUN_NAME}.jsonl}
REJECTED_COT=${REJECTED_COT:-$OUT_DIR/rejected_cot_${RUN_NAME}.jsonl}
SFT_DATASET=${SFT_DATASET:-$OUT_DIR/sft_${RUN_NAME}.jsonl}
GRPO_DATASET=${GRPO_DATASET:-$OUT_DIR/grpo_${RUN_NAME}.jsonl}
GAIN_ITEM_INFO=${GAIN_ITEM_INFO:-$ROOT/github_artifacts/$CATEGORY/rrec_eval/item_info.jsonl}

SFT_OUT=${SFT_OUT:-$CKPT_ROOT/qwen3_4b_sft_${RUN_NAME}}
SFT_MERGED_MODEL=${SFT_MERGED_MODEL:-$CKPT_ROOT/qwen3_4b_sft_merged_${RUN_NAME}}
GRPO_OUT=${GRPO_OUT:-$CKPT_ROOT/qwen3_4b_grpo_${RUN_NAME}}

# Stage flags. Use 1/0 or auto. "auto" runs the stage when the expected output
# is missing. Training stages default to 1 because this script is the full runner.
RUN_PREPARE_EXAMPLES=${RUN_PREPARE_EXAMPLES:-auto}
RUN_EMBEDDER_DATA=${RUN_EMBEDDER_DATA:-auto}
RUN_EMBEDDER_TRAIN=${RUN_EMBEDDER_TRAIN:-1}
RUN_COT_GENERATE=${RUN_COT_GENERATE:-auto}
RUN_RUBRIC_SCORE=${RUN_RUBRIC_SCORE:-auto}
RUN_MERGE=${RUN_MERGE:-auto}
RUN_GAIN=${RUN_GAIN:-auto}
RUN_SELECT=${RUN_SELECT:-auto}
RUN_DATASETS=${RUN_DATASETS:-auto}
RUN_SFT=${RUN_SFT:-1}
RUN_SFT_MERGE=${RUN_SFT_MERGE:-0}
RUN_GRPO=${RUN_GRPO:-1}
RUN_EVAL=${RUN_EVAL:-0}
TRAIN_ONLY=${TRAIN_ONLY:-0}

if [[ "$TRAIN_ONLY" == "1" ]]; then
  RUN_PREPARE_EXAMPLES=0
  RUN_EMBEDDER_DATA=0
  RUN_COT_GENERATE=0
  RUN_RUBRIC_SCORE=0
  RUN_MERGE=0
  RUN_GAIN=0
  RUN_SELECT=0
  RUN_DATASETS=0
fi

# Data sizes.
MAX_EXAMPLES=${MAX_EXAMPLES:-0}
MAX_HISTORY_ITEMS=${MAX_HISTORY_ITEMS:-20}
MIN_HISTORY=${MIN_HISTORY:-1}
MIN_RATING=${MIN_RATING:-0}
MAX_TARGET_CHARS=${MAX_TARGET_CHARS:-1400}
EMBEDDER_CATEGORIES=${EMBEDDER_CATEGORIES:-$CATEGORY}
EMBEDDER_MAX_EXAMPLES_PER_CATEGORY=${EMBEDDER_MAX_EXAMPLES_PER_CATEGORY:-0}

# Phase 0 embedder training.
EMBEDDER_MAX_ROWS=${EMBEDDER_MAX_ROWS:-0}
EMBEDDER_MAX_LENGTH=${EMBEDDER_MAX_LENGTH:-2048}
EMBEDDER_BATCH_SIZE=${EMBEDDER_BATCH_SIZE:-16}
EMBEDDER_GRAD_ACCUM=${EMBEDDER_GRAD_ACCUM:-1}
EMBEDDER_EPOCHS=${EMBEDDER_EPOCHS:-1}
EMBEDDER_MAX_STEPS=${EMBEDDER_MAX_STEPS:--1}
EMBEDDER_LR=${EMBEDDER_LR:-1e-5}
EMBEDDER_SAVE_STEPS=${EMBEDDER_SAVE_STEPS:-0}
EMBEDDER_TORCH_DTYPE=${EMBEDDER_TORCH_DTYPE:-bfloat16}

# Candidate generation.
NUM_CANDIDATES=${NUM_CANDIDATES:-4}
COT_TEMPERATURES=${COT_TEMPERATURES:-0.6,0.8,1.0,1.1}
COT_MAX_WORKERS=${COT_MAX_WORKERS:-1}
COT_MAX_NEW_TOKENS=${COT_MAX_NEW_TOKENS:-2048}
COT_MAX_PROMPT_TOKENS=${COT_MAX_PROMPT_TOKENS:-4096}
COT_TOP_P=${COT_TOP_P:-0.9}
COT_GENERATION_API_PROVIDER=${COT_GENERATION_API_PROVIDER:-glm_codeplan}
COT_GENERATION_API_BASE_URL=${COT_GENERATION_API_BASE_URL:-https://open.bigmodel.cn/api/coding/paas/v4}
COT_GENERATION_API_MODEL=${COT_GENERATION_API_MODEL:-glm-5.2}
COT_GENERATION_API_KEY=${COT_GENERATION_API_KEY:-${BIGMODEL_API_KEY:-}}
COT_GENERATION_API_TIMEOUT=${COT_GENERATION_API_TIMEOUT:-300}
COT_GENERATION_API_MAX_RETRIES=${COT_GENERATION_API_MAX_RETRIES:-3}
COT_GENERATION_API_MIN_INTERVAL=${COT_GENERATION_API_MIN_INTERVAL:-0}
COT_GENERATION_API_THINKING=${COT_GENERATION_API_THINKING:-enabled}
COT_GENERATION_API_REASONING_EFFORT=${COT_GENERATION_API_REASONING_EFFORT:-medium}

# Rubric scoring for reject sampling. Defaults are direct JSON scoring without
# reasoning tokens.
RUBRIC_JUDGE_PROVIDER=${RUBRIC_JUDGE_PROVIDER:-openai_compatible}
RUBRIC_JUDGE_API_BASE_URL=${RUBRIC_JUDGE_API_BASE_URL:-https://open.bigmodel.cn/api/coding/paas/v4}
RUBRIC_JUDGE_API_MODEL=${RUBRIC_JUDGE_API_MODEL:-glm-5.2}
RUBRIC_JUDGE_API_KEY=${RUBRIC_JUDGE_API_KEY:-${BIGMODEL_API_KEY:-}}
RUBRIC_JUDGE_API_TIMEOUT=${RUBRIC_JUDGE_API_TIMEOUT:-180}
RUBRIC_JUDGE_API_MAX_RETRIES=${RUBRIC_JUDGE_API_MAX_RETRIES:-1}
RUBRIC_JUDGE_API_MAX_TOKENS=${RUBRIC_JUDGE_API_MAX_TOKENS:-128}
RUBRIC_JUDGE_API_THINKING=${RUBRIC_JUDGE_API_THINKING:-disabled}
RUBRIC_JUDGE_API_REASONING_EFFORT=${RUBRIC_JUDGE_API_REASONING_EFFORT:-}
RUBRIC_JUDGE_WORKERS=${RUBRIC_JUDGE_WORKERS:-1}
RUBRIC_JUDGE_MIN_INTERVAL=${RUBRIC_JUDGE_MIN_INTERVAL:-0}
RUBRIC_JUDGE_USE_TARGET=${RUBRIC_JUDGE_USE_TARGET:-1}
RUBRIC_JUDGE_SAVE_RAW=${RUBRIC_JUDGE_SAVE_RAW:-0}

# Gain/select.
GAIN_EMBEDDER_MODE=${GAIN_EMBEDDER_MODE:-qwen3_embedding}
GAIN_MODE=${GAIN_MODE:-ndcg}
GAIN_NDCG_K=${GAIN_NDCG_K:-100}
GRPO_BASELINE_EMBEDDER_MODE=${GRPO_BASELINE_EMBEDDER_MODE:-$GAIN_EMBEDDER_MODE}
MIN_RUBRIC=${MIN_RUBRIC:-0.5}
MIN_GAIN=${MIN_GAIN:-0.0}
TOP_K=${TOP_K:-1}
FALLBACK_WHEN_EMPTY=${FALLBACK_WHEN_EMPTY:-1}
GRPO_EXCLUDE_SFT=${GRPO_EXCLUDE_SFT:-1}

# SFT. Default to full-parameter training. Set SFT_TRAIN_TYPE=lora to use LoRA.
SFT_TRAIN_TYPE=${SFT_TRAIN_TYPE:-full}
SFT_LORA_RANK=${SFT_LORA_RANK:-64}
SFT_LORA_ALPHA=${SFT_LORA_ALPHA:-128}
SFT_MAX_STEPS=${SFT_MAX_STEPS:--1}
SFT_NUM_TRAIN_EPOCHS=${SFT_NUM_TRAIN_EPOCHS:-1}
SFT_BATCH_SIZE=${SFT_BATCH_SIZE:-1}
SFT_GRAD_ACCUM=${SFT_GRAD_ACCUM:-8}
SFT_MAX_LENGTH=${SFT_MAX_LENGTH:-4096}
SFT_LEARNING_RATE=${SFT_LEARNING_RATE:-1e-5}
SFT_SAVE_STEPS=${SFT_SAVE_STEPS:-200}
SFT_CUDA_VISIBLE_DEVICES=${SFT_CUDA_VISIBLE_DEVICES:-0}
SFT_NPROC_PER_NODE=${SFT_NPROC_PER_NODE:-1}
SFT_MASTER_PORT=${SFT_MASTER_PORT:-29500}

# GRPO. Default to full-parameter training. Set GRPO_TRAIN_TYPE=lora to use LoRA.
GRPO_MODEL=${GRPO_MODEL:-}
GRPO_ADAPTERS=${GRPO_ADAPTERS:-}
GRPO_TRAIN_TYPE=${GRPO_TRAIN_TYPE:-full}
GRPO_LORA_RANK=${GRPO_LORA_RANK:-32}
GRPO_LORA_ALPHA=${GRPO_LORA_ALPHA:-64}
GRPO_MAX_STEPS=${GRPO_MAX_STEPS:-20}
GRPO_NUM_GENERATIONS=${GRPO_NUM_GENERATIONS:-2}
GRPO_BATCH_SIZE=${GRPO_BATCH_SIZE:-1}
GRPO_GRAD_ACCUM=${GRPO_GRAD_ACCUM:-2}
GRPO_MAX_LENGTH=${GRPO_MAX_LENGTH:-4096}
GRPO_MAX_COMPLETION_LENGTH=${GRPO_MAX_COMPLETION_LENGTH:-2048}
GRPO_LEARNING_RATE=${GRPO_LEARNING_RATE:-1e-6}
GRPO_CUDA_VISIBLE_DEVICES=${GRPO_CUDA_VISIBLE_DEVICES:-0}
GRPO_NPROC_PER_NODE=${GRPO_NPROC_PER_NODE:-1}
GRPO_MASTER_PORT=${GRPO_MASTER_PORT:-29501}
FORMAT_WEIGHT=${FORMAT_WEIGHT:-0.2}
QUALITY_WEIGHT=${QUALITY_WEIGHT:-0.3}
GAIN_WEIGHT=${GAIN_WEIGHT:-1.0}
RUBRIC_REWARD_API_PROVIDER=${RUBRIC_REWARD_API_PROVIDER:-openai_compatible}
RUBRIC_REWARD_API_BASE_URL=${RUBRIC_REWARD_API_BASE_URL:-$RUBRIC_JUDGE_API_BASE_URL}
RUBRIC_REWARD_API_MODEL=${RUBRIC_REWARD_API_MODEL:-$RUBRIC_JUDGE_API_MODEL}
RUBRIC_REWARD_API_KEY=${RUBRIC_REWARD_API_KEY:-$RUBRIC_JUDGE_API_KEY}
RUBRIC_REWARD_API_TIMEOUT=${RUBRIC_REWARD_API_TIMEOUT:-180}
RUBRIC_REWARD_API_MAX_RETRIES=${RUBRIC_REWARD_API_MAX_RETRIES:-1}
RUBRIC_REWARD_API_MAX_TOKENS=${RUBRIC_REWARD_API_MAX_TOKENS:-128}
RUBRIC_REWARD_API_THINKING=${RUBRIC_REWARD_API_THINKING:-disabled}
RUBRIC_REWARD_API_FALLBACK=${RUBRIC_REWARD_API_FALLBACK:-none}

# Optional vLLM rollout server for GRPO on a second GPU.
RUN_VLLM_SERVER=${RUN_VLLM_SERVER:-0}
VLLM_CUDA_VISIBLE_DEVICES=${VLLM_CUDA_VISIBLE_DEVICES:-1}
VLLM_SERVER_HOST=${VLLM_SERVER_HOST:-127.0.0.1}
VLLM_SERVER_PORT=${VLLM_SERVER_PORT:-8001}
VLLM_SERVER_TIMEOUT=${VLLM_SERVER_TIMEOUT:-300}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.8}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-6144}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-$GRPO_NUM_GENERATIONS}
VLLM_TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE:-1}
VLLM_PIPELINE_PARALLEL_SIZE=${VLLM_PIPELINE_PARALLEL_SIZE:-1}
VLLM_ENABLE_LORA=${VLLM_ENABLE_LORA:-auto}
VLLM_MAX_LORA_RANK=${VLLM_MAX_LORA_RANK:-$GRPO_LORA_RANK}
VLLM_SERVED_MODEL_NAME=${VLLM_SERVED_MODEL_NAME:-qwen3-rrec-grpo}

# Evaluation smoke.
EVAL_SPLIT=${EVAL_SPLIT:-test}
EVAL_MAX_EXAMPLES=${EVAL_MAX_EXAMPLES:-20}
EVAL_MAX_NEW_TOKENS=${EVAL_MAX_NEW_TOKENS:-2048}
EVAL_SCORER=${EVAL_SCORER:-qwen3_embedding}

SMOKE=${SMOKE:-0}
if [[ "$SMOKE" == "1" ]]; then
  MAX_EXAMPLES=${SMOKE_MAX_EXAMPLES:-20}
  EMBEDDER_MAX_EXAMPLES_PER_CATEGORY=${SMOKE_EMBEDDER_MAX_EXAMPLES_PER_CATEGORY:-20}
  EMBEDDER_MAX_STEPS=1
  NUM_CANDIDATES=2
  COT_MAX_WORKERS=1
  COT_TEMPERATURES=${SMOKE_COT_TEMPERATURES:-0.6,0.8}
  SFT_MAX_STEPS=1
  SFT_SAVE_STEPS=1
  GRPO_MAX_STEPS=1
  GRPO_NUM_GENERATIONS=2
  EVAL_MAX_EXAMPLES=2
fi

log() {
  printf '\n[%s] %s\n' "$(date '+%F %T')" "$*"
}

stage_enabled() {
  local flag
  flag=$(printf '%s' "$1" | tr '[:upper:]' '[:lower:]')
  local output="${2:-}"
  case "$flag" in
    1|true|yes|on) return 0 ;;
    auto)
      if [[ -z "$output" ]]; then
        return 0
      fi
      [[ ! -e "$output" || ! -s "$output" ]]
      ;;
    *) return 1 ;;
  esac
}

require_file() {
  local path="$1"
  if [[ ! -s "$path" ]]; then
    echo "Required file is missing or empty: $path" >&2
    exit 1
  fi
}

require_local_path() {
  local label="$1"
  local path="$2"
  case "$path" in
    /*|./*|../*)
      if [[ ! -e "$path" ]]; then
        echo "Required $label path does not exist: $path" >&2
        exit 1
      fi
      ;;
  esac
}

latest_checkpoint() {
  local dir="$1"
  find "$dir" -type d -name 'checkpoint-*' -print 2>/dev/null | sort -V | tail -n 1
}

wait_for_vllm() {
  local url="http://$VLLM_SERVER_HOST:$VLLM_SERVER_PORT/get_world_size/"
  local deadline=$((SECONDS + VLLM_SERVER_TIMEOUT))
  while (( SECONDS < deadline )); do
    if curl -fsS --max-time 5 "$url" >/dev/null 2>&1; then
      log "vLLM server is ready at $url"
      return 0
    fi
    sleep 5
  done
  echo "Timed out waiting for vLLM server: $url" >&2
  return 1
}

if [[ ! -f "$VENV/bin/activate" ]]; then
  echo "Virtualenv is missing: $VENV" >&2
  exit 1
fi

mkdir -p "$DATA_DIR" "$OUT_DIR" "$LOG_DIR" "$CKPT_ROOT"
# shellcheck source=/dev/null
source "$VENV/bin/activate"
cd "$ROOT"

export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export MODELSCOPE_CACHE=${MODELSCOPE_CACHE:-/root/autodl-tmp/modelscope_cache}
export QWEN3_EMBEDDING_MODEL

log "Pipeline root: $ROOT"
log "Category: $CATEGORY run: $RUN_NAME"

if [[ "$TRAIN_ONLY" == "1" ]]; then
  log "TRAIN_ONLY=1: skipping data construction and reading prebuilt datasets"
  require_local_path "MODEL" "$MODEL"
  if stage_enabled "$RUN_EMBEDDER_TRAIN" "$(latest_checkpoint "$EMBEDDER_OUT")"; then
    require_file "$EMBEDDER_DATASET"
    require_local_path "BASE_EMBEDDING_MODEL" "$BASE_EMBEDDING_MODEL"
  else
    require_local_path "QWEN3_EMBEDDING_MODEL" "$QWEN3_EMBEDDING_MODEL"
  fi
  if stage_enabled "$RUN_SFT" "$(latest_checkpoint "$SFT_OUT")"; then
    require_file "$SFT_DATASET"
  fi
  if stage_enabled "$RUN_GRPO" "$(latest_checkpoint "$GRPO_OUT")"; then
    require_file "$GRPO_DATASET"
  fi
fi

if stage_enabled "$RUN_PREPARE_EXAMPLES" "$EXAMPLES_FILE"; then
  log "Preparing RRec examples -> $EXAMPLES_FILE"
  "$PYTHON_BIN" scripts/prepare_rrec_amazon_examples.py \
    --data-root "$RREC_DATA_ROOT" \
    --category "$CATEGORY" \
    --split "$SPLIT" \
    --output "$EXAMPLES_FILE" \
    --max-examples "$MAX_EXAMPLES" \
    --max-history-items "$MAX_HISTORY_ITEMS" \
    --min-history "$MIN_HISTORY" \
    --min-rating "$MIN_RATING" \
    --max-target-chars "$MAX_TARGET_CHARS" \
    --shuffle \
    --seed "$SEED"
else
  log "Skipping example preparation: $EXAMPLES_FILE"
fi

if stage_enabled "$RUN_EMBEDDER_DATA" "$EMBEDDER_DATASET"; then
  log "Building phase-0 embedder dataset -> $EMBEDDER_DATASET"
  read -r -a embedder_categories <<< "$EMBEDDER_CATEGORIES"
  "$PYTHON_BIN" scripts/make_phase0_embedder_dataset.py \
    --data-root "$RREC_DATA_ROOT" \
    --categories "${embedder_categories[@]}" \
    --split "$SPLIT" \
    --output "$EMBEDDER_DATASET" \
    --max-examples-per-category "$EMBEDDER_MAX_EXAMPLES_PER_CATEGORY" \
    --max-history-items "$MAX_HISTORY_ITEMS" \
    --min-history "$MIN_HISTORY" \
    --min-rating "$MIN_RATING" \
    --max-target-chars "$MAX_TARGET_CHARS" \
    --shuffle \
    --seed "$SEED"
else
  log "Skipping embedder dataset: $EMBEDDER_DATASET"
fi

if stage_enabled "$RUN_EMBEDDER_TRAIN" "$(latest_checkpoint "$EMBEDDER_OUT")"; then
  require_file "$EMBEDDER_DATASET"
  log "Training phase-0 embedder -> $EMBEDDER_OUT"
  CUDA_VISIBLE_DEVICES=${EMBEDDER_CUDA_VISIBLE_DEVICES:-0} \
  "$PYTHON_BIN" scripts/train_phase0_embedder.py \
    --model "$BASE_EMBEDDING_MODEL" \
    --dataset "$EMBEDDER_DATASET" \
    --output-dir "$EMBEDDER_OUT" \
    --max-rows "$EMBEDDER_MAX_ROWS" \
    --max-length "$EMBEDDER_MAX_LENGTH" \
    --batch-size "$EMBEDDER_BATCH_SIZE" \
    --grad-accum "$EMBEDDER_GRAD_ACCUM" \
    --epochs "$EMBEDDER_EPOCHS" \
    --max-steps "$EMBEDDER_MAX_STEPS" \
    --learning-rate "$EMBEDDER_LR" \
    --torch-dtype "$EMBEDDER_TORCH_DTYPE" \
    --save-steps "$EMBEDDER_SAVE_STEPS" \
    --seed "$SEED"
else
  log "Skipping embedder training: $EMBEDDER_OUT"
fi

trained_embedder=$(latest_checkpoint "$EMBEDDER_OUT")
if [[ -n "$trained_embedder" ]]; then
  QWEN3_EMBEDDING_MODEL="$trained_embedder"
  export QWEN3_EMBEDDING_MODEL
  log "Using trained embedder: $QWEN3_EMBEDDING_MODEL"
else
  log "Using embedding model: $QWEN3_EMBEDDING_MODEL"
fi

if stage_enabled "$RUN_COT_GENERATE" "$CANDIDATE_LISTS"; then
  require_file "$EXAMPLES_FILE"
  log "Generating CoT candidate lists -> $CANDIDATE_LISTS"
  "$PYTHON_BIN" scripts/generate_cot_candidate_lists.py \
    --input "$EXAMPLES_FILE" \
    --output "$CANDIDATE_LISTS" \
    --max-examples "$MAX_EXAMPLES" \
    --num-candidates "$NUM_CANDIDATES" \
    --temperatures "$COT_TEMPERATURES" \
    --max-workers "$COT_MAX_WORKERS" \
    --api-provider "$COT_GENERATION_API_PROVIDER" \
    --api-base-url "$COT_GENERATION_API_BASE_URL" \
    --api-key "$COT_GENERATION_API_KEY" \
    --api-model "$COT_GENERATION_API_MODEL" \
    --api-timeout "$COT_GENERATION_API_TIMEOUT" \
    --api-max-retries "$COT_GENERATION_API_MAX_RETRIES" \
    --api-min-interval "$COT_GENERATION_API_MIN_INTERVAL" \
    --api-thinking "$COT_GENERATION_API_THINKING" \
    --api-reasoning-effort "$COT_GENERATION_API_REASONING_EFFORT" \
    --max-new-tokens "$COT_MAX_NEW_TOKENS" \
    --max-prompt-tokens "$COT_MAX_PROMPT_TOKENS" \
    --top-p "$COT_TOP_P"
else
  log "Skipping CoT generation: $CANDIDATE_LISTS"
fi

if stage_enabled "$RUN_RUBRIC_SCORE" "$RUBRIC_SCORES"; then
  require_file "$CANDIDATE_LISTS"
  log "Scoring CoT candidates -> $RUBRIC_SCORES"
  judge_target_arg=(--use-target)
  if [[ "$RUBRIC_JUDGE_USE_TARGET" != "1" ]]; then
    judge_target_arg=(--no-use-target)
  fi
  judge_raw_arg=(--no-save-raw)
  if [[ "$RUBRIC_JUDGE_SAVE_RAW" == "1" ]]; then
    judge_raw_arg=(--save-raw)
  fi
  "$PYTHON_BIN" scripts/score_cot_candidate_lists.py \
    --input "$CANDIDATE_LISTS" \
    --output "$RUBRIC_SCORES" \
    --max-examples "$MAX_EXAMPLES" \
    --max-workers "$RUBRIC_JUDGE_WORKERS" \
    --judge-provider "$RUBRIC_JUDGE_PROVIDER" \
    --judge-base-url "$RUBRIC_JUDGE_API_BASE_URL" \
    --judge-api-key "$RUBRIC_JUDGE_API_KEY" \
    --judge-model "$RUBRIC_JUDGE_API_MODEL" \
    --judge-timeout "$RUBRIC_JUDGE_API_TIMEOUT" \
    --judge-max-retries "$RUBRIC_JUDGE_API_MAX_RETRIES" \
    --judge-max-tokens "$RUBRIC_JUDGE_API_MAX_TOKENS" \
    --judge-thinking "$RUBRIC_JUDGE_API_THINKING" \
    --judge-reasoning-effort "$RUBRIC_JUDGE_API_REASONING_EFFORT" \
    --min-interval "$RUBRIC_JUDGE_MIN_INTERVAL" \
    "${judge_target_arg[@]}" \
    "${judge_raw_arg[@]}"
else
  log "Skipping rubric scoring: $RUBRIC_SCORES"
fi

if stage_enabled "$RUN_MERGE" "$COT_JUDGED"; then
  require_file "$CANDIDATE_LISTS"
  require_file "$RUBRIC_SCORES"
  log "Merging candidates and rubric scores -> $COT_JUDGED"
  "$PYTHON_BIN" scripts/merge_candidate_list_rubric.py \
    --candidate-lists "$CANDIDATE_LISTS" \
    --rubric-scores "$RUBRIC_SCORES" \
    --output "$COT_JUDGED" \
    --scored-examples-output "$SCORED_EXAMPLES"
else
  log "Skipping merge: $COT_JUDGED"
fi

if stage_enabled "$RUN_GAIN" "$COT_SCORED"; then
  require_file "$COT_JUDGED"
  if [[ "$GAIN_MODE" == "ndcg" ]]; then
    require_file "$GAIN_ITEM_INFO"
  fi
  log "Computing CoT gain with $GAIN_EMBEDDER_MODE -> $COT_SCORED"
  "$PYTHON_BIN" scripts/compute_cot_gain.py \
    --input "$COT_JUDGED" \
    --output "$COT_SCORED" \
    --embedder-mode "$GAIN_EMBEDDER_MODE" \
    --gain-mode "$GAIN_MODE" \
    --item-info "$GAIN_ITEM_INFO" \
    --ndcg-k "$GAIN_NDCG_K" \
    --model "$MODEL" \
    --embedding-model "$QWEN3_EMBEDDING_MODEL" \
    --embedding-batch-size "${GAIN_EMBEDDING_BATCH_SIZE:-8}" \
    --embedding-max-length "${GAIN_EMBEDDING_MAX_LENGTH:-8192}"
else
  log "Skipping gain: $COT_SCORED"
fi

if stage_enabled "$RUN_SELECT" "$FILTERED_COT"; then
  require_file "$COT_SCORED"
  log "Selecting filtered CoT -> $FILTERED_COT"
  select_args=()
  if [[ "$FALLBACK_WHEN_EMPTY" == "1" ]]; then
    select_args+=(--fallback-when-empty)
  fi
  "$PYTHON_BIN" scripts/select_filtered_cot.py \
    --input "$COT_SCORED" \
    --output "$FILTERED_COT" \
    --rejected-output "$REJECTED_COT" \
    --top-k "$TOP_K" \
    --min-rubric "$MIN_RUBRIC" \
    --min-gain "$MIN_GAIN" \
    "${select_args[@]}"
else
  log "Skipping select: $FILTERED_COT"
fi

if stage_enabled "$RUN_DATASETS" "$SFT_DATASET" || stage_enabled "$RUN_DATASETS" "$GRPO_DATASET"; then
  require_file "$FILTERED_COT"
  require_file "$SCORED_EXAMPLES"
  log "Building SFT dataset -> $SFT_DATASET"
  "$PYTHON_BIN" scripts/make_sft_dataset.py \
    --input "$FILTERED_COT" \
    --output "$SFT_DATASET"

  log "Building GRPO dataset -> $GRPO_DATASET"
  grpo_exclude_args=()
  if [[ "$GRPO_EXCLUDE_SFT" == "1" ]]; then
    grpo_exclude_args+=(--exclude-prompts-from "$SFT_DATASET")
  fi
  "$PYTHON_BIN" scripts/make_grpo_dataset.py \
    --input "$SCORED_EXAMPLES" \
    --output "$GRPO_DATASET" \
    --baseline-mode "$GRPO_BASELINE_EMBEDDER_MODE" \
    --embedding-model "$QWEN3_EMBEDDING_MODEL" \
    --embedding-batch-size "${GRPO_BASELINE_EMBEDDING_BATCH_SIZE:-8}" \
    --embedding-max-length "${GRPO_BASELINE_EMBEDDING_MAX_LENGTH:-8192}" \
    "${grpo_exclude_args[@]}"
else
  log "Skipping dataset build: $SFT_DATASET / $GRPO_DATASET"
fi

if stage_enabled "$RUN_SFT" "$(latest_checkpoint "$SFT_OUT")"; then
  require_file "$SFT_DATASET"
  log "Running SFT -> $SFT_OUT"
  CUDA_VISIBLE_DEVICES="$SFT_CUDA_VISIBLE_DEVICES" \
  NPROC_PER_NODE="$SFT_NPROC_PER_NODE" \
  MASTER_PORT="$SFT_MASTER_PORT" \
  MODEL="$MODEL" \
  DATASET="$SFT_DATASET" \
  OUT="$SFT_OUT" \
  TRAIN_TYPE="$SFT_TRAIN_TYPE" \
  LORA_RANK="$SFT_LORA_RANK" \
  LORA_ALPHA="$SFT_LORA_ALPHA" \
  MAX_STEPS="$SFT_MAX_STEPS" \
  NUM_TRAIN_EPOCHS="$SFT_NUM_TRAIN_EPOCHS" \
  BATCH_SIZE="$SFT_BATCH_SIZE" \
  GRAD_ACCUM="$SFT_GRAD_ACCUM" \
  MAX_LENGTH="$SFT_MAX_LENGTH" \
  LEARNING_RATE="$SFT_LEARNING_RATE" \
  SAVE_STEPS="$SFT_SAVE_STEPS" \
  bash scripts/run_sft_qwen3_4b.sh
else
  log "Skipping SFT: $SFT_OUT"
fi

sft_checkpoint=${SFT_ADAPTER:-}
if [[ -z "$sft_checkpoint" ]]; then
  sft_checkpoint=$(latest_checkpoint "$SFT_OUT")
fi
if [[ -z "$sft_checkpoint" && "$RUN_GRPO" != "0" && -z "$GRPO_MODEL" ]]; then
  echo "No SFT checkpoint found. Set SFT_ADAPTER, set GRPO_MODEL, or run SFT first." >&2
  exit 1
fi

if [[ "$RUN_SFT_MERGE" != "0" && "$SFT_TRAIN_TYPE" != "lora" ]]; then
  log "Skipping SFT merge because SFT_TRAIN_TYPE=$SFT_TRAIN_TYPE is not lora"
elif stage_enabled "$RUN_SFT_MERGE" "$SFT_MERGED_MODEL"; then
  if [[ -z "$sft_checkpoint" ]]; then
    echo "Cannot merge SFT adapter because no checkpoint was found." >&2
    exit 1
  fi
  log "Merging SFT adapter -> $SFT_MERGED_MODEL"
  swift export \
    --model "$MODEL" \
    --adapters "$sft_checkpoint" \
    --merge_lora true \
    --output_dir "$SFT_MERGED_MODEL" \
    ${SWIFT_EXPORT_EXTRA_ARGS:-}
else
  log "Skipping SFT merge"
fi

if [[ -z "$GRPO_MODEL" ]]; then
  if [[ "$SFT_TRAIN_TYPE" != "lora" ]]; then
    GRPO_MODEL="$sft_checkpoint"
    GRPO_ADAPTERS=${GRPO_ADAPTERS:-}
  elif [[ "$RUN_SFT_MERGE" != "0" && -d "$SFT_MERGED_MODEL" ]]; then
    GRPO_MODEL="$SFT_MERGED_MODEL"
    GRPO_ADAPTERS=${GRPO_ADAPTERS:-}
  else
    GRPO_MODEL="$MODEL"
    GRPO_ADAPTERS=${GRPO_ADAPTERS:-$sft_checkpoint}
  fi
fi

if [[ "$VLLM_ENABLE_LORA" == "auto" ]]; then
  if [[ -n "$GRPO_ADAPTERS" || "$GRPO_TRAIN_TYPE" == "lora" ]]; then
    VLLM_ENABLE_LORA=true
  else
    VLLM_ENABLE_LORA=false
  fi
fi

if [[ "$RUN_VLLM_SERVER" == "1" ]]; then
  log "Starting vLLM rollout server on GPU(s) $VLLM_CUDA_VISIBLE_DEVICES"
  CUDA_VISIBLE_DEVICES="$VLLM_CUDA_VISIBLE_DEVICES" \
  nohup swift rollout \
    --model "$GRPO_MODEL" \
    --model_type "$MODEL_TYPE" \
    --infer_backend vllm \
    --host "$VLLM_SERVER_HOST" \
    --port "$VLLM_SERVER_PORT" \
    --served_model_name "$VLLM_SERVED_MODEL_NAME" \
    --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
    --vllm_max_model_len "$VLLM_MAX_MODEL_LEN" \
    --vllm_max_num_seqs "$VLLM_MAX_NUM_SEQS" \
    --vllm_tensor_parallel_size "$VLLM_TENSOR_PARALLEL_SIZE" \
    --vllm_pipeline_parallel_size "$VLLM_PIPELINE_PARALLEL_SIZE" \
    --vllm_enable_lora "$VLLM_ENABLE_LORA" \
    --vllm_max_lora_rank "$VLLM_MAX_LORA_RANK" \
    > "$LOG_DIR/vllm_${RUN_NAME}.log" 2>&1 &
  echo $! > "$LOG_DIR/vllm_${RUN_NAME}.pid"
  wait_for_vllm
fi

if stage_enabled "$RUN_GRPO" "$(latest_checkpoint "$GRPO_OUT")"; then
  require_file "$GRPO_DATASET"
  log "Running GRPO -> $GRPO_OUT"
  CUDA_VISIBLE_DEVICES="$GRPO_CUDA_VISIBLE_DEVICES" \
  NPROC_PER_NODE="$GRPO_NPROC_PER_NODE" \
  MASTER_PORT="$GRPO_MASTER_PORT" \
  MODEL="$GRPO_MODEL" \
  MODEL_TYPE="$MODEL_TYPE" \
  ADAPTERS="$GRPO_ADAPTERS" \
  DATASET="$GRPO_DATASET" \
  OUT="$GRPO_OUT" \
  TRAIN_TYPE="$GRPO_TRAIN_TYPE" \
  LORA_RANK="$GRPO_LORA_RANK" \
  LORA_ALPHA="$GRPO_LORA_ALPHA" \
  MAX_STEPS="$GRPO_MAX_STEPS" \
  NUM_GENERATIONS="$GRPO_NUM_GENERATIONS" \
  BATCH_SIZE="$GRPO_BATCH_SIZE" \
  GRAD_ACCUM="$GRPO_GRAD_ACCUM" \
  MAX_LENGTH="$GRPO_MAX_LENGTH" \
  MAX_COMPLETION_LENGTH="$GRPO_MAX_COMPLETION_LENGTH" \
  LEARNING_RATE="$GRPO_LEARNING_RATE" \
  FORMAT_WEIGHT="$FORMAT_WEIGHT" \
  QUALITY_WEIGHT="$QUALITY_WEIGHT" \
  GAIN_WEIGHT="$GAIN_WEIGHT" \
  RUBRIC_REWARD_SCORER=api \
  RUBRIC_REWARD_API_PROVIDER="$RUBRIC_REWARD_API_PROVIDER" \
  RUBRIC_REWARD_API_BASE_URL="$RUBRIC_REWARD_API_BASE_URL" \
  RUBRIC_REWARD_API_MODEL="$RUBRIC_REWARD_API_MODEL" \
  RUBRIC_REWARD_API_KEY="$RUBRIC_REWARD_API_KEY" \
  RUBRIC_REWARD_API_TIMEOUT="$RUBRIC_REWARD_API_TIMEOUT" \
  RUBRIC_REWARD_API_MAX_RETRIES="$RUBRIC_REWARD_API_MAX_RETRIES" \
  RUBRIC_REWARD_API_MAX_TOKENS="$RUBRIC_REWARD_API_MAX_TOKENS" \
  RUBRIC_REWARD_API_THINKING="$RUBRIC_REWARD_API_THINKING" \
  RUBRIC_REWARD_API_FALLBACK="$RUBRIC_REWARD_API_FALLBACK" \
  RUBRIC_GAIN_EMBEDDER_MODE="$GAIN_EMBEDDER_MODE" \
  QWEN3_EMBEDDING_MODEL="$QWEN3_EMBEDDING_MODEL" \
  USE_VLLM=${USE_VLLM:-$RUN_VLLM_SERVER} \
  VLLM_MODE=${VLLM_MODE:-server} \
  VLLM_SERVER_HOST="$VLLM_SERVER_HOST" \
  VLLM_SERVER_PORT="$VLLM_SERVER_PORT" \
  VLLM_SERVER_TIMEOUT="$VLLM_SERVER_TIMEOUT" \
  bash scripts/run_grpo_qwen3_4b.sh
else
  log "Skipping GRPO: $GRPO_OUT"
fi

grpo_checkpoint=$(latest_checkpoint "$GRPO_OUT")

if [[ "$RUN_EVAL" == "1" ]]; then
  eval_adapter=${EVAL_ADAPTER:-$grpo_checkpoint}
  if [[ -z "$eval_adapter" ]]; then
    echo "RUN_EVAL=1 but no GRPO checkpoint was found." >&2
    exit 1
  fi
  log "Running reasoner proxy eval for $eval_adapter"
  CUDA_VISIBLE_DEVICES=${EVAL_CUDA_VISIBLE_DEVICES:-0} \
  "$PYTHON_BIN" scripts/evaluate_reasoner_fullset_proxy.py \
    --category "$CATEGORY" \
    --split "$EVAL_SPLIT" \
    --model "$GRPO_MODEL" \
    --adapter "$eval_adapter" \
    --adapter-name "$RUN_NAME" \
    --max-examples "$EVAL_MAX_EXAMPLES" \
    --max-history-items "$MAX_HISTORY_ITEMS" \
    --max-prompt-tokens "$GRPO_MAX_LENGTH" \
    --max-new-tokens "$EVAL_MAX_NEW_TOKENS" \
    --temperature 0.0 \
    --scorer "$EVAL_SCORER" \
    --embedding-model "$QWEN3_EMBEDDING_MODEL" \
    --output "$OUT_DIR/eval_${RUN_NAME}.json" \
    --predictions-output "$OUT_DIR/pred_${RUN_NAME}.jsonl"
fi

MANIFEST=${MANIFEST:-$OUT_DIR/pipeline_manifest_${RUN_NAME}.txt}
cat > "$MANIFEST" <<EOF
run_name=$RUN_NAME
category=$CATEGORY
examples=$EXAMPLES_FILE
embedder_dataset=$EMBEDDER_DATASET
embedding_model=$QWEN3_EMBEDDING_MODEL
candidate_lists=$CANDIDATE_LISTS
rubric_scores=$RUBRIC_SCORES
cot_judged=$COT_JUDGED
cot_scored=$COT_SCORED
filtered_cot=$FILTERED_COT
rejected_cot=$REJECTED_COT
sft_dataset=$SFT_DATASET
grpo_dataset=$GRPO_DATASET
sft_out=$SFT_OUT
sft_checkpoint=$sft_checkpoint
sft_merged_model=$SFT_MERGED_MODEL
grpo_model=$GRPO_MODEL
grpo_adapters=$GRPO_ADAPTERS
grpo_out=$GRPO_OUT
grpo_checkpoint=$grpo_checkpoint
EOF

log "Pipeline artifacts"
cat "$MANIFEST"
