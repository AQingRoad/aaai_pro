#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/autodl-tmp/rec/aaai_pro}
CONFIG_ENV_FILE=${CONFIG_ENV_FILE:-$ROOT/configs/glm_codeplan.env}
VENV=${VENV:-/root/autodl-tmp/rec/ms-swift-312-cu124-venv}
PYTHON_BIN=${PYTHON_BIN:-}
MODEL=${MODEL:-/root/autodl-tmp/modelscope_cache/models/Qwen/Qwen3-4B}
MODEL_TYPE=${MODEL_TYPE:-qwen3}
TEMPLATE=${TEMPLATE:-qwen3}
QWEN3_EMBEDDING_MODEL=${QWEN3_EMBEDDING_MODEL:-/root/autodl-tmp/modelscope_cache/models/Qwen/Qwen3-Embedding-0.6B}
ADAPTERS=${ADAPTERS:-}
DATASET=${DATASET:-$ROOT/outputs/ml1m/grpo.jsonl}
OUT=${OUT:-$ROOT/checkpoints/qwen3_4b_grpo_rubric_gated}
MAX_STEPS=${MAX_STEPS:-20}
NUM_GENERATIONS=${NUM_GENERATIONS:-4}
GENERATION_BATCH_SIZE=${GENERATION_BATCH_SIZE:-}
TRAIN_TYPE=${TRAIN_TYPE:-lora}
LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-128}
BATCH_SIZE=${BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-4}
MAX_LENGTH=${MAX_LENGTH:-2048}
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-384}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
SAVE_STEPS=${SAVE_STEPS:-$MAX_STEPS}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-2}
FORMAT_WEIGHT=${FORMAT_WEIGHT:-0.2}
QUALITY_WEIGHT=${QUALITY_WEIGHT:-0.3}
GAIN_WEIGHT=${GAIN_WEIGHT:-1.0}
RUBRIC_REWARD_SCORER=${RUBRIC_REWARD_SCORER:-api}
RUBRIC_REWARD_API_PROVIDER=${RUBRIC_REWARD_API_PROVIDER:-${RUBRIC_JUDGE_API_PROVIDER:-zhipu}}
RUBRIC_REWARD_API_BASE_URL=${RUBRIC_REWARD_API_BASE_URL:-${RUBRIC_JUDGE_API_BASE_URL:-https://open.bigmodel.cn/api/coding/paas/v4}}
RUBRIC_REWARD_API_MODEL=${RUBRIC_REWARD_API_MODEL:-${RUBRIC_JUDGE_API_MODEL:-glm-5.2}}

load_bigmodel_api_key() {
  if [[ -n "${BIGMODEL_API_KEY:-}" || ! -f "$CONFIG_ENV_FILE" ]]; then
    return
  fi

  local line value
  line="$(grep -E '^[[:space:]]*(export[[:space:]]+)?BIGMODEL_API_KEY=' "$CONFIG_ENV_FILE" | tail -n 1 || true)"
  if [[ -z "$line" ]]; then
    return
  fi
  value="${line#*=}"
  value="${value%$'\r'}"
  value="${value%\"}"
  value="${value#\"}"
  value="${value%\'}"
  value="${value#\'}"
  if [[ -n "$value" ]]; then
    export BIGMODEL_API_KEY="$value"
  fi
}

load_bigmodel_api_key
RUBRIC_REWARD_API_KEY=${RUBRIC_REWARD_API_KEY:-${RUBRIC_JUDGE_API_KEY:-${BIGMODEL_API_KEY:-}}}
RUBRIC_REWARD_API_TIMEOUT=${RUBRIC_REWARD_API_TIMEOUT:-${RUBRIC_JUDGE_API_TIMEOUT:-60}}
RUBRIC_REWARD_API_MAX_RETRIES=${RUBRIC_REWARD_API_MAX_RETRIES:-${RUBRIC_JUDGE_API_MAX_RETRIES:-2}}
RUBRIC_REWARD_API_MAX_TOKENS=${RUBRIC_REWARD_API_MAX_TOKENS:-${RUBRIC_JUDGE_API_MAX_TOKENS:-128}}
RUBRIC_REWARD_API_THINKING=${RUBRIC_REWARD_API_THINKING:-${RUBRIC_JUDGE_API_THINKING:-disabled}}
RUBRIC_REWARD_API_FALLBACK=${RUBRIC_REWARD_API_FALLBACK:-rules}
RUBRIC_REWARD_SOURCE_LOG=${RUBRIC_REWARD_SOURCE_LOG:-$OUT/rubric_reward_sources.jsonl}
RUBRIC_REWARD_SOURCE_LOG_EVERY=${RUBRIC_REWARD_SOURCE_LOG_EVERY:-20}
RUBRIC_GAIN_MODE=${RUBRIC_GAIN_MODE:-ndcg}
RUBRIC_NDCG_ITEM_INFO=${RUBRIC_NDCG_ITEM_INFO:-$ROOT/github_artifacts/CDs_and_Vinyl/rrec_eval/item_info.jsonl}
RUBRIC_NDCG_K=${RUBRIC_NDCG_K:-100}
RUBRIC_NDCG_ITEM_MAX_CHARS=${RUBRIC_NDCG_ITEM_MAX_CHARS:-1200}
RUBRIC_NDCG_MASK_HISTORY_ITEMS=${RUBRIC_NDCG_MASK_HISTORY_ITEMS:-1}
RUBRIC_NDCG_MASK_PAD_ITEM=${RUBRIC_NDCG_MASK_PAD_ITEM:-1}
USE_VLLM=${USE_VLLM:-0}
VLLM_MODE=${VLLM_MODE:-colocate}
VLLM_TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE:-1}
VLLM_PIPELINE_PARALLEL_SIZE=${VLLM_PIPELINE_PARALLEL_SIZE:-1}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.45}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-4096}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-$NUM_GENERATIONS}
VLLM_ENABLE_LORA=${VLLM_ENABLE_LORA:-true}
VLLM_MAX_LORA_RANK=${VLLM_MAX_LORA_RANK:-$LORA_RANK}
VLLM_SERVER_BASE_URL=${VLLM_SERVER_BASE_URL:-}
VLLM_SERVER_HOST=${VLLM_SERVER_HOST:-}
VLLM_SERVER_PORT=${VLLM_SERVER_PORT:-8000}
VLLM_SERVER_TIMEOUT=${VLLM_SERVER_TIMEOUT:-240}
SWIFT_MODEL_TYPE_FLAG=${SWIFT_MODEL_TYPE_FLAG:-}
NPROC_PER_NODE=${NPROC_PER_NODE:-auto}
MASTER_PORT=${MASTER_PORT:-29501}

activate_swift_env() {
  if command -v swift >/dev/null 2>&1; then
    return
  fi

  if [[ -n "${VENV:-}" && -x "$VENV/bin/swift" ]]; then
    export PATH="$VENV/bin:$PATH"
    return
  fi

  echo "Cannot find swift. Activate the swift env first or set VENV to an env that contains bin/swift." >&2
  exit 1
}

activate_swift_env
cd "$ROOT"

if [[ "$RUBRIC_GAIN_MODE" == "ndcg" && ! -s "$RUBRIC_NDCG_ITEM_INFO" ]]; then
  echo "Missing RUBRIC_NDCG_ITEM_INFO for online NDCG reward: $RUBRIC_NDCG_ITEM_INFO" >&2
  exit 1
fi

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MODELSCOPE_CACHE=${MODELSCOPE_CACHE:-/root/autodl-tmp/modelscope_cache}
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export RUBRIC_GATE_THRESHOLD=${RUBRIC_GATE_THRESHOLD:-0.45}
export RUBRIC_GATE_PENALTY=${RUBRIC_GATE_PENALTY:--0.05}
export RUBRIC_REWARD_SCORER="$RUBRIC_REWARD_SCORER"
export RUBRIC_REWARD_API_PROVIDER="$RUBRIC_REWARD_API_PROVIDER"
export RUBRIC_REWARD_API_BASE_URL="$RUBRIC_REWARD_API_BASE_URL"
export RUBRIC_REWARD_API_MODEL="$RUBRIC_REWARD_API_MODEL"
if [[ -n "$RUBRIC_REWARD_API_KEY" ]]; then
  export RUBRIC_REWARD_API_KEY="$RUBRIC_REWARD_API_KEY"
fi
export RUBRIC_REWARD_API_TIMEOUT="$RUBRIC_REWARD_API_TIMEOUT"
export RUBRIC_REWARD_API_MAX_RETRIES="$RUBRIC_REWARD_API_MAX_RETRIES"
export RUBRIC_REWARD_API_MAX_TOKENS="$RUBRIC_REWARD_API_MAX_TOKENS"
export RUBRIC_REWARD_API_THINKING="$RUBRIC_REWARD_API_THINKING"
export RUBRIC_REWARD_API_FALLBACK="$RUBRIC_REWARD_API_FALLBACK"
export RUBRIC_REWARD_SOURCE_LOG="$RUBRIC_REWARD_SOURCE_LOG"
export RUBRIC_REWARD_SOURCE_LOG_EVERY="$RUBRIC_REWARD_SOURCE_LOG_EVERY"
export RUBRIC_GAIN_MODE="$RUBRIC_GAIN_MODE"
export RUBRIC_GAIN_EMBEDDER_MODE=${RUBRIC_GAIN_EMBEDDER_MODE:-qwen3_embedding}
export RUBRIC_NDCG_ITEM_INFO="$RUBRIC_NDCG_ITEM_INFO"
export RUBRIC_NDCG_K="$RUBRIC_NDCG_K"
export RUBRIC_NDCG_ITEM_MAX_CHARS="$RUBRIC_NDCG_ITEM_MAX_CHARS"
export RUBRIC_NDCG_MASK_HISTORY_ITEMS="$RUBRIC_NDCG_MASK_HISTORY_ITEMS"
export RUBRIC_NDCG_MASK_PAD_ITEM="$RUBRIC_NDCG_MASK_PAD_ITEM"
export QWEN3_EMBEDDING_MODEL="$QWEN3_EMBEDDING_MODEL"
export QWEN3_EMBEDDING_BATCH_SIZE=${QWEN3_EMBEDDING_BATCH_SIZE:-4}
export QWEN3_EMBEDDING_MAX_LENGTH=${QWEN3_EMBEDDING_MAX_LENGTH:-4096}

resolve_python_bin() {
  if [[ -n "$PYTHON_BIN" ]]; then
    echo "$PYTHON_BIN"
  elif [[ -n "${VENV:-}" && -x "$VENV/bin/python" ]]; then
    echo "$VENV/bin/python"
  else
    command -v python3
  fi
}

resolve_nproc() {
  if [[ "$NPROC_PER_NODE" == "auto" ]]; then
    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      echo 1
      return
    fi
    awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES"
  else
    echo "$NPROC_PER_NODE"
  fi
}

PYTHON_BIN="$(resolve_python_bin)"
SWIFT_BIN="$(command -v swift)"
NPROC="$(resolve_nproc)"
if ((NPROC < 1)); then
  echo "NPROC_PER_NODE must be >= 1" >&2
  exit 1
fi

TRAIN_ARGS=(--train_type "$TRAIN_TYPE")
if [[ "$TRAIN_TYPE" == "lora" ]]; then
  TRAIN_ARGS+=(--lora_rank "$LORA_RANK" --lora_alpha "$LORA_ALPHA")
fi

ADAPTER_ARGS=()
if [[ -n "$ADAPTERS" ]]; then
  ADAPTER_ARGS+=(--adapters "$ADAPTERS")
fi

MODEL_ARGS=(--model "$MODEL")
if [[ -n "$MODEL_TYPE" ]]; then
  resolve_model_type_flag() {
    if [[ -n "$SWIFT_MODEL_TYPE_FLAG" ]]; then
      echo "$SWIFT_MODEL_TYPE_FLAG"
      return
    fi

    local help_text
    help_text="$(swift rlhf --help 2>&1 || true)"
    if grep -q -- "--model-type" <<<"$help_text" && ! grep -q -- "--model_type" <<<"$help_text"; then
      echo "--model-type"
    else
      echo "--model_type"
    fi
  }
  MODEL_TYPE_FLAG="$(resolve_model_type_flag)"
  MODEL_ARGS+=("$MODEL_TYPE_FLAG" "$MODEL_TYPE")
fi
if [[ -n "$TEMPLATE" ]]; then
  MODEL_ARGS+=(--template "$TEMPLATE")
fi

echo "GRPO config:"
echo "  MODEL=$MODEL"
echo "  MODEL_TYPE=$MODEL_TYPE"
echo "  TEMPLATE=$TEMPLATE"
echo "  DATASET=$DATASET"
echo "  OUT=$OUT"
echo "  TRAIN_TYPE=$TRAIN_TYPE"
echo "  MAX_STEPS=$MAX_STEPS"
echo "  SAVE_STEPS=$SAVE_STEPS"
echo "  SAVE_TOTAL_LIMIT=$SAVE_TOTAL_LIMIT"
echo "  NUM_GENERATIONS=$NUM_GENERATIONS"
echo "  GENERATION_BATCH_SIZE=${GENERATION_BATCH_SIZE:-auto}"
echo "  CONFIG_ENV_FILE=$CONFIG_ENV_FILE"
echo "  RUBRIC_REWARD_API_MODEL=$RUBRIC_REWARD_API_MODEL"
echo "  RUBRIC_REWARD_SOURCE_LOG=$RUBRIC_REWARD_SOURCE_LOG"
echo "  RUBRIC_REWARD_SOURCE_LOG_EVERY=$RUBRIC_REWARD_SOURCE_LOG_EVERY"
echo "  RUBRIC_GAIN_MODE=$RUBRIC_GAIN_MODE"
echo "  RUBRIC_NDCG_ITEM_INFO=$RUBRIC_NDCG_ITEM_INFO"
echo "  RUBRIC_NDCG_K=$RUBRIC_NDCG_K"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  NPROC=$NPROC"
echo "  MASTER_PORT=$MASTER_PORT"

VLLM_ARGS=()
if [[ "$USE_VLLM" == "1" || "$USE_VLLM" == "true" ]]; then
  VLLM_ARGS+=(
    --use_vllm true
    --vllm_mode "$VLLM_MODE"
  )
  if [[ "$VLLM_MODE" == "server" ]]; then
    if [[ -n "$VLLM_SERVER_BASE_URL" ]]; then
      VLLM_ARGS+=(--vllm_server_base_url "$VLLM_SERVER_BASE_URL")
    else
      VLLM_ARGS+=(--vllm_server_host "$VLLM_SERVER_HOST" --vllm_server_port "$VLLM_SERVER_PORT")
    fi
    VLLM_ARGS+=(--vllm_server_timeout "$VLLM_SERVER_TIMEOUT")
  else
    VLLM_ARGS+=(
      --vllm_tensor_parallel_size "$VLLM_TENSOR_PARALLEL_SIZE"
      --vllm_pipeline_parallel_size "$VLLM_PIPELINE_PARALLEL_SIZE"
      --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION"
      --vllm_max_model_len "$VLLM_MAX_MODEL_LEN"
      --vllm_max_num_seqs "$VLLM_MAX_NUM_SEQS"
    )
  fi
  if [[ "$TRAIN_TYPE" == "lora" && "$VLLM_MODE" != "server" ]]; then
    VLLM_ARGS+=(
      --vllm_enable_lora "$VLLM_ENABLE_LORA"
      --vllm_max_lora_rank "$VLLM_MAX_LORA_RANK"
    )
  fi
fi

GENERATION_ARGS=(--num_generations "$NUM_GENERATIONS")
if [[ -n "$GENERATION_BATCH_SIZE" ]]; then
  GENERATION_ARGS+=(--generation_batch_size "$GENERATION_BATCH_SIZE")
fi

GRPO_ARGS=(
  rlhf
  --rlhf_type grpo \
  "${MODEL_ARGS[@]}" \
  "${ADAPTER_ARGS[@]}" \
  "${VLLM_ARGS[@]}" \
  --dataset "$DATASET" \
  --external_plugins "$ROOT/scripts/rubric_gated_reward.py" \
  --reward_funcs rubric_format rubric_quality rubric_gated_gain \
  --reward_weights "$FORMAT_WEIGHT" "$QUALITY_WEIGHT" "$GAIN_WEIGHT" \
  "${GENERATION_ARGS[@]}" \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --max_steps "$MAX_STEPS" \
  --max_length "$MAX_LENGTH" \
  --max_completion_length "$MAX_COMPLETION_LENGTH" \
  --learning_rate "$LEARNING_RATE" \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.05 \
  "${TRAIN_ARGS[@]}" \
  --torch_dtype bfloat16 \
  --gradient_checkpointing true \
  --save_only_model true \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit "$SAVE_TOTAL_LIMIT" \
  --logging_steps 1 \
  --log_completions true \
  --report_to none \
  --output_dir "$OUT"
)

if ((NPROC > 1)); then
  "$PYTHON_BIN" -m torch.distributed.run \
    --nproc_per_node "$NPROC" \
    --master_port "$MASTER_PORT" \
    "$SWIFT_BIN" "${GRPO_ARGS[@]}"
else
  "$SWIFT_BIN" "${GRPO_ARGS[@]}"
fi
