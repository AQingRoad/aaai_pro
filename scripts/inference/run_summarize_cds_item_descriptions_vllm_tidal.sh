#!/usr/bin/env bash
set -euo pipefail

# Summarize each CDs_and_Vinyl item description once with a local vLLM model.
# The output is a sidecar JSONL keyed by item_id and can be joined later with:
#   HISTORY_METADATA_MODE=summary ITEM_METADATA_SUMMARY=/path/to/item_metadata_summary_*.jsonl

ROOT=${ROOT:-/mnt/tidal-sh01/usr/xiayu6/xiayu/aaai_pro}
VENV=${VENV:-/root/miniconda3/envs/swift}
PYTHON_BIN=${PYTHON_BIN:-$VENV/bin/python}

CATEGORY=${CATEGORY:-CDs_and_Vinyl}
RREC_EVAL_DIR=${RREC_EVAL_DIR:-$ROOT/github_artifacts/$CATEGORY/rrec_eval}
ITEM_INFO=${ITEM_INFO:-$RREC_EVAL_DIR/item_info.jsonl}
OUT_DIR=${OUT_DIR:-$ROOT/github_artifacts/$CATEGORY/rrec_eval}
RUN_NAME=${RUN_NAME:-qwen3_32b_desc60}
OUTPUT=${OUTPUT:-$OUT_DIR/item_metadata_summary_${RUN_NAME}.jsonl}

MODEL=${MODEL:-/mnt/tidal-sh01/usr/xiayu6/xiayu/checkpoint/Qwen3/32B}
TOKENIZER=${TOKENIZER:-}
DEVICES=${DEVICES:-0,1,2,3,4,5,6,7}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-auto}

MAX_EXAMPLES=${MAX_EXAMPLES:-0}
NUM_SHARDS=${NUM_SHARDS:-1}
SHARD_INDEX=${SHARD_INDEX:-0}
GENERATION_BATCH_SIZE=${GENERATION_BATCH_SIZE:-64}
SUMMARY_MAX_WORDS=${SUMMARY_MAX_WORDS:-60}
MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS:-2048}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-128}
TEMPERATURE=${TEMPERATURE:-0.0}
TOP_P=${TOP_P:-0.9}

VLLM_DTYPE=${VLLM_DTYPE:-bfloat16}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-4096}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-$GENERATION_BATCH_SIZE}
VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-0}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.85}
VLLM_ENFORCE_EAGER=${VLLM_ENFORCE_EAGER:-0}
SAVE_EVERY=${SAVE_EVERY:-200}
SEED=${SEED:-42}

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

require_path "project root" "$ROOT"
require_path "python" "$PYTHON_BIN"
require_path "item_info JSONL" "$ITEM_INFO"
require_path "vLLM model" "$MODEL"

if [[ "$TENSOR_PARALLEL_SIZE" == "auto" ]]; then
  TENSOR_PARALLEL_SIZE=$(count_devices "$DEVICES")
fi

cd "$ROOT"
mkdir -p "$(dirname "$OUTPUT")"
export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

echo "ROOT=$ROOT"
echo "ITEM_INFO=$ITEM_INFO"
echo "OUTPUT=$OUTPUT"
echo "MODEL=$MODEL"
echo "DEVICES=$DEVICES"
echo "TENSOR_PARALLEL_SIZE=$TENSOR_PARALLEL_SIZE"
echo "GENERATION_BATCH_SIZE=$GENERATION_BATCH_SIZE"
echo "SUMMARY_MAX_WORDS=$SUMMARY_MAX_WORDS"
echo "MAX_EXAMPLES=$MAX_EXAMPLES"

tokenizer_args=()
if [[ -n "$TOKENIZER" ]]; then
  tokenizer_args+=(--tokenizer "$TOKENIZER")
fi

eager_args=()
if [[ "$VLLM_ENFORCE_EAGER" == "1" || "$VLLM_ENFORCE_EAGER" == "true" ]]; then
  eager_args+=(--enforce-eager)
fi

CUDA_VISIBLE_DEVICES="$DEVICES" \
"$PYTHON_BIN" scripts/inference/vllm_batch_infer_jsonl.py \
  --task description_summary \
  --input "$ITEM_INFO" \
  --output "$OUTPUT" \
  --model "$MODEL" \
  "${tokenizer_args[@]}" \
  --max-examples "$MAX_EXAMPLES" \
  --num-shards "$NUM_SHARDS" \
  --shard-index "$SHARD_INDEX" \
  --generation-batch-size "$GENERATION_BATCH_SIZE" \
  --summary-max-words "$SUMMARY_MAX_WORDS" \
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
  --save-every "$SAVE_EVERY" \
  --seed "$SEED"
