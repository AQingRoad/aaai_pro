#!/usr/bin/env bash
set -euo pipefail

# Evaluate every checkpoint under CHECKPOINT_ROOT with vLLM generation and
# full-candidate RRec ranking. Existing per-checkpoint eval JSON files are
# skipped, so this script can be rerun after interruptions.

ROOT=${ROOT:-/mnt/tidal-sh01/usr/xiayu6/xiayu/aaai_pro}
VENV=${VENV:-/root/miniconda3/envs/swift}
PYTHON_BIN=${PYTHON_BIN:-$VENV/bin/python}

CATEGORY=${CATEGORY:-CDs_and_Vinyl}
SPLIT=${SPLIT:-test}
RREC_EVAL_DIR=${RREC_EVAL_DIR:-$ROOT/github_artifacts/$CATEGORY/rrec_eval}
EVAL_EXAMPLES=${EVAL_EXAMPLES:-$RREC_EVAL_DIR/${SPLIT}.jsonl}
ITEM_INFO=${ITEM_INFO:-$RREC_EVAL_DIR/item_info.jsonl}

CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-}
CHECKPOINT_LIMIT=${CHECKPOINT_LIMIT:-0}
CHECKPOINT_PATTERN=${CHECKPOINT_PATTERN:-checkpoint-*}
MODEL_KIND=${MODEL_KIND:-reasoner}
TOKENIZER=${TOKENIZER:-}

QWEN3_EMBEDDING_MODEL=${QWEN3_EMBEDDING_MODEL:-$ROOT/checkpoints/rrec_amazon_CDs_and_Vinyl/qwen3_embedding_cds_tidal/checkpoint-249}
SCORER=${SCORER:-qwen3_embedding}
EMBEDDING_BATCH_SIZE=${EMBEDDING_BATCH_SIZE:-128}
EMBEDDING_MAX_LENGTH=${EMBEDDING_MAX_LENGTH:-2048}
EMBEDDING_TORCH_DTYPE=${EMBEDDING_TORCH_DTYPE:-bfloat16}
EMBEDDING_DEVICE=${EMBEDDING_DEVICE:-cuda:0}

DEVICES=${DEVICES:-0,1,2,3,4,5,6,7}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-auto}
MAX_EXAMPLES=${MAX_EXAMPLES:-0}
MAX_HISTORY_ITEMS=${MAX_HISTORY_ITEMS:-20}
HISTORY_METADATA_MODE=${HISTORY_METADATA_MODE:-none}
HISTORY_MAX_ITEM_CHARS=${HISTORY_MAX_ITEM_CHARS:-320}
ITEM_METADATA_SUMMARY=${ITEM_METADATA_SUMMARY:-}
MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS:-2048}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
GENERATION_BATCH_SIZE=${GENERATION_BATCH_SIZE:-64}
TEMPERATURE=${TEMPERATURE:-0.0}
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
KS=${KS:-5,10,20}

EXPERIMENT_NAME=${EXPERIMENT_NAME:-}
RESULT_ROOT=${RESULT_ROOT:-}
SKIP_EXISTING=${SKIP_EXISTING:-1}

require_path() {
  local label="$1"
  local path="$2"
  if [[ ! -e "$path" ]]; then
    echo "Missing $label: $path" >&2
    exit 1
  fi
}

safe_name() {
  local text="$1"
  text="${text#"$ROOT"/}"
  text="${text//\//__}"
  text="${text// /_}"
  text="${text//:/_}"
  echo "$text"
}

embedding_tag() {
  local path="$1"
  local parent
  parent="$(basename "$(dirname "$path")")"
  echo "${parent}_$(basename "$path")"
}

find_checkpoints() {
  local root="$1"
  if [[ "$(basename "$root")" == checkpoint-* ]]; then
    printf '%s\n' "$root"
  else
    find "$root" -type d -name "$CHECKPOINT_PATTERN" -print | sort -V
  fi
}

require_path "project root" "$ROOT"
require_path "python" "$PYTHON_BIN"
if [[ -z "$CHECKPOINT_ROOT" ]]; then
  echo "Set CHECKPOINT_ROOT to an SFT/GRPO output dir or a checkpoint-* dir." >&2
  exit 1
fi
require_path "checkpoint root" "$CHECKPOINT_ROOT"
require_path "eval examples JSONL" "$EVAL_EXAMPLES"
require_path "item info JSONL" "$ITEM_INFO"
require_path "embedding checkpoint" "$QWEN3_EMBEDDING_MODEL"

IFS=',' read -r -a DEVICE_LIST <<< "$DEVICES"
if [[ "${#DEVICE_LIST[@]}" -eq 0 || -z "${DEVICE_LIST[0]}" ]]; then
  echo "DEVICES is empty" >&2
  exit 1
fi
if [[ "$TENSOR_PARALLEL_SIZE" == "auto" ]]; then
  TENSOR_PARALLEL_SIZE=${#DEVICE_LIST[@]}
fi

if [[ -z "$EXPERIMENT_NAME" ]]; then
  ckpt_tag="$(safe_name "$CHECKPOINT_ROOT")"
  emb_tag="$(safe_name "$(embedding_tag "$QWEN3_EMBEDDING_MODEL")")"
  EXPERIMENT_NAME="${CATEGORY}_${MODEL_KIND}_${ckpt_tag}_${SPLIT}_${SCORER}_${emb_tag}_vllm_tp${TENSOR_PARALLEL_SIZE}"
fi
if [[ -z "$RESULT_ROOT" ]]; then
  RESULT_ROOT="$ROOT/outputs/rrec_amazon/$CATEGORY/eval_vllm/$EXPERIMENT_NAME"
fi

cd "$ROOT"
mkdir -p "$RESULT_ROOT/logs" "$RESULT_ROOT/predictions"
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

mapfile -t CHECKPOINTS < <(find_checkpoints "$CHECKPOINT_ROOT")
if [[ "${#CHECKPOINTS[@]}" -eq 0 ]]; then
  echo "No checkpoints found under $CHECKPOINT_ROOT with pattern $CHECKPOINT_PATTERN" >&2
  exit 1
fi
if (( CHECKPOINT_LIMIT > 0 && CHECKPOINT_LIMIT < ${#CHECKPOINTS[@]} )); then
  CHECKPOINTS=("${CHECKPOINTS[@]:0:$CHECKPOINT_LIMIT}")
fi

manifest="$RESULT_ROOT/manifest.txt"
cat > "$manifest" <<EOF
experiment_name=$EXPERIMENT_NAME
model_kind=$MODEL_KIND
checkpoint_root=$CHECKPOINT_ROOT
category=$CATEGORY
split=$SPLIT
examples=$EVAL_EXAMPLES
item_info=$ITEM_INFO
embedding_model=$QWEN3_EMBEDDING_MODEL
scorer=$SCORER
devices=$DEVICES
tensor_parallel_size=$TENSOR_PARALLEL_SIZE
max_examples=$MAX_EXAMPLES
history_metadata_mode=$HISTORY_METADATA_MODE
history_max_item_chars=$HISTORY_MAX_ITEM_CHARS
item_metadata_summary=$ITEM_METADATA_SUMMARY
max_new_tokens=$MAX_NEW_TOKENS
generation_batch_size=$GENERATION_BATCH_SIZE
vllm_disable_custom_all_reduce=$VLLM_DISABLE_CUSTOM_ALL_REDUCE
vllm_distributed_executor_backend=$VLLM_DISTRIBUTED_EXECUTOR_BACKEND
nccl_net=$NCCL_NET
nccl_ib_disable=$NCCL_IB_DISABLE
nccl_p2p_disable=$NCCL_P2P_DISABLE
ks=$KS
EOF

echo "RESULT_ROOT=$RESULT_ROOT"
echo "Found ${#CHECKPOINTS[@]} checkpoint(s)"
echo "VLLM_DISABLE_CUSTOM_ALL_REDUCE=$VLLM_DISABLE_CUSTOM_ALL_REDUCE"
echo "VLLM_DISTRIBUTED_EXECUTOR_BACKEND=$VLLM_DISTRIBUTED_EXECUTOR_BACKEND"
echo "NCCL_NET=$NCCL_NET NCCL_IB_DISABLE=$NCCL_IB_DISABLE NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE"

for checkpoint in "${CHECKPOINTS[@]}"; do
  require_path "checkpoint" "$checkpoint"
  rel="$(safe_name "$checkpoint")"
  eval_out="$RESULT_ROOT/${rel}.eval.json"
  pred_out="$RESULT_ROOT/predictions/${rel}.pred.jsonl"
  log_out="$RESULT_ROOT/logs/${rel}.log"

  if [[ "$SKIP_EXISTING" == "1" || "$SKIP_EXISTING" == "true" ]]; then
    if [[ -s "$eval_out" ]]; then
      echo "Skipping existing result: $eval_out"
      continue
    fi
  fi

  echo "Evaluating checkpoint: $checkpoint"
  echo "  eval: $eval_out"
  echo "  pred: $pred_out"
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

  CUDA_VISIBLE_DEVICES="$DEVICES" \
  QWEN3_EMBEDDING_DEVICE="$EMBEDDING_DEVICE" \
  "$PYTHON_BIN" scripts/eval/evaluate_reasoner_vllm_fullset.py \
    --examples "$EVAL_EXAMPLES" \
    --item-info "$ITEM_INFO" \
    --category "$CATEGORY" \
    --split "$SPLIT" \
    --model "$checkpoint" \
    "${tokenizer_args[@]}" \
    --run-name "$rel" \
    --max-examples "$MAX_EXAMPLES" \
    --max-history-items "$MAX_HISTORY_ITEMS" \
    --history-metadata-mode "$HISTORY_METADATA_MODE" \
    --history-max-item-chars "$HISTORY_MAX_ITEM_CHARS" \
    --item-summary "$ITEM_METADATA_SUMMARY" \
    --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --generation-batch-size "$GENERATION_BATCH_SIZE" \
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
    --scorer "$SCORER" \
    --embedding-model "$QWEN3_EMBEDDING_MODEL" \
    --embedding-max-length "$EMBEDDING_MAX_LENGTH" \
    --embedding-batch-size "$EMBEDDING_BATCH_SIZE" \
    --embedding-torch-dtype "$EMBEDDING_TORCH_DTYPE" \
    --embedding-device "$EMBEDDING_DEVICE" \
    --ks "$KS" \
    --output "$eval_out" \
    --predictions-output "$pred_out" \
    > "$log_out" 2>&1
done

echo "Finished. Results are under: $RESULT_ROOT"
