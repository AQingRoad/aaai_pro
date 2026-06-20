#!/usr/bin/env bash
set -euo pipefail

# Multi-GPU, sharded reasoner evaluation for RRec Amazon CDs_and_Vinyl.
# Each GPU runs one shard and uses batched generation inside the shard.

ROOT=${ROOT:-/mnt/tidal-sh01/usr/xiayu6/xiayu/aaai_pro}
VENV=${VENV:-/root/miniconda3/envs/swift}
PYTHON_BIN=${PYTHON_BIN:-$VENV/bin/python}

CATEGORY=${CATEGORY:-CDs_and_Vinyl}
SPLIT=${SPLIT:-test}
DATA_ROOT=${DATA_ROOT:-$ROOT/data}
OUT_DIR=${OUT_DIR:-$ROOT/outputs/rrec_amazon/$CATEGORY}
RUN_NAME=${RUN_NAME:-grpo_reasoner_multigpu}

MODEL=${MODEL:-}
ADAPTER=${ADAPTER:-}
GRPO_OUT=${GRPO_OUT:-$ROOT/checkpoints/rrec_amazon_CDs_and_Vinyl/qwen3_4b_grpo_cds_from_sft81_ndcg}

QWEN3_EMBEDDING_MODEL=${QWEN3_EMBEDDING_MODEL:-$ROOT/checkpoints/rrec_amazon_CDs_and_Vinyl/qwen3_embedding_cds_tidal/checkpoint-249}
SCORER=${SCORER:-qwen3_embedding}
EMBEDDING_BATCH_SIZE=${EMBEDDING_BATCH_SIZE:-128}
EMBEDDING_MAX_LENGTH=${EMBEDDING_MAX_LENGTH:-2048}
EMBEDDING_TORCH_DTYPE=${EMBEDDING_TORCH_DTYPE:-bfloat16}

DEVICES=${DEVICES:-0,1,2,3,4,5,6,7}
NUM_SHARDS=${NUM_SHARDS:-}
MAX_EXAMPLES=${MAX_EXAMPLES:-0}
MAX_HISTORY_ITEMS=${MAX_HISTORY_ITEMS:-20}
MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS:-2048}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
GENERATION_BATCH_SIZE=${GENERATION_BATCH_SIZE:-4}
TEMPERATURE=${TEMPERATURE:-0.0}
TOP_P=${TOP_P:-0.9}
TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
KS=${KS:-5,10,20,100}

SHARD_DIR=${SHARD_DIR:-$OUT_DIR/${RUN_NAME}_${SPLIT}_shards}
EVAL_OUT=${EVAL_OUT:-$OUT_DIR/eval_${RUN_NAME}_${SPLIT}.json}
PRED_OUT=${PRED_OUT:-$OUT_DIR/pred_${RUN_NAME}_${SPLIT}.jsonl}

latest_checkpoint() {
  local dir="$1"
  find "$dir" -type d -name 'checkpoint-*' -print 2>/dev/null | sort -V | tail -n 1
}

require_path() {
  local label="$1"
  local path="$2"
  if [[ ! -e "$path" ]]; then
    echo "Missing $label: $path" >&2
    exit 1
  fi
}

if [[ -z "$MODEL" ]]; then
  require_path "GRPO output dir" "$GRPO_OUT"
  MODEL=$(latest_checkpoint "$GRPO_OUT")
fi

require_path "project root" "$ROOT"
require_path "python" "$PYTHON_BIN"
require_path "reasoner model/checkpoint" "$MODEL"
require_path "Qwen3 embedding model" "$QWEN3_EMBEDDING_MODEL"

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
  echo "NUM_SHARDS=$NUM_SHARDS is larger than number of DEVICES=${#DEVICE_LIST[@]}. Use at most one shard per GPU." >&2
  exit 1
fi

cd "$ROOT"
mkdir -p "$SHARD_DIR" "$(dirname "$EVAL_OUT")" "$(dirname "$PRED_OUT")"

export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

echo "ROOT=$ROOT"
echo "MODEL=$MODEL"
echo "ADAPTER=${ADAPTER:-none}"
echo "QWEN3_EMBEDDING_MODEL=$QWEN3_EMBEDDING_MODEL"
echo "DATA_ROOT=$DATA_ROOT"
echo "SPLIT=$SPLIT"
echo "DEVICES=$DEVICES"
echo "NUM_SHARDS=$NUM_SHARDS"
echo "GENERATION_BATCH_SIZE=$GENERATION_BATCH_SIZE"
echo "MAX_EXAMPLES=$MAX_EXAMPLES"
echo "MAX_NEW_TOKENS=$MAX_NEW_TOKENS"
echo "EVAL_OUT=$EVAL_OUT"
echo "PRED_OUT=$PRED_OUT"

pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  device="${DEVICE_LIST[$shard]}"
  shard_output="$SHARD_DIR/eval_shard${shard}.json"
  shard_predictions="$SHARD_DIR/pred_shard${shard}.jsonl"
  shard_log="$SHARD_DIR/shard${shard}.log"
  echo "Launching shard $shard/$NUM_SHARDS on GPU $device -> $shard_log"
  (
    CUDA_VISIBLE_DEVICES="$device" \
    QWEN3_EMBEDDING_DEVICE=cuda:0 \
    "$PYTHON_BIN" scripts/evaluate_reasoner_fullset_proxy.py \
      --data-root "$DATA_ROOT" \
      --category "$CATEGORY" \
      --split "$SPLIT" \
      --model "$MODEL" \
      --adapter "$ADAPTER" \
      --adapter-name "$RUN_NAME" \
      --max-examples "$MAX_EXAMPLES" \
      --max-history-items "$MAX_HISTORY_ITEMS" \
      --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --generation-batch-size "$GENERATION_BATCH_SIZE" \
      --temperature "$TEMPERATURE" \
      --top-p "$TOP_P" \
      --torch-dtype "$TORCH_DTYPE" \
      --scorer "$SCORER" \
      --embedding-model "$QWEN3_EMBEDDING_MODEL" \
      --embedding-max-length "$EMBEDDING_MAX_LENGTH" \
      --embedding-batch-size "$EMBEDDING_BATCH_SIZE" \
      --embedding-torch-dtype "$EMBEDDING_TORCH_DTYPE" \
      --embedding-device cuda:0 \
      --ks "$KS" \
      --num-shards "$NUM_SHARDS" \
      --shard-index "$shard" \
      --output "$shard_output" \
      --predictions-output "$shard_predictions"
  ) >"$shard_log" 2>&1 &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

if (( status != 0 )); then
  echo "At least one shard failed. Check logs under $SHARD_DIR" >&2
  exit "$status"
fi

pred_files=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  pred_files+=("$SHARD_DIR/pred_shard${shard}.jsonl")
done

"$PYTHON_BIN" scripts/aggregate_reasoner_eval_shards.py \
  --predictions "${pred_files[@]}" \
  --ks "$KS" \
  --category "$CATEGORY" \
  --split "$SPLIT" \
  --adapter "$ADAPTER" \
  --adapter-name "$RUN_NAME" \
  --scorer "$SCORER" \
  --embedding-model "$QWEN3_EMBEDDING_MODEL" \
  --output "$EVAL_OUT" \
  --combined-predictions-output "$PRED_OUT"

echo "Wrote metrics: $EVAL_OUT"
echo "Wrote predictions: $PRED_OUT"
echo "Shard logs: $SHARD_DIR"
