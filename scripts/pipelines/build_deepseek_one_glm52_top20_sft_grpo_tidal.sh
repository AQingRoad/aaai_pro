#!/usr/bin/env bash
set -euo pipefail

# Build CDs_and_Vinyl SFT/GRPO data from the DeepSeek one-CoT candidate list
# and GLM-5.2 rubric scores. The slow NDCG gain step runs in parallel over GPUs.

ROOT=${ROOT:-/mnt/tidal-sh01/usr/xiayu6/xiayu/aaai_pro}
VENV=${VENV:-/root/miniconda3/envs/swift}
PYTHON_BIN=${PYTHON_BIN:-$VENV/bin/python}

CATEGORY=${CATEGORY:-CDs_and_Vinyl}
OUT=${OUT:-$ROOT/outputs/rrec_amazon/$CATEGORY}

COT=${COT:-$ROOT/github_artifacts/CDs_and_Vinyl/cot/cot_candidate_one_lists_deepseek_v4_pro_low.jsonl}
RUBRIC=${RUBRIC:-$ROOT/github_artifacts/CDs_and_Vinyl/cot/cot_candidate_one_lists_deepseek_v4_pro_low.rubric_glm_5_2.jsonl}
ITEM_INFO=${ITEM_INFO:-$ROOT/github_artifacts/CDs_and_Vinyl/rrec_eval/item_info.jsonl}
EMBEDDING_MODEL=${EMBEDDING_MODEL:-$ROOT/checkpoints/rrec_amazon_CDs_and_Vinyl/qwen3_embedding_cds_cot_deepseek_v4_pro_low_one_cot_global_bsz128_xgpu/checkpoint-167}

RUN_TAG=${RUN_TAG:-deepseek_one_glm52_ckpt167}
NDCG_K=${NDCG_K:-100}
TOP_PERCENT=${TOP_PERCENT:-0.2}
MIN_RUBRIC=${MIN_RUBRIC:-0}
MIN_GAIN=${MIN_GAIN:-0}

DEVICES=${DEVICES:-0,1,2,3,4,5,6,7}
NUM_SHARDS=${NUM_SHARDS:-auto}
GAIN_EMBEDDING_BATCH_SIZE=${GAIN_EMBEDDING_BATCH_SIZE:-128}
GAIN_ROW_BATCH_SIZE=${GAIN_ROW_BATCH_SIZE:-128}
GAIN_TORCH_DTYPE=${GAIN_TORCH_DTYPE:-bfloat16}
GAIN_DEVICE=${GAIN_DEVICE:-cuda:0}
CLEAN_SHARDS=${CLEAN_SHARDS:-1}

JUDGED=${JUDGED:-$OUT/cot_judged_${RUN_TAG}.jsonl}
GAIN_OUT=${GAIN_OUT:-$OUT/cot_scored_${RUN_TAG}_ndcg${NDCG_K}.jsonl}
SHARD_DIR=${SHARD_DIR:-$OUT/cot_scored_${RUN_TAG}_ndcg${NDCG_K}_shards}
FILTERED=${FILTERED:-$OUT/filtered_top20_${RUN_TAG}.jsonl}
REJECTED=${REJECTED:-$OUT/rejected_top20_${RUN_TAG}.jsonl}
SUMMARY=${SUMMARY:-$OUT/filtered_top20_${RUN_TAG}.summary.json}
SFT=${SFT:-$OUT/sft_top20_${RUN_TAG}.jsonl}
GRPO=${GRPO:-$OUT/grpo_${RUN_TAG}_exclude_sft_top20.jsonl}

require_file() {
  local label="$1"
  local path="$2"
  if [[ ! -s "$path" ]]; then
    echo "Missing or empty $label: $path" >&2
    exit 1
  fi
}

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
require_file "DeepSeek one-CoT candidate list" "$COT"
require_file "GLM-5.2 rubric scores" "$RUBRIC"
require_file "item info" "$ITEM_INFO"
require_path "embedding checkpoint" "$EMBEDDING_MODEL"

IFS=',' read -r -a DEVICE_LIST <<< "$DEVICES"
if [[ "${#DEVICE_LIST[@]}" -eq 0 || -z "${DEVICE_LIST[0]}" ]]; then
  echo "DEVICES is empty" >&2
  exit 1
fi
if [[ "$NUM_SHARDS" == "auto" ]]; then
  NUM_SHARDS=${#DEVICE_LIST[@]}
fi
if ((NUM_SHARDS < 1)); then
  echo "NUM_SHARDS must be >= 1" >&2
  exit 1
fi
if ((NUM_SHARDS > ${#DEVICE_LIST[@]})); then
  echo "NUM_SHARDS=$NUM_SHARDS is larger than device count=${#DEVICE_LIST[@]}" >&2
  exit 1
fi

mkdir -p "$OUT" "$SHARD_DIR"
cd "$ROOT"

export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export MODELSCOPE_CACHE=${MODELSCOPE_CACHE:-/mnt/tidal-sh01/usr/xiayu6/xiayu/modelscope_cache}

echo "ROOT=$ROOT"
echo "COT=$COT"
echo "RUBRIC=$RUBRIC"
echo "ITEM_INFO=$ITEM_INFO"
echo "EMBEDDING_MODEL=$EMBEDDING_MODEL"
echo "RUN_TAG=$RUN_TAG"
echo "OUT=$OUT"
echo "DEVICES=$DEVICES"
echo "NUM_SHARDS=$NUM_SHARDS"
echo "NDCG_K=$NDCG_K"
echo "TOP_PERCENT=$TOP_PERCENT"
echo "MIN_RUBRIC=$MIN_RUBRIC"
echo "MIN_GAIN=$MIN_GAIN"

"$PYTHON_BIN" scripts/cot/merge_candidate_list_rubric.py \
  --candidate-lists "$COT" \
  --rubric-scores "$RUBRIC" \
  --output "$JUDGED"

if [[ "$CLEAN_SHARDS" == "1" || "$CLEAN_SHARDS" == "true" ]]; then
  rm -f "$SHARD_DIR"/shard_*.jsonl
  rm -f "$SHARD_DIR"/shard_*.log
  rm -f "$SHARD_DIR"/baseline_cache_shard_*.jsonl
fi
rm -f "$GAIN_OUT"

pids=()
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  gpu="${DEVICE_LIST[$shard]}"
  shard_output="$SHARD_DIR/shard_${shard}.jsonl"
  shard_log="$SHARD_DIR/shard_${shard}.log"
  echo "Launching gain shard $shard/$NUM_SHARDS on GPU $gpu -> $shard_log"
  CUDA_VISIBLE_DEVICES="$gpu" "$PYTHON_BIN" scripts/selection/compute_cot_gain.py \
    --input "$JUDGED" \
    --output "$shard_output" \
    --embedder-mode qwen3_embedding \
    --gain-mode ndcg \
    --item-info "$ITEM_INFO" \
    --ndcg-k "$NDCG_K" \
    --embedding-model "$EMBEDDING_MODEL" \
    --device "$GAIN_DEVICE" \
    --embedding-batch-size "$GAIN_EMBEDDING_BATCH_SIZE" \
    --row-batch-size "$GAIN_ROW_BATCH_SIZE" \
    --torch-dtype "$GAIN_TORCH_DTYPE" \
    --num-shards "$NUM_SHARDS" \
    --shard-index "$shard" \
    --baseline-cache "$SHARD_DIR/baseline_cache_shard_${shard}.jsonl" \
    >"$shard_log" 2>&1 &
  pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failed=1
  fi
done
if [[ "$failed" != "0" ]]; then
  echo "At least one gain shard failed. Check logs under: $SHARD_DIR" >&2
  exit 1
fi

: >"$GAIN_OUT"
for shard in $(seq 0 $((NUM_SHARDS - 1))); do
  shard_output="$SHARD_DIR/shard_${shard}.jsonl"
  require_file "gain shard output $shard" "$shard_output"
  cat "$shard_output" >>"$GAIN_OUT"
done

judged_rows=$(wc -l < "$JUDGED" | tr -d ' ')
gain_rows=$(wc -l < "$GAIN_OUT" | tr -d ' ')
if [[ "$judged_rows" != "$gain_rows" ]]; then
  echo "Row count mismatch: judged=$judged_rows gain=$gain_rows" >&2
  exit 1
fi

"$PYTHON_BIN" scripts/selection/select_top_percent_cot.py \
  --input "$GAIN_OUT" \
  --output "$FILTERED" \
  --rejected-output "$REJECTED" \
  --summary-output "$SUMMARY" \
  --top-percent "$TOP_PERCENT" \
  --score-field selection_score \
  --min-rubric "$MIN_RUBRIC" \
  --min-gain "$MIN_GAIN"

"$PYTHON_BIN" scripts/datasets/make_sft_dataset.py \
  --input "$FILTERED" \
  --output "$SFT"

"$PYTHON_BIN" scripts/datasets/make_grpo_dataset.py \
  --input "$GAIN_OUT" \
  --output "$GRPO" \
  --exclude-prompts-from "$SFT"

wc -l "$JUDGED" "$GAIN_OUT" "$FILTERED" "$SFT" "$GRPO"

echo "JUDGED=$JUDGED"
echo "GAIN_OUT=$GAIN_OUT"
echo "FILTERED=$FILTERED"
echo "SFT=$SFT"
echo "GRPO=$GRPO"
echo "SUMMARY=$SUMMARY"
