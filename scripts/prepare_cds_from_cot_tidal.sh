#!/usr/bin/env bash
set -euo pipefail

# Rebuild CDs_and_Vinyl training data from existing CoT candidate lists and
# rubric scores on the Tidal server. This script does not generate or rejudge CoT.

ROOT=${ROOT:-/mnt/tidal-sh01/usr/xiayu6/xiayu/aaai_pro}
VENV=${VENV:-/root/miniconda3/envs/swift}
PYTHON_BIN=${PYTHON_BIN:-$VENV/bin/python}

CATEGORY=${CATEGORY:-CDs_and_Vinyl}
RUN_NAME=${RUN_NAME:-deepseek_v4_pro_cds_remerged}
MODEL=${MODEL:-/mnt/tidal-sh01/usr/xiayu6/xiayu/checkpoint/Qwen3/4B}

COT_ARTIFACT_DIR=${COT_ARTIFACT_DIR:-$ROOT/github_artifacts/CDs_and_Vinyl/cot}
CANDIDATE_LISTS=${CANDIDATE_LISTS:-$COT_ARTIFACT_DIR/cot_candidate_lists_deepseek_v4_pro_low.jsonl}
RUBRIC_SCORES=${RUBRIC_SCORES:-$COT_ARTIFACT_DIR/cot_candidate_lists_deepseek_v4_pro_low.rubric_deepseek_v4_pro.jsonl}

OUT_DIR=${OUT_DIR:-$ROOT/outputs/rrec_amazon/$CATEGORY}
COT_JUDGED=${COT_JUDGED:-$OUT_DIR/cot_judged_${RUN_NAME}.jsonl}
COT_SCORED=${COT_SCORED:-$OUT_DIR/cot_scored_${RUN_NAME}.jsonl}
SCORED_EXAMPLES=${SCORED_EXAMPLES:-$OUT_DIR/examples_${RUN_NAME}.jsonl}
FILTERED_COT=${FILTERED_COT:-$OUT_DIR/filtered_high_quality_cot_${RUN_NAME}.jsonl}
REJECTED_COT=${REJECTED_COT:-$OUT_DIR/rejected_cot_${RUN_NAME}.jsonl}
SFT_DATASET=${SFT_DATASET:-$OUT_DIR/sft_${RUN_NAME}.jsonl}
GRPO_DATASET=${GRPO_DATASET:-$OUT_DIR/grpo_${RUN_NAME}.jsonl}

EMBEDDER_OUT=${EMBEDDER_OUT:-$ROOT/checkpoints/rrec_amazon_CDs_and_Vinyl/qwen3_embedding_cds_tidal}
QWEN3_EMBEDDING_MODEL=${QWEN3_EMBEDDING_MODEL:-}
GAIN_CUDA_VISIBLE_DEVICES=${GAIN_CUDA_VISIBLE_DEVICES:-0}
GAIN_EMBEDDER_MODE=${GAIN_EMBEDDER_MODE:-qwen3_embedding}
GRPO_BASELINE_EMBEDDER_MODE=${GRPO_BASELINE_EMBEDDER_MODE:-$GAIN_EMBEDDER_MODE}
GAIN_EMBEDDING_BATCH_SIZE=${GAIN_EMBEDDING_BATCH_SIZE:-8}
GAIN_EMBEDDING_MAX_LENGTH=${GAIN_EMBEDDING_MAX_LENGTH:-8192}
GAIN_EMBEDDING_DEVICE=${GAIN_EMBEDDING_DEVICE:-cuda:0}
GRPO_BASELINE_EMBEDDING_BATCH_SIZE=${GRPO_BASELINE_EMBEDDING_BATCH_SIZE:-8}
GRPO_BASELINE_EMBEDDING_MAX_LENGTH=${GRPO_BASELINE_EMBEDDING_MAX_LENGTH:-8192}
GRPO_BASELINE_EMBEDDING_DEVICE=${GRPO_BASELINE_EMBEDDING_DEVICE:-cuda:0}

MIN_RUBRIC=${MIN_RUBRIC:-0.5}
MIN_GAIN=${MIN_GAIN:-0.0}
TOP_K=${TOP_K:-1}
FALLBACK_WHEN_EMPTY=${FALLBACK_WHEN_EMPTY:-1}
GRPO_EXCLUDE_SFT=${GRPO_EXCLUDE_SFT:-1}

RUN_MERGE=${RUN_MERGE:-1}
RUN_GAIN=${RUN_GAIN:-1}
RUN_SELECT=${RUN_SELECT:-1}
RUN_DATASETS=${RUN_DATASETS:-1}

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

latest_checkpoint() {
  local dir="$1"
  find "$dir" -type d -name 'checkpoint-*' -print 2>/dev/null | sort -V | tail -n 1
}

require_path "project root" "$ROOT"
require_path "python" "$PYTHON_BIN"
require_file "candidate lists" "$CANDIDATE_LISTS"
require_file "rubric scores" "$RUBRIC_SCORES"

if [[ "$RUN_GAIN" == "1" || "$RUN_DATASETS" == "1" ]]; then
  if [[ -z "$QWEN3_EMBEDDING_MODEL" ]]; then
    QWEN3_EMBEDDING_MODEL=$(latest_checkpoint "$EMBEDDER_OUT")
  fi
  require_path "Qwen3 embedding checkpoint" "$QWEN3_EMBEDDING_MODEL"
fi

mkdir -p "$OUT_DIR"
cd "$ROOT"

export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export MODELSCOPE_CACHE=${MODELSCOPE_CACHE:-/mnt/tidal-sh01/usr/xiayu6/xiayu/modelscope_cache}
export QWEN3_EMBEDDING_MODEL

echo "ROOT=$ROOT"
echo "CANDIDATE_LISTS=$CANDIDATE_LISTS"
echo "RUBRIC_SCORES=$RUBRIC_SCORES"
echo "QWEN3_EMBEDDING_MODEL=$QWEN3_EMBEDDING_MODEL"
echo "RUN_NAME=$RUN_NAME"
echo "OUT_DIR=$OUT_DIR"
echo "GAIN_EMBEDDING_DEVICE=$GAIN_EMBEDDING_DEVICE"

if [[ "$RUN_MERGE" == "1" ]]; then
  "$PYTHON_BIN" scripts/merge_candidate_list_rubric.py \
    --candidate-lists "$CANDIDATE_LISTS" \
    --rubric-scores "$RUBRIC_SCORES" \
    --output "$COT_JUDGED" \
    --scored-examples-output "$SCORED_EXAMPLES"
else
  echo "Skipping merge: $COT_JUDGED"
fi

if [[ "$RUN_GAIN" == "1" ]]; then
  require_file "merged CoT" "$COT_JUDGED"
  CUDA_VISIBLE_DEVICES="$GAIN_CUDA_VISIBLE_DEVICES" \
  "$PYTHON_BIN" scripts/compute_cot_gain.py \
    --input "$COT_JUDGED" \
    --output "$COT_SCORED" \
    --embedder-mode "$GAIN_EMBEDDER_MODE" \
    --model "$MODEL" \
    --embedding-model "$QWEN3_EMBEDDING_MODEL" \
    --embedding-batch-size "$GAIN_EMBEDDING_BATCH_SIZE" \
    --embedding-max-length "$GAIN_EMBEDDING_MAX_LENGTH" \
    --device "$GAIN_EMBEDDING_DEVICE"
else
  echo "Skipping gain: $COT_SCORED"
fi

if [[ "$RUN_SELECT" == "1" ]]; then
  require_file "gain-scored CoT" "$COT_SCORED"
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
  echo "Skipping select: $FILTERED_COT"
fi

if [[ "$RUN_DATASETS" == "1" ]]; then
  require_file "filtered CoT" "$FILTERED_COT"
  require_file "scored examples" "$SCORED_EXAMPLES"
  "$PYTHON_BIN" scripts/make_sft_dataset.py \
    --input "$FILTERED_COT" \
    --output "$SFT_DATASET"

  grpo_exclude_args=()
  if [[ "$GRPO_EXCLUDE_SFT" == "1" ]]; then
    grpo_exclude_args+=(--exclude-prompts-from "$SFT_DATASET")
  fi
  "$PYTHON_BIN" scripts/make_grpo_dataset.py \
    --input "$SCORED_EXAMPLES" \
    --output "$GRPO_DATASET" \
    --baseline-mode "$GRPO_BASELINE_EMBEDDER_MODE" \
    --embedding-model "$QWEN3_EMBEDDING_MODEL" \
    --embedding-batch-size "$GRPO_BASELINE_EMBEDDING_BATCH_SIZE" \
    --embedding-max-length "$GRPO_BASELINE_EMBEDDING_MAX_LENGTH" \
    --device "$GRPO_BASELINE_EMBEDDING_DEVICE" \
    "${grpo_exclude_args[@]}"
else
  echo "Skipping dataset build: $SFT_DATASET / $GRPO_DATASET"
fi

echo "Rebuilt CoT artifacts:"
echo "  judged:   $COT_JUDGED"
echo "  scored:   $COT_SCORED"
echo "  filtered: $FILTERED_COT"
echo "  rejected: $REJECTED_COT"
echo "  examples: $SCORED_EXAMPLES"
echo "  sft:      $SFT_DATASET"
echo "  grpo:     $GRPO_DATASET"
