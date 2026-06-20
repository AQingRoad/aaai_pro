#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/autodl-tmp/rec/aaai_pro}
VENV=${VENV:-/root/autodl-tmp/rec/ms-swift-312-cu124-venv}
MODEL=${MODEL:-/root/autodl-tmp/modelscope_cache/models/Qwen/Qwen3-4B}
QWEN3_EMBEDDING_MODEL=${QWEN3_EMBEDDING_MODEL:-/root/autodl-tmp/modelscope_cache/models/Qwen/Qwen3-Embedding-0.6B}
CATEGORY=${CATEGORY:-CDs_and_Vinyl}
RUN_NAME=${RUN_NAME:-deepseek_v4_pro_partial}

OUT_DIR=${OUT_DIR:-$ROOT/outputs/rrec_amazon/$CATEGORY}
CANDIDATE_LISTS=${CANDIDATE_LISTS:-$OUT_DIR/cot_candidate_lists_deepseek_v4_pro_low.jsonl}
RUBRIC_SCORES=${RUBRIC_SCORES:-$OUT_DIR/cot_candidate_lists_deepseek_v4_pro_low.rubric_deepseek_v4_pro.jsonl}

COT_JUDGED=${COT_JUDGED:-$OUT_DIR/cot_judged_${RUN_NAME}.jsonl}
COT_SCORED=${COT_SCORED:-$OUT_DIR/cot_scored_${RUN_NAME}.jsonl}
SCORED_EXAMPLES=${SCORED_EXAMPLES:-$OUT_DIR/examples_${RUN_NAME}.jsonl}
FILTERED_COT=${FILTERED_COT:-$OUT_DIR/filtered_high_quality_cot_${RUN_NAME}.jsonl}
REJECTED_COT=${REJECTED_COT:-$OUT_DIR/rejected_cot_${RUN_NAME}.jsonl}
SFT_DATASET=${SFT_DATASET:-$OUT_DIR/sft_${RUN_NAME}.jsonl}
GRPO_DATASET=${GRPO_DATASET:-$OUT_DIR/grpo_${RUN_NAME}.jsonl}
GAIN_ITEM_INFO=${GAIN_ITEM_INFO:-$ROOT/github_artifacts/$CATEGORY/rrec_eval/item_info.jsonl}

GAIN_EMBEDDER_MODE=${GAIN_EMBEDDER_MODE:-qwen3_embedding}
GAIN_MODE=${GAIN_MODE:-ndcg}
GAIN_NDCG_K=${GAIN_NDCG_K:-100}
GRPO_BASELINE_EMBEDDER_MODE=${GRPO_BASELINE_EMBEDDER_MODE:-$GAIN_EMBEDDER_MODE}
MIN_RUBRIC=${MIN_RUBRIC:-0.5}
MIN_GAIN=${MIN_GAIN:-0.0}
TOP_K=${TOP_K:-1}
FALLBACK_WHEN_EMPTY=${FALLBACK_WHEN_EMPTY:-1}

RUN_PREPARE=${RUN_PREPARE:-1}
RUN_SFT=${RUN_SFT:-0}
RUN_GRPO=${RUN_GRPO:-0}

SFT_OUT=${SFT_OUT:-$ROOT/checkpoints/rrec_amazon_${CATEGORY}_qwen3_4b_sft_${RUN_NAME}}
GRPO_OUT=${GRPO_OUT:-$ROOT/checkpoints/rrec_amazon_${CATEGORY}_qwen3_4b_grpo_${RUN_NAME}}
ADAPTERS=${ADAPTERS:-}
SFT_MAX_STEPS=${SFT_MAX_STEPS:-${MAX_STEPS:--1}}
SFT_NUM_TRAIN_EPOCHS=${SFT_NUM_TRAIN_EPOCHS:-${NUM_TRAIN_EPOCHS:-1}}
SFT_SAVE_STEPS=${SFT_SAVE_STEPS:-${SAVE_STEPS:-200}}
GRPO_MAX_STEPS=${GRPO_MAX_STEPS:-${MAX_STEPS:-20}}
GRPO_NUM_GENERATIONS=${GRPO_NUM_GENERATIONS:-${NUM_GENERATIONS:-4}}

if [[ -f "$VENV/bin/activate" ]]; then
  source "$VENV/bin/activate"
elif [[ "$RUN_SFT" == "1" || "$RUN_GRPO" == "1" ]]; then
  echo "Training requires a valid virtualenv: $VENV" >&2
  exit 1
else
  echo "Virtualenv not found at $VENV; continuing with the current Python for data preparation only." >&2
fi
cd "$ROOT"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MODELSCOPE_CACHE=${MODELSCOPE_CACHE:-/root/autodl-tmp/modelscope_cache}
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export QWEN3_EMBEDDING_MODEL
export RUBRIC_GAIN_EMBEDDER_MODE=${RUBRIC_GAIN_EMBEDDER_MODE:-$GAIN_EMBEDDER_MODE}
PYTHON_BIN=${PYTHON_BIN:-}
if [[ -z "$PYTHON_BIN" ]]; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN=python3
  else
    PYTHON_BIN=python
  fi
fi

require_file() {
  local path="$1"
  if [[ ! -s "$path" ]]; then
    echo "Required file is missing or empty: $path" >&2
    exit 1
  fi
}

latest_checkpoint() {
  local dir="$1"
  find "$dir" -type d -name 'checkpoint-*' -print 2>/dev/null \
    | sort -V \
    | tail -n 1
}

if [[ "$RUN_PREPARE" == "1" ]]; then
  require_file "$CANDIDATE_LISTS"
  require_file "$RUBRIC_SCORES"
  mkdir -p "$OUT_DIR"

  "$PYTHON_BIN" scripts/merge_candidate_list_rubric.py \
    --candidate-lists "$CANDIDATE_LISTS" \
    --rubric-scores "$RUBRIC_SCORES" \
    --output "$COT_JUDGED" \
    --scored-examples-output "$SCORED_EXAMPLES"

  if [[ "$GAIN_MODE" == "ndcg" ]]; then
    require_file "$GAIN_ITEM_INFO"
  fi
  "$PYTHON_BIN" scripts/compute_cot_gain.py \
    --input "$COT_JUDGED" \
    --output "$COT_SCORED" \
    --embedder-mode "$GAIN_EMBEDDER_MODE" \
    --gain-mode "$GAIN_MODE" \
    --item-info "$GAIN_ITEM_INFO" \
    --ndcg-k "$GAIN_NDCG_K" \
    --model "$MODEL" \
    --embedding-model "$QWEN3_EMBEDDING_MODEL"

  SELECT_ARGS=()
  if [[ "$FALLBACK_WHEN_EMPTY" == "1" ]]; then
    SELECT_ARGS+=(--fallback-when-empty)
  fi

  "$PYTHON_BIN" scripts/select_filtered_cot.py \
    --input "$COT_SCORED" \
    --output "$FILTERED_COT" \
    --rejected-output "$REJECTED_COT" \
    --top-k "$TOP_K" \
    --min-rubric "$MIN_RUBRIC" \
    --min-gain "$MIN_GAIN" \
    "${SELECT_ARGS[@]}"

  "$PYTHON_BIN" scripts/make_sft_dataset.py \
    --input "$FILTERED_COT" \
    --output "$SFT_DATASET"

  "$PYTHON_BIN" scripts/make_grpo_dataset.py \
    --input "$SCORED_EXAMPLES" \
    --output "$GRPO_DATASET" \
    --baseline-mode "$GRPO_BASELINE_EMBEDDER_MODE" \
    --embedding-model "$QWEN3_EMBEDDING_MODEL"
fi

if [[ "$RUN_SFT" == "1" ]]; then
  require_file "$SFT_DATASET"
  DATASET="$SFT_DATASET" \
  OUT="$SFT_OUT" \
  MODEL="$MODEL" \
  MAX_STEPS="$SFT_MAX_STEPS" \
  NUM_TRAIN_EPOCHS="$SFT_NUM_TRAIN_EPOCHS" \
  SAVE_STEPS="$SFT_SAVE_STEPS" \
  bash scripts/run_sft_qwen3_4b.sh
fi

if [[ "$RUN_GRPO" == "1" ]]; then
  require_file "$GRPO_DATASET"
  if [[ -z "$ADAPTERS" ]]; then
    ADAPTERS=$(latest_checkpoint "$SFT_OUT")
  fi
  if [[ -z "$ADAPTERS" ]]; then
    echo "ADAPTERS is empty and no checkpoint was found under $SFT_OUT" >&2
    exit 1
  fi
  DATASET="$GRPO_DATASET" \
  OUT="$GRPO_OUT" \
  MODEL="$MODEL" \
  ADAPTERS="$ADAPTERS" \
  MAX_STEPS="$GRPO_MAX_STEPS" \
  NUM_GENERATIONS="$GRPO_NUM_GENERATIONS" \
  QWEN3_EMBEDDING_MODEL="$QWEN3_EMBEDDING_MODEL" \
  RUBRIC_GAIN_EMBEDDER_MODE="${RUBRIC_GAIN_EMBEDDER_MODE:-$GAIN_EMBEDDER_MODE}" \
  bash scripts/run_grpo_qwen3_4b.sh
fi

echo "RRec training pipeline artifacts:"
echo "  judged:   $COT_JUDGED"
echo "  scored:   $COT_SCORED"
echo "  filtered: $FILTERED_COT"
echo "  sft:      $SFT_DATASET"
echo "  grpo:     $GRPO_DATASET"
echo "  sft_out:  $SFT_OUT"
echo "  grpo_out: $GRPO_OUT"
