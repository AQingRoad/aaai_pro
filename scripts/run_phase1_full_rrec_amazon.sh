#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/autodl-tmp/rec/aaai_pro}
RREC_DATA_ROOT=${RREC_DATA_ROOT:-/root/autodl-tmp/rec/RRec_official/data}
VENV=${VENV:-/root/autodl-tmp/rec/ms-swift-312-cu124-venv}
MODEL=${MODEL:-/root/autodl-tmp/modelscope_cache/models/Qwen/Qwen3-4B}
QWEN3_EMBEDDING_MODEL=${QWEN3_EMBEDDING_MODEL:-/root/autodl-tmp/modelscope_cache/models/Qwen/Qwen3-Embedding-0.6B}
CATEGORY=${CATEGORY:-Musical_Instruments}
SPLIT=${SPLIT:-train}
MAX_EXAMPLES=${MAX_EXAMPLES:-1000}
NUM_CANDIDATES=${NUM_CANDIDATES:-4}
GAIN_EMBEDDER_MODE=${GAIN_EMBEDDER_MODE:-qwen3_embedding}
GAIN_MODE=${GAIN_MODE:-ndcg}
GAIN_NDCG_K=${GAIN_NDCG_K:-100}
GAIN_ITEM_INFO=${GAIN_ITEM_INFO:-$ROOT/github_artifacts/$CATEGORY/rrec_eval/item_info.jsonl}
GRPO_BASELINE_EMBEDDER_MODE=${GRPO_BASELINE_EMBEDDER_MODE:-$GAIN_EMBEDDER_MODE}
JUDGE_MODE=${JUDGE_MODE:-api}
API_PROVIDER=${API_PROVIDER:-openai_compatible}
API_WORKERS=${API_WORKERS:-1}
API_MODEL=${API_MODEL:-glm-5-1}
API_BASE_URL=${API_BASE_URL:-${RUBRIC_JUDGE_API_BASE_URL:-http://127.0.0.1:18080/v1}}
API_KEY=${API_KEY:-${RUBRIC_JUDGE_API_KEY:-}}
API_TIMEOUT=${API_TIMEOUT:-${RUBRIC_JUDGE_API_TIMEOUT:-60}}
API_MAX_RETRIES=${API_MAX_RETRIES:-${RUBRIC_JUDGE_API_MAX_RETRIES:-2}}
API_MIN_INTERVAL=${API_MIN_INTERVAL:-0}
GENERATION_MODE=${GENERATION_MODE:-local}
COT_API_PROVIDER=${COT_API_PROVIDER:-$API_PROVIDER}
COT_API_MODEL=${COT_API_MODEL:-$API_MODEL}
COT_API_BASE_URL=${COT_API_BASE_URL:-$API_BASE_URL}
COT_API_KEY=${COT_API_KEY:-$API_KEY}
COT_API_TIMEOUT=${COT_API_TIMEOUT:-120}
COT_API_MAX_RETRIES=${COT_API_MAX_RETRIES:-2}
COT_API_MIN_INTERVAL=${COT_API_MIN_INTERVAL:-0}
MAX_HISTORY_ITEMS=${MAX_HISTORY_ITEMS:-20}

source "$VENV/bin/activate"
cd "$ROOT"
export RUBRIC_JUDGE_API_PROVIDER="$API_PROVIDER"
export RUBRIC_JUDGE_API_MODEL="$API_MODEL"
export RUBRIC_JUDGE_API_WORKERS="$API_WORKERS"
if [[ -n "$API_BASE_URL" ]]; then
  export RUBRIC_JUDGE_API_BASE_URL="$API_BASE_URL"
fi
if [[ -n "$API_KEY" ]]; then
  export RUBRIC_JUDGE_API_KEY="$API_KEY"
fi
export RUBRIC_JUDGE_API_MIN_INTERVAL="$API_MIN_INTERVAL"
export RUBRIC_JUDGE_API_TIMEOUT="$API_TIMEOUT"
export RUBRIC_JUDGE_API_MAX_RETRIES="$API_MAX_RETRIES"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MODELSCOPE_CACHE=${MODELSCOPE_CACHE:-/root/autodl-tmp/modelscope_cache}
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

OUT_DIR="outputs/rrec_amazon/${CATEGORY}"
DATA_DIR="data/rrec_amazon/${CATEGORY}"
EXAMPLES_FILE=${EXAMPLES_FILE:-"$DATA_DIR/phase1_examples.jsonl"}
mkdir -p "$DATA_DIR" "$OUT_DIR"

python scripts/prepare_rrec_amazon_examples.py \
  --data-root "$RREC_DATA_ROOT" \
  --category "$CATEGORY" \
  --split "$SPLIT" \
  --max-examples "$MAX_EXAMPLES" \
  --max-history-items "$MAX_HISTORY_ITEMS" \
  --shuffle \
  --output "$EXAMPLES_FILE"

python scripts/generate_cot_candidates.py \
  --input "$EXAMPLES_FILE" \
  --output "$OUT_DIR/cot_candidates.jsonl" \
  --model "$MODEL" \
  --num-candidates "$NUM_CANDIDATES" \
  --generation-mode "$GENERATION_MODE" \
  --api-provider "$COT_API_PROVIDER" \
  --api-base-url "$COT_API_BASE_URL" \
  --api-key "$COT_API_KEY" \
  --api-model "$COT_API_MODEL" \
  --api-timeout "$COT_API_TIMEOUT" \
  --api-max-retries "$COT_API_MAX_RETRIES" \
  --api-min-interval "$COT_API_MIN_INTERVAL"

python scripts/judge_cot_quality.py \
  --input "$OUT_DIR/cot_candidates.jsonl" \
  --output "$OUT_DIR/cot_judged.jsonl" \
  --judge-mode "$JUDGE_MODE" \
  --model "$MODEL" \
  --api-provider "$API_PROVIDER" \
  --api-base-url "${RUBRIC_JUDGE_API_BASE_URL:-}" \
  --api-key "${RUBRIC_JUDGE_API_KEY:-}" \
  --api-model "$API_MODEL" \
  --api-timeout "$API_TIMEOUT" \
  --api-max-retries "$API_MAX_RETRIES" \
  --api-workers "$API_WORKERS"

python scripts/compute_cot_gain.py \
  --input "$OUT_DIR/cot_judged.jsonl" \
  --output "$OUT_DIR/cot_scored.jsonl" \
  --embedder-mode "$GAIN_EMBEDDER_MODE" \
  --gain-mode "$GAIN_MODE" \
  --item-info "$GAIN_ITEM_INFO" \
  --ndcg-k "$GAIN_NDCG_K" \
  --model "$MODEL" \
  --embedding-model "$QWEN3_EMBEDDING_MODEL"

python scripts/select_filtered_cot.py \
  --input "$OUT_DIR/cot_scored.jsonl" \
  --output "$OUT_DIR/filtered_high_quality_cot.jsonl" \
  --rejected-output "$OUT_DIR/rejected_cot.jsonl" \
  --top-k 1 \
  --min-rubric 0.5 \
  --min-gain 0.0 \
  --fallback-when-empty

python scripts/make_sft_dataset.py \
  --input "$OUT_DIR/filtered_high_quality_cot.jsonl" \
  --output "$OUT_DIR/sft.jsonl"

python scripts/make_grpo_dataset.py \
  --input "$EXAMPLES_FILE" \
  --output "$OUT_DIR/grpo.jsonl" \
  --baseline-mode "$GRPO_BASELINE_EMBEDDER_MODE" \
  --embedding-model "$QWEN3_EMBEDDING_MODEL"

echo "RRec/Amazon Phase 1 artifacts are under $ROOT/$OUT_DIR"
