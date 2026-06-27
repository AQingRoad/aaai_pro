#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/autodl-tmp/rec/aaai_pro}
SOURCE_ROOT=${SOURCE_ROOT:-/root/autodl-tmp/rec/Open-World-Knowledge-Augmented-Recommendation}
VENV=${VENV:-/root/autodl-tmp/rec/ms-swift-312-cu124-venv}
MODEL=${MODEL:-/root/autodl-tmp/modelscope_cache/models/Qwen/Qwen3-4B}
MAX_USERS=${MAX_USERS:-1000}
NUM_CANDIDATES=${NUM_CANDIDATES:-4}
GAIN_EMBEDDER_MODE=${GAIN_EMBEDDER_MODE:-qwen_hidden}

source "$VENV/bin/activate"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MODELSCOPE_CACHE=${MODELSCOPE_CACHE:-/root/autodl-tmp/modelscope_cache}
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

mkdir -p data/ml1m outputs/ml1m

python scripts/data/prepare_ml1m_examples.py \
  --source-root "$SOURCE_ROOT" \
  --max-users "$MAX_USERS" \
  --output data/ml1m/examples.jsonl

python scripts/cot/generate_cot_candidates.py \
  --input data/ml1m/examples.jsonl \
  --output outputs/ml1m/cot_candidates.jsonl \
  --model "$MODEL" \
  --num-candidates "$NUM_CANDIDATES"

python scripts/cot/judge_cot_quality.py \
  --input outputs/ml1m/cot_candidates.jsonl \
  --output outputs/ml1m/cot_judged.jsonl \
  --judge-mode rules

python scripts/selection/compute_cot_gain.py \
  --input outputs/ml1m/cot_judged.jsonl \
  --output outputs/ml1m/cot_scored.jsonl \
  --embedder-mode "$GAIN_EMBEDDER_MODE" \
  --gain-mode sim \
  --model "$MODEL"

python scripts/selection/select_filtered_cot.py \
  --input outputs/ml1m/cot_scored.jsonl \
  --output outputs/ml1m/filtered_high_quality_cot.jsonl \
  --rejected-output outputs/ml1m/rejected_cot.jsonl \
  --top-k 1 \
  --min-rubric 0.5 \
  --min-gain 0.0

python scripts/datasets/make_sft_dataset.py \
  --input outputs/ml1m/filtered_high_quality_cot.jsonl \
  --output outputs/ml1m/sft.jsonl

python scripts/datasets/make_grpo_dataset.py \
  --input data/ml1m/examples.jsonl \
  --output outputs/ml1m/grpo.jsonl \
  --precompute-lexical-baseline

echo "Phase 1 artifacts are under $ROOT/outputs/ml1m"
