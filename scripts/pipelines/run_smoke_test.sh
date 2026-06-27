#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/autodl-tmp/rec/aaai_pro}
SOURCE_ROOT=${SOURCE_ROOT:-/root/autodl-tmp/rec/Open-World-Knowledge-Augmented-Recommendation}
VENV=${VENV:-/root/autodl-tmp/rec/ms-swift-312-cu124-venv}
MODEL=${MODEL:-/root/autodl-tmp/modelscope_cache/models/Qwen/Qwen3-4B}

source "$VENV/bin/activate"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MODELSCOPE_CACHE=${MODELSCOPE_CACHE:-/root/autodl-tmp/modelscope_cache}
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

mkdir -p data/smoke outputs/smoke

python scripts/data/prepare_ml1m_examples.py \
  --source-root "$SOURCE_ROOT" \
  --max-users 3 \
  --max-history-items 20 \
  --output data/smoke/ml1m_examples.jsonl

python scripts/cot/generate_cot_candidates.py \
  --input data/smoke/ml1m_examples.jsonl \
  --output outputs/smoke/cot_candidates.jsonl \
  --model "$MODEL" \
  --max-examples 2 \
  --num-candidates 1 \
  --max-new-tokens 128

python scripts/cot/judge_cot_quality.py \
  --input outputs/smoke/cot_candidates.jsonl \
  --output outputs/smoke/cot_judged.jsonl \
  --judge-mode rules

python scripts/selection/compute_cot_gain.py \
  --input outputs/smoke/cot_judged.jsonl \
  --output outputs/smoke/cot_scored.jsonl \
  --embedder-mode lexical \
  --gain-mode sim

python scripts/selection/select_filtered_cot.py \
  --input outputs/smoke/cot_scored.jsonl \
  --output outputs/smoke/filtered_high_quality_cot.jsonl \
  --rejected-output outputs/smoke/rejected_cot.jsonl \
  --top-k 1 \
  --fallback-when-empty

python scripts/datasets/make_sft_dataset.py \
  --input outputs/smoke/filtered_high_quality_cot.jsonl \
  --output outputs/smoke/sft.jsonl

python scripts/datasets/make_grpo_dataset.py \
  --input data/smoke/ml1m_examples.jsonl \
  --output outputs/smoke/grpo.jsonl \
  --max-examples 2 \
  --precompute-lexical-baseline

python -m py_compile \
  rubric_cot_pipeline/*.py \
  scripts/data/prepare_ml1m_examples.py \
  scripts/cot/generate_cot_candidates.py \
  scripts/cot/judge_cot_quality.py \
  scripts/selection/compute_cot_gain.py \
  scripts/selection/select_filtered_cot.py \
  scripts/datasets/make_sft_dataset.py \
  scripts/datasets/make_grpo_dataset.py \
  scripts/train/rubric_gated_reward.py

echo "Smoke pipeline completed."
echo "SFT data:  $ROOT/outputs/smoke/sft.jsonl"
echo "GRPO data: $ROOT/outputs/smoke/grpo.jsonl"
