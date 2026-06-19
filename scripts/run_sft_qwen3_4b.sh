#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/autodl-tmp/rec/aaai_pro}
VENV=${VENV:-/root/autodl-tmp/rec/ms-swift-312-cu124-venv}
MODEL=${MODEL:-/root/autodl-tmp/modelscope_cache/models/Qwen/Qwen3-4B}
DATASET=${DATASET:-$ROOT/outputs/ml1m/sft.jsonl}
OUT=${OUT:-$ROOT/checkpoints/qwen3_4b_sft_rubric_cot}
TRAIN_TYPE=${TRAIN_TYPE:-lora}
LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-128}
MAX_STEPS=${MAX_STEPS:--1}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
BATCH_SIZE=${BATCH_SIZE:-1}
GRAD_ACCUM=${GRAD_ACCUM:-8}
MAX_LENGTH=${MAX_LENGTH:-2048}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
SAVE_STEPS=${SAVE_STEPS:-200}

source "$VENV/bin/activate"
cd "$ROOT"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MODELSCOPE_CACHE=${MODELSCOPE_CACHE:-/root/autodl-tmp/modelscope_cache}
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

TRAIN_ARGS=(--train_type "$TRAIN_TYPE")
if [[ "$TRAIN_TYPE" == "lora" ]]; then
  TRAIN_ARGS+=(--lora_rank "$LORA_RANK" --lora_alpha "$LORA_ALPHA")
fi

STEP_ARGS=(--num_train_epochs "$NUM_TRAIN_EPOCHS")
if [[ "$MAX_STEPS" != "-1" ]]; then
  STEP_ARGS=(--max_steps "$MAX_STEPS")
fi

swift sft \
  --model "$MODEL" \
  --dataset "$DATASET" \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --max_length "$MAX_LENGTH" \
  --learning_rate "$LEARNING_RATE" \
  --lr_scheduler_type cosine \
  --warmup_ratio 0.03 \
  "${STEP_ARGS[@]}" \
  "${TRAIN_ARGS[@]}" \
  --torch_dtype bfloat16 \
  --gradient_checkpointing true \
  --save_only_model true \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit 2 \
  --logging_steps 10 \
  --report_to none \
  --output_dir "$OUT"
