#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/root/autodl-tmp/rec/aaai_pro}
VENV=${VENV:-/root/autodl-tmp/rec/ms-swift-312-cu124-venv}
MODEL=${MODEL:-/root/autodl-tmp/modelscope_cache/models/Qwen/Qwen3-4B}
MODEL_TYPE=${MODEL_TYPE:-qwen3}
TEMPLATE=${TEMPLATE:-qwen3}
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
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-2}
SWIFT_MODEL_TYPE_FLAG=${SWIFT_MODEL_TYPE_FLAG:-}
NPROC_PER_NODE=${NPROC_PER_NODE:-auto}
MASTER_PORT=${MASTER_PORT:-29500}

activate_swift_env() {
  if command -v swift >/dev/null 2>&1; then
    return
  fi

  if [[ -n "${VENV:-}" && -x "$VENV/bin/swift" ]]; then
    export PATH="$VENV/bin:$PATH"
    return
  fi

  echo "Cannot find swift. Activate the swift env first or set VENV to an env that contains bin/swift." >&2
  exit 1
}

activate_swift_env
cd "$ROOT"

export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export MODELSCOPE_CACHE=${MODELSCOPE_CACHE:-/root/autodl-tmp/modelscope_cache}
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

resolve_nproc() {
  if [[ "$NPROC_PER_NODE" == "auto" ]]; then
    if [[ -z "${CUDA_VISIBLE_DEVICES:-}" ]]; then
      echo 1
      return
    fi
    awk -F',' '{print NF}' <<< "$CUDA_VISIBLE_DEVICES"
  else
    echo "$NPROC_PER_NODE"
  fi
}

SWIFT_BIN="$(command -v swift)"
NPROC="$(resolve_nproc)"
if ((NPROC < 1)); then
  echo "NPROC_PER_NODE must be >= 1" >&2
  exit 1
fi
export NPROC_PER_NODE="$NPROC"
export MASTER_PORT="$MASTER_PORT"

TRAIN_ARGS=(--train_type "$TRAIN_TYPE")
if [[ "$TRAIN_TYPE" == "lora" ]]; then
  TRAIN_ARGS+=(--lora_rank "$LORA_RANK" --lora_alpha "$LORA_ALPHA")
fi

STEP_ARGS=(--num_train_epochs "$NUM_TRAIN_EPOCHS")
if [[ "$MAX_STEPS" != "-1" ]]; then
  STEP_ARGS=(--max_steps "$MAX_STEPS")
fi

resolve_model_type_flag() {
  if [[ -n "$SWIFT_MODEL_TYPE_FLAG" ]]; then
    echo "$SWIFT_MODEL_TYPE_FLAG"
    return
  fi

  local help_text
  help_text="$(swift sft --help 2>&1 || true)"
  if grep -q -- "--model-type" <<<"$help_text" && ! grep -q -- "--model_type" <<<"$help_text"; then
    echo "--model-type"
  else
    echo "--model_type"
  fi
}

MODEL_TYPE_FLAG="$(resolve_model_type_flag)"

echo "SFT config:"
echo "  MODEL=$MODEL"
echo "  MODEL_TYPE=$MODEL_TYPE"
echo "  TEMPLATE=$TEMPLATE"
echo "  MODEL_TYPE_FLAG=$MODEL_TYPE_FLAG"
echo "  DATASET=$DATASET"
echo "  OUT=$OUT"
echo "  TRAIN_TYPE=$TRAIN_TYPE"
echo "  CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"
echo "  NPROC=$NPROC"
echo "  MASTER_PORT=$MASTER_PORT"

SFT_ARGS=(
  sft
  --model "$MODEL" \
  "$MODEL_TYPE_FLAG" "$MODEL_TYPE" \
  --template "$TEMPLATE" \
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
  --save_total_limit "$SAVE_TOTAL_LIMIT" \
  --logging_steps 10 \
  --report_to none \
  --output_dir "$OUT"
)

"$SWIFT_BIN" "${SFT_ARGS[@]}"
