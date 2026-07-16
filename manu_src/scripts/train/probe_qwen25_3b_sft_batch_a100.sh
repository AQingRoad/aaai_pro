#!/usr/bin/env bash
set -uo pipefail

ROOT=${ROOT:-/home/user/aaai_pro}
VENV=${VENV:-/home/user/.conda/envs/aaai_pro}
MODEL=${MODEL:-/home/user/models_hf/Qwen2.5-3B-Instruct}
DATASET=${DATASET:-$ROOT/manu_src/datas/CDs_and_Vinyl/sft/history_only_next_item_feature_cot__input_time_title_rating_store_categories_desc256_details256_v1/sft_messages_full_seed42.jsonl}
START_BATCH=${START_BATCH:-32}
MAX_PROBE_BATCH=${MAX_PROBE_BATCH:-512}
MAX_STEPS=${MAX_STEPS:-2}
MAX_LENGTH=${MAX_LENGTH:-4096}
SEED=${SEED:-42}

TMP_ROOT=${TMP_ROOT:-/home/user/tmp/qwen25_3b_sft_batch_probe_seed42}
PROBE_DATASET=$TMP_ROOT/longest_1068.jsonl
SWIFT_BIN=$VENV/bin/swift
PYTHON_BIN=$VENV/bin/python

cleanup() {
  rm -rf "$TMP_ROOT"
}
trap cleanup EXIT

case "$TMP_ROOT" in
  /home/user/tmp/qwen25_3b_sft_batch_probe_*) ;;
  *)
    echo "TMP_ROOT 必须位于 /home/user/tmp/qwen25_3b_sft_batch_probe_* 下" >&2
    exit 1
    ;;
esac

for path in "$MODEL" "$DATASET" "$SWIFT_BIN" "$PYTHON_BIN"; do
  if [[ ! -e "$path" ]]; then
    echo "缺少探测依赖：$path" >&2
    exit 1
  fi
done
if [[ "$SEED" != "42" ]]; then
  echo "项目随机种子必须为 42" >&2
  exit 1
fi

mkdir -p "$TMP_ROOT"
cd "$ROOT"
export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export NPROC_PER_NODE=1
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

echo "构造最长序列压力测试集：读取完整 chat template token 长度，选取最长 1068 条。"
"$PYTHON_BIN" - "$DATASET" "$MODEL" "$PROBE_DATASET" <<'PY'
import json
import sys
from transformers import AutoTokenizer

source, model, output = sys.argv[1:]
tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, use_fast=True)
rows = []
for line in open(source, encoding="utf-8"):
    if not line.strip():
        continue
    row = json.loads(line)
    length = len(tokenizer.apply_chat_template(row["messages"], tokenize=True, add_generation_prompt=False))
    rows.append((length, row))
rows.sort(key=lambda pair: pair[0], reverse=True)
selected = rows[:1068]
with open(output, "w", encoding="utf-8") as file:
    for _, row in selected:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")
print(json.dumps({
    "source_rows": len(rows),
    "selected_rows": len(selected),
    "raw_token_min": min(length for length, _ in selected),
    "raw_token_max": max(length for length, _ in selected),
    "train_max_length": 4096,
    "truncation_strategy": "left",
}, ensure_ascii=False))
PY

declare -A PEAK_MIB
declare -A STATUS

run_probe() {
  local batch=$1
  local run_dir=$TMP_ROOT/batch_$batch
  local log=$TMP_ROOT/batch_$batch.log
  local memory_log=$TMP_ROOT/batch_$batch.memory
  rm -rf "$run_dir"
  : > "$log"
  : > "$memory_log"

  echo "PROBE_START batch=$batch grad_accum=1 max_length=$MAX_LENGTH max_steps=$MAX_STEPS"
  nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 200 > "$memory_log" 2>/dev/null &
  local monitor_pid=$!

  "$SWIFT_BIN" sft \
    --model "$MODEL" \
    --model_type qwen2_5 \
    --template qwen2_5 \
    --dataset "$PROBE_DATASET" \
    --train_type lora \
    --lora_rank 64 \
    --lora_alpha 128 \
    --per_device_train_batch_size "$batch" \
    --gradient_accumulation_steps 1 \
    --max_length "$MAX_LENGTH" \
    --truncation_strategy left \
    --attn_impl flash_attn \
    --padding_free true \
    --packing false \
    --learning_rate 2e-5 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.03 \
    --max_steps "$MAX_STEPS" \
    --torch_dtype bfloat16 \
    --gradient_checkpointing true \
    --seed "$SEED" \
    --data_seed "$SEED" \
    --save_strategy no \
    --logging_steps 1 \
    --report_to none \
    --dataloader_num_workers 0 \
    --output_dir "$run_dir" \
    > "$log" 2>&1
  local code=$?

  kill "$monitor_pid" 2>/dev/null || true
  wait "$monitor_pid" 2>/dev/null || true
  local peak=0
  if [[ -s "$memory_log" ]]; then
    peak=$(sort -nr "$memory_log" | head -1 | tr -d ' ')
  fi
  PEAK_MIB[$batch]=${peak:-0}
  rm -rf "$run_dir"

  if [[ $code -eq 0 ]]; then
    STATUS[$batch]=PASS
    echo "PROBE_RESULT batch=$batch status=PASS peak_mib=${PEAK_MIB[$batch]}"
    grep -E "train_runtime|train_samples_per_second|train_steps_per_second|train_loss" "$log" | tail -8 || true
    return 0
  fi
  if grep -Eqi "CUDA out of memory|OutOfMemoryError|CUDA error: out of memory" "$log"; then
    STATUS[$batch]=OOM
    echo "PROBE_RESULT batch=$batch status=OOM peak_mib=${PEAK_MIB[$batch]}"
    return 1
  fi
  STATUS[$batch]=ERROR
  echo "PROBE_RESULT batch=$batch status=ERROR exit_code=$code peak_mib=${PEAK_MIB[$batch]}" >&2
  tail -80 "$log" >&2
  return 2
}

low=0
high=0
candidate=$START_BATCH

while (( candidate <= MAX_PROBE_BATCH )); do
  run_probe "$candidate"
  code=$?
  if (( code == 0 )); then
    low=$candidate
    if (( candidate == MAX_PROBE_BATCH )); then
      high=$((MAX_PROBE_BATCH + 1))
      break
    fi
    candidate=$((candidate * 2))
    if (( candidate > MAX_PROBE_BATCH )); then
      candidate=$MAX_PROBE_BATCH
    fi
  elif (( code == 1 )); then
    high=$candidate
    break
  else
    exit "$code"
  fi
done

if (( low == 0 )); then
  high=${high:-$START_BATCH}
fi
if (( high == 0 )); then
  high=$((MAX_PROBE_BATCH + 1))
fi

while (( high - low > 1 )); do
  candidate=$(((low + high) / 2))
  if (( candidate < 1 )); then
    candidate=1
  fi
  run_probe "$candidate"
  code=$?
  if (( code == 0 )); then
    low=$candidate
  elif (( code == 1 )); then
    high=$candidate
  else
    exit "$code"
  fi
done

echo "PROBE_SUMMARY max_stable_batch=$low first_oom_batch=$high grad_accum=1 max_length=$MAX_LENGTH seed=$SEED"
for batch in "${!STATUS[@]}"; do
  echo "PROBE_CASE batch=$batch status=${STATUS[$batch]} peak_mib=${PEAK_MIB[$batch]}"
done | sort -t= -k2,2n
