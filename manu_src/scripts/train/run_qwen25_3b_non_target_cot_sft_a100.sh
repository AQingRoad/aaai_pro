#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/user/aaai_pro}
VENV=${VENV:-/home/user/.conda/envs/aaai_pro}
MODEL=${MODEL:-/home/user/models_hf/Qwen2.5-3B-Instruct}
DATASET=${DATASET:-$ROOT/manu_src/datas/CDs_and_Vinyl/sft/history_only_next_item_feature_cot__input_time_title_rating_store_categories_desc256_details256_v1/sft_messages_full_seed42.jsonl}
OUT=${OUT:-$ROOT/manu_src/model_outputs/CDs_and_Vinyl/sft/qwen25_3b_lora_history_only_next_item_feature_cot_time_title_rating_store_categories_desc256_details256_v1_bs32_ga1_lr2e5_ep5_len4096_seed42}

BATCH_SIZE=${BATCH_SIZE:-32}
GRAD_ACCUM=${GRAD_ACCUM:-1}
MAX_LENGTH=${MAX_LENGTH:-4096}
EPOCHS=${EPOCHS:-5}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-128}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
SEED=${SEED:-42}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-6}
EXPECTED_ROWS=${EXPECTED_ROWS:-10722}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
CONFIRM_RUN=${CONFIRM_RUN:-0}

print_param() {
  local name=$1 value=$2 description=$3
  printf '  %-34s %s\n      %s\n' "$name=$value" "" "$description"
}

for path in "$ROOT/AGENTS.md" "$VENV/bin/swift" "$MODEL" "$DATASET"; do
  if [[ ! -e "$path" ]]; then
    echo "缺少训练依赖：$path" >&2
    exit 1
  fi
done
if [[ "$SEED" != "42" || "$GRAD_ACCUM" != "1" || "$BATCH_SIZE" != "32" ]]; then
  echo "本实验固定 seed=42、batch_size=32、grad_accum=1。" >&2
  exit 1
fi

ROWS=$(wc -l < "$DATASET" | tr -d ' ')
if [[ "$ROWS" != "$EXPECTED_ROWS" ]]; then
  echo "SFT 数据应为 $EXPECTED_ROWS 条，当前为 $ROWS。" >&2
  exit 1
fi
SAVE_STEPS=$(((ROWS + BATCH_SIZE - 1) / BATCH_SIZE))

echo "Qwen2.5-3B LoRA SFT 正式训练参数："
print_param ROOT "$ROOT" "A100 上的项目根目录。"
print_param MODEL "$MODEL" "Qwen2.5-3B-Instruct 基座模型。"
print_param DATASET "$DATASET" "$ROWS 条英文 history-only <analysis>/<answer> SFT messages。"
print_param EXPECTED_ROWS "$EXPECTED_ROWS" "训练数据预期行数；用于阻止误用全量或错误子集。"
print_param OUT "$OUT" "LoRA checkpoint 与训练日志输出目录。"
print_param BATCH_SIZE "$BATCH_SIZE" "单卡 micro batch；已用最长序列完成 2 step 压力测试。"
print_param GRAD_ACCUM "$GRAD_ACCUM" "梯度累积固定为 1，有效 batch 为 32。"
print_param MAX_LENGTH "$MAX_LENGTH" "完整 chat 序列上限；超长样本从左侧截断，保留 assistant CoT。"
print_param EPOCHS "$EPOCHS" "完整训练轮数；每轮保存一次。"
print_param LEARNING_RATE "$LEARNING_RATE" "LoRA AdamW 峰值学习率。"
print_param LORA_RANK "$LORA_RANK" "LoRA 低秩维度。"
print_param LORA_ALPHA "$LORA_ALPHA" "LoRA 缩放系数。"
print_param SAVE_STEPS "$SAVE_STEPS" "按样本数与 batch 自动计算，每个 epoch 保存一次。"
print_param SAVE_TOTAL_LIMIT "$SAVE_TOTAL_LIMIT" "最多保留 6 个 checkpoint，覆盖 5 个 epoch。"
print_param TRUNCATION_STRATEGY left "仅 SFT chat 序列使用左截断。"
print_param ATTN_IMPL flash_attn "使用 FlashAttention 降低长序列显存和耗时。"
print_param PADDING_FREE true "将同一 batch 的有效 token 展平计算。"
print_param SEED "$SEED" "模型、数据顺序和采样随机种子。"
print_param CONFIRM_RUN "$CONFIRM_RUN" "必须为 1 才执行正式训练。"

if [[ "$CONFIRM_RUN" != "1" ]]; then
  echo "未启动。"
  exit 0
fi

source /opt/miniforge3/bin/activate "$VENV"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES
export NPROC_PER_NODE=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p "$OUT"

"$VENV/bin/swift" sft \
  --model "$MODEL" \
  --model_type qwen2_5 \
  --template qwen2_5 \
  --dataset "$DATASET" \
  --train_type lora \
  --lora_rank "$LORA_RANK" \
  --lora_alpha "$LORA_ALPHA" \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --max_length "$MAX_LENGTH" \
  --truncation_strategy left \
  --attn_impl flash_attn \
  --padding_free true \
  --packing false \
  --learning_rate "$LEARNING_RATE" \
  --weight_decay "$WEIGHT_DECAY" \
  --lr_scheduler_type cosine \
  --warmup_ratio "$WARMUP_RATIO" \
  --num_train_epochs "$EPOCHS" \
  --torch_dtype bfloat16 \
  --gradient_checkpointing true \
  --seed "$SEED" \
  --data_seed "$SEED" \
  --save_strategy steps \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit "$SAVE_TOTAL_LIMIT" \
  --save_only_model true \
  --logging_steps 10 \
  --report_to none \
  --dataloader_num_workers 0 \
  --output_dir "$OUT"
