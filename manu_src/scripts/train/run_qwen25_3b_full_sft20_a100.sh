#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/user/aaai_pro}
VENV=${VENV:-/home/user/.conda/envs/aaai_pro}
MODEL=${MODEL:-/home/user/models_hf/Qwen2.5-3B-Instruct}
SPLIT_DIR=${SPLIT_DIR:-$ROOT/manu_src/datas/CDs_and_Vinyl/sft_grpo/disjoint_example20_80__input_time_title_rating_store_categories_desc256_details256_v1}
DATASET=${DATASET:-$SPLIT_DIR/sft_train20_seed42_n2144.jsonl}
OUT=${OUT:-$ROOT/manu_src/model_outputs/CDs_and_Vinyl/sft/qwen25_3b_full_sft20_disjoint_time_title_rating_store_categories_desc256_details256_v1_bs16_ga1_lr2e5_ep1_len4096_seed42}

EXPECTED_ROWS=${EXPECTED_ROWS:-2144}
BATCH_SIZE=${BATCH_SIZE:-16}
GRAD_ACCUM=${GRAD_ACCUM:-1}
MAX_LENGTH=${MAX_LENGTH:-4096}
EPOCHS=${EPOCHS:-1}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
SEED=${SEED:-42}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-2}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
CONFIRM_RUN=${CONFIRM_RUN:-0}

print_param() {
  local name=$1 value=$2 description=$3
  printf '  %-36s %s\n      %s\n' "$name=$value" "" "$description"
}

require_path() {
  local label=$1 path=$2
  if [[ ! -e "$path" ]]; then
    echo "缺少${label}: $path" >&2
    exit 1
  fi
}

for dependency in "$ROOT/AGENTS.md" "$VENV/bin/swift" "$MODEL" "$DATASET"; do
  require_path "SFT 依赖" "$dependency"
done
if [[ "$SEED" != "42" ]]; then
  echo "项目随机种子必须为 42。" >&2
  exit 1
fi
if [[ "$GRAD_ACCUM" != "1" ]]; then
  echo "当前实验固定 gradient_accumulation_steps=1。" >&2
  exit 1
fi
if [[ "$MAX_LENGTH" != "4096" ]]; then
  echo "当前实验固定 max_length=4096。" >&2
  exit 1
fi
if [[ "$EPOCHS" != "1" ]]; then
  echo "当前实验先执行 1 个 SFT epoch。" >&2
  exit 1
fi
if [[ -e "$MODEL/adapter_config.json" ]]; then
  echo "全参数 SFT 的 MODEL 必须是完整基座，不能是 LoRA adapter。" >&2
  exit 1
fi

rows=$(wc -l < "$DATASET" | tr -d ' ')
if [[ "$rows" != "$EXPECTED_ROWS" ]]; then
  echo "SFT 20% 数据应为 $EXPECTED_ROWS 条，当前为 $rows。" >&2
  exit 1
fi
save_steps=$(((rows + BATCH_SIZE - 1) / BATCH_SIZE))

echo "Qwen2.5-3B 前 20% 数据全参数 SFT 参数："
print_param ROOT "$ROOT" "A100 项目根目录；正式运行前重新读取 AGENTS.md。"
print_param MODEL "$MODEL" "Qwen2.5-3B-Instruct 完整基座模型。"
print_param DATASET "$DATASET" "固定 seed=42 划分得到的前 20% SFT messages，共 $rows 条。"
print_param INPUT_SCHEMA time_title_rating_store_categories_desc256_details256_v1 "history 字段截取口径；后续 GRPO 和评测必须保持一致。"
print_param SPLIT disjoint_example20_80 "SFT 与 GRPO 按 example_id 精确分区，交集为 0。"
print_param TRAIN_TYPE full "更新 Qwen2.5-3B 的全部参数，输出独立完整模型。"
print_param BATCH_SIZE "$BATCH_SIZE" "单卡 micro batch；未执行额外显存探测。"
print_param GRAD_ACCUM "$GRAD_ACCUM" "梯度累积固定为 1，有效 batch 等于 micro batch。"
print_param MAX_LENGTH "$MAX_LENGTH" "完整 chat 序列 token 上限；超长样本从左侧截断并保留 assistant CoT。"
print_param EPOCHS "$EPOCHS" "SFT 训练 1 个完整 epoch。"
print_param LEARNING_RATE "$LEARNING_RATE" "全参数 AdamW 峰值学习率；本轮按确认值使用 2e-5。"
print_param WEIGHT_DECAY "$WEIGHT_DECAY" "AdamW 权重衰减。"
print_param WARMUP_RATIO "$WARMUP_RATIO" "前 3% optimizer step 线性 warmup。"
print_param SAVE_STEPS "$save_steps" "在单个 epoch 结束时保存完整模型 checkpoint。"
print_param ATTN_IMPL flash_attn "使用 FlashAttention 处理 4096-token 长序列。"
print_param PADDING_FREE true "展平 batch 内有效 token，减少 padding 计算。"
print_param GRADIENT_CHECKPOINTING true "重算中间激活以降低全参数训练显存。"
print_param OUT "$OUT" "完整 SFT 模型、训练状态和日志输出根目录。"
print_param SEED "$SEED" "模型初始化、DataLoader shuffle 和训练随机种子固定为 42。"
print_param CONFIRM_RUN "$CONFIRM_RUN" "必须为 1 才启动正式训练。"

if [[ "$CONFIRM_RUN" != "1" ]]; then
  echo "未启动训练。"
  exit 0
fi

source /opt/miniforge3/bin/activate "$VENV"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES
export NPROC_PER_NODE=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

mkdir -p "$OUT"
nvidia-smi -q -d ECC > "$OUT/gpu_ecc_before_train.txt"
nvidia-smi -q -d ROW_REMAPPER > "$OUT/gpu_row_remapper_before_train.txt"

"$VENV/bin/swift" sft \
  --model "$MODEL" \
  --model_type qwen2_5 \
  --template qwen2_5 \
  --dataset "$DATASET" \
  --train_type full \
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
  --save_steps "$save_steps" \
  --save_total_limit "$SAVE_TOTAL_LIMIT" \
  --save_only_model true \
  --logging_steps 1 \
  --report_to none \
  --dataloader_num_workers 0 \
  --output_dir "$OUT"
