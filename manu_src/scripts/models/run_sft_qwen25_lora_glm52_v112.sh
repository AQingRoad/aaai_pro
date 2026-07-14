#!/usr/bin/env bash
set -euo pipefail

# 服务器目录按 manu_src 组织；环境变量允许复用脚本时显式覆盖参数。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANU_ROOT=${MANU_ROOT:-"$(cd "$SCRIPT_DIR/../.." && pwd)"}
VENV=${VENV:-/home/user/.conda/envs/aaai_pro}
MODEL=${MODEL:-/home/user/models_hf/Qwen2.5-3B-Instruct}
MODEL_TYPE=${MODEL_TYPE:-qwen2_5}
TEMPLATE=${TEMPLATE:-qwen2_5}
DATASET=${DATASET:-$MANU_ROOT/datas/CDs_and_Vinyl/sft_datas/sft_glm52_v1.12_train_head20_query_only_seed42_n2144_len4096.jsonl}
OUT=${OUT:-$MANU_ROOT/model_outputs/CDs_and_Vinyl/sft/qwen25_3b_lora_glm52_v1p12_train_head20_metadata_rich_len4096_bs16_ga1_lr1e5_ep3_seed42}

TRAIN_TYPE=${TRAIN_TYPE:-lora}
TARGET_MODULES=${TARGET_MODULES:-all-linear}
LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-128}
EPOCHS=${EPOCHS:-3}
BATCH_SIZE=${BATCH_SIZE:-16}
GRAD_ACCUM=${GRAD_ACCUM:-1}
MAX_LENGTH=${MAX_LENGTH:-4096}
LEARNING_RATE=${LEARNING_RATE:-1e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.1}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
LR_SCHEDULER=${LR_SCHEDULER:-cosine}
OPTIM=${OPTIM:-adamw_torch}
LOGGING_STEPS=${LOGGING_STEPS:-10}
SAVE_STEPS=${SAVE_STEPS:-auto}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-10}
SEED=${SEED:-42}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
CONFIRM_RUN=${CONFIRM_RUN:-0}

require_path() {
  local label="$1"
  local path="$2"
  if [[ ! -e "$path" ]]; then
    echo "缺少${label}: $path" >&2
    exit 1
  fi
}

print_param() {
  local name="$1"
  local value="$2"
  local description="$3"
  printf '  %-30s\n      %s\n' "$name=$value" "$description"
}

require_path "manu_src 根目录" "$MANU_ROOT"
require_path "Qwen2.5-3B-Instruct 模型" "$MODEL"
require_path "SFT 训练数据" "$DATASET"
require_path "Conda 环境" "$VENV"

if [[ "$SEED" != "42" ]]; then
  echo "项目随机种子固定为 42，当前为 $SEED" >&2
  exit 1
fi

export PATH="$VENV/bin:$PATH"
export CUDA_VISIBLE_DEVICES
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

rows="$(wc -l < "$DATASET" | tr -d ' ')"
if [[ "$rows" != "2144" ]]; then
  echo "SFT 数据应为 2144 行，当前为 $rows" >&2
  exit 1
fi

# 单卡下每轮更新数为 ceil(样本数 / micro_batch / 梯度累积步数)。
if [[ "$SAVE_STEPS" == "auto" ]]; then
  effective_batch=$((BATCH_SIZE * GRAD_ACCUM))
  SAVE_STEPS=$(((rows + effective_batch - 1) / effective_batch))
fi

echo "即将启动 QUERY-only CoT LoRA SFT，关键参数如下："
print_param "MANU_ROOT" "$MANU_ROOT" "服务器上的 manu_src 根目录；数据、脚本和输出均按此目录组织。"
print_param "MODEL" "$MODEL" "Qwen2.5-3B-Instruct 基座权重和 tokenizer 路径。"
print_param "MODEL_TYPE" "$MODEL_TYPE" "告诉 ms-swift 按 Qwen2.5 架构加载模型。"
print_param "TEMPLATE" "$TEMPLATE" "使用 Qwen2.5 chat template 拼接 messages，与预处理 token 统计保持一致。"
print_param "DATASET" "$DATASET" "2144 条 messages 格式训练样本，只读取 QUERY 并监督完整 think/answer。"
print_param "OUT" "$OUT" "本实验 checkpoint 和训练日志输出目录。"
print_param "TRAIN_TYPE" "$TRAIN_TYPE" "使用 LoRA 微调，基座参数保持冻结。"
print_param "TARGET_MODULES" "$TARGET_MODULES" "向模型内全部线性层注入 LoRA adapter。"
print_param "LORA_RANK" "$LORA_RANK" "LoRA 低秩矩阵维度。"
print_param "LORA_ALPHA" "$LORA_ALPHA" "LoRA 更新的缩放系数。"
print_param "EPOCHS" "$EPOCHS" "完整遍历训练数据的轮数。"
print_param "BATCH_SIZE" "$BATCH_SIZE" "单卡每次前向和反向使用的样本数。"
print_param "GRAD_ACCUM" "$GRAD_ACCUM" "累计多少个 micro batch 后更新一次参数；有效 batch 为 $((BATCH_SIZE * GRAD_ACCUM))。"
print_param "MAX_LENGTH" "$MAX_LENGTH" "system、query 和完整 CoT 合计允许的最大 token 数。"
print_param "LEARNING_RATE" "$LEARNING_RATE" "AdamW 对 LoRA 参数使用的初始学习率。"
print_param "WEIGHT_DECAY" "$WEIGHT_DECAY" "AdamW 权重衰减系数。"
print_param "WARMUP_RATIO" "$WARMUP_RATIO" "总更新步数中用于学习率预热的比例。"
print_param "LR_SCHEDULER" "$LR_SCHEDULER" "预热结束后使用余弦学习率衰减。"
print_param "OPTIM" "$OPTIM" "使用 PyTorch AdamW 更新 LoRA 参数。"
print_param "SPLIT_DATASET_RATIO" "0" "不从训练集切分验证集，2144 条样本全部参与训练。"
print_param "TORCH_DTYPE" "bfloat16" "模型前向和反向使用 BF16 精度。"
print_param "ATTN_IMPL" "flash_attn" "使用 FlashAttention 降低 4096-token 序列的显存和计算开销。"
print_param "GRADIENT_CHECKPOINTING" "true" "反向传播时重算部分激活，降低显存占用。"
print_param "PACKING" "false" "每条样本独立编码，禁止把多条会话拼进同一序列。"
print_param "LOSS_SCALE" "default" "使用 ms-swift 对话模板的默认 assistant-token loss 掩码。"
print_param "EVAL_STRATEGY" "no" "训练期间不运行验证或测试。"
print_param "SAVE_STRATEGY" "steps" "按优化器更新步数保存 checkpoint。"
print_param "SAVE_STEPS" "$SAVE_STEPS" "每轮约保存一次 checkpoint；2144 条样本对应 134 个更新步。"
print_param "SAVE_TOTAL_LIMIT" "$SAVE_TOTAL_LIMIT" "最多保留的 checkpoint 数；本实验预计生成 3 个，不会触发删除。"
print_param "SAVE_ONLY_MODEL" "false" "同时保存 adapter、优化器和调度器状态，支持断点恢复。"
print_param "LOGGING_STEPS" "$LOGGING_STEPS" "每 10 个参数更新步记录一次训练指标。"
print_param "REPORT_TO" "none" "不向外部实验跟踪服务发送训练日志。"
print_param "SEED" "$SEED" "固定模型初始化、数据 shuffle 和训练随机状态。"
print_param "CUDA_VISIBLE_DEVICES" "$CUDA_VISIBLE_DEVICES" "使用 A100 的 GPU 编号。"
print_param "PYTORCH_CUDA_ALLOC_CONF" "$PYTORCH_CUDA_ALLOC_CONF" "启用可扩展显存段，减少长序列训练中的显存碎片。"
print_param "CONFIRM_RUN" "$CONFIRM_RUN" "值为 1 时才提交正式训练。"

if [[ "$CONFIRM_RUN" != "1" ]]; then
  echo "参数仅打印，训练尚未启动。确认后设置 CONFIRM_RUN=1。"
  exit 0
fi

mkdir -p "$OUT"

swift sft \
  --model "$MODEL" \
  --model_type "$MODEL_TYPE" \
  --template "$TEMPLATE" \
  --dataset "$DATASET" \
  --split_dataset_ratio 0 \
  --train_type "$TRAIN_TYPE" \
  --target_modules "$TARGET_MODULES" \
  --lora_rank "$LORA_RANK" \
  --lora_alpha "$LORA_ALPHA" \
  --num_train_epochs "$EPOCHS" \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --max_length "$MAX_LENGTH" \
  --learning_rate "$LEARNING_RATE" \
  --weight_decay "$WEIGHT_DECAY" \
  --warmup_ratio "$WARMUP_RATIO" \
  --lr_scheduler_type "$LR_SCHEDULER" \
  --optim "$OPTIM" \
  --torch_dtype bfloat16 \
  --attn_impl flash_attn \
  --gradient_checkpointing true \
  --packing false \
  --loss_scale default \
  --eval_strategy no \
  --save_strategy steps \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit "$SAVE_TOTAL_LIMIT" \
  --save_only_model false \
  --logging_steps "$LOGGING_STEPS" \
  --report_to none \
  --seed "$SEED" \
  --output_dir "$OUT"
