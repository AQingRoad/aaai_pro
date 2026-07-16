#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/user/aaai_pro}
VENV=${VENV:-/home/user/.conda/envs/aaai_pro}
MODEL=${MODEL:-/home/user/models_hf/Qwen3-Embedding-0.6B}
TRAIN_FILE=${TRAIN_FILE:-$ROOT/manu_src/datas/CDs_and_Vinyl/embedding/history_plus_non_target_cot__input_time_title_rating_store_categories_desc256_details256_v1/train.jsonl}
COT_MASK_PROB=${COT_MASK_PROB:-0.0}

BATCH_SIZE=${BATCH_SIZE:-64}
GRAD_ACCUM=${GRAD_ACCUM:-1}
MAX_LENGTH=${MAX_LENGTH:-4096}
EPOCHS=${EPOCHS:-5}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
SEED=${SEED:-42}
CONFIRM_RUN=${CONFIRM_RUN:-0}

if [[ "$COT_MASK_PROB" == "0" || "$COT_MASK_PROB" == "0.0" ]]; then
  RUN_VARIANT=history_plus_cot
  DEFAULT_OUT=$ROOT/manu_src/model_outputs/CDs_and_Vinyl/embedding/qwen3emb06b_history_plus_glm52_non_target_cot_input_time_title_rating_store_categories_desc256_details256_v1_bs64_ga1_lr2e5_ep5_len4096_seed42
elif [[ "$COT_MASK_PROB" == "0.5" ]]; then
  RUN_VARIANT=history_plus_cot_whole_cot_mask_p0p5
  DEFAULT_OUT=$ROOT/manu_src/model_outputs/CDs_and_Vinyl/embedding/qwen3emb06b_history_plus_glm52_non_target_cot_whole_cot_mask_p0p5_input_time_title_rating_store_categories_desc256_details256_v1_bs64_ga1_lr2e5_ep5_len4096_seed42
else
  echo "本组实验只允许 COT_MASK_PROB=0.0 或 0.5。" >&2
  exit 1
fi
OUT=${OUT:-$DEFAULT_OUT}

print_param() {
  local name=$1 value=$2 description=$3
  printf '  %-34s %s\n      %s\n' "$name=$value" "" "$description"
}

for path in "$ROOT/AGENTS.md" "$VENV/bin/python" "$MODEL" "$TRAIN_FILE"; do
  if [[ ! -e "$path" ]]; then
    echo "缺少 embedding 训练依赖：$path" >&2
    exit 1
  fi
done
if [[ "$SEED" != "42" ]]; then
  echo "项目随机种子必须为 42。" >&2
  exit 1
fi
ROWS=$(wc -l < "$TRAIN_FILE" | tr -d ' ')
if [[ "$ROWS" != "10722" ]]; then
  echo "embedding 训练数据应为 10722 条，当前为 $ROWS。" >&2
  exit 1
fi

echo "Qwen3-Embedding-0.6B 正式训练参数："
print_param RUN_VARIANT "$RUN_VARIANT" "本次实验变量。"
print_param MODEL "$MODEL" "Qwen3-Embedding-0.6B 基座。"
print_param TRAIN_FILE "$TRAIN_FILE" "history、完整 API CoT 与 full positive 对齐后的 10722 条训练 pair。"
print_param OUT "$OUT" "每轮 embedding checkpoint 与训练指标输出目录。"
print_param COT_MASK_PROB "$COT_MASK_PROB" "按样本删除整段 CoT 的概率；0.5 时每个 epoch 重新采样。"
print_param QUERY_TRUNCATION drop_oldest_description_then_details_then_item "超长 query 先删最旧物品 Description，再删 Details，再删整条历史；CoT 保持完整。"
print_param POSITIVE_TRUNCATION false "positive 使用 full target text，超过上限会报错。"
print_param BATCH_SIZE "$BATCH_SIZE" "单卡 micro batch；两组 embedding 使用相同值。"
print_param GRAD_ACCUM "$GRAD_ACCUM" "梯度累积步数；两组实验保持一致。"
print_param MAX_LENGTH "$MAX_LENGTH" "query 最大 token 数，字段级裁剪后不超过 4096。"
print_param EPOCHS "$EPOCHS" "完整训练轮数；每轮保存一次 checkpoint。"
print_param LEARNING_RATE "$LEARNING_RATE" "AdamW 峰值学习率。"
print_param SEED "$SEED" "模型初始化、DataLoader shuffle 和 CoT mask 随机种子。"
print_param TEST_EVAL skipped "按用户要求，本轮两个 embedding 暂不读取或评测 test。"
print_param CONFIRM_RUN "$CONFIRM_RUN" "必须为 1 才执行。"

if [[ "$CONFIRM_RUN" != "1" ]]; then
  echo "未执行。"
  exit 0
fi

source /opt/miniforge3/bin/activate "$VENV"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p "$OUT"

"$VENV/bin/python" manu_src/scripts/models/train_embedding.py \
  --model "$MODEL" \
  --train-file "$TRAIN_FILE" \
  --output-dir "$OUT" \
  --max-length "$MAX_LENGTH" \
  --batch-size "$BATCH_SIZE" \
  --grad-accum "$GRAD_ACCUM" \
  --epochs "$EPOCHS" \
  --learning-rate "$LEARNING_RATE" \
  --weight-decay 0.01 \
  --warmup-ratio 0.03 \
  --temperature 0.05 \
  --cot-mask-prob "$COT_MASK_PROB" \
  --seed "$SEED" \
  --attn-implementation flash_attention_2 \
  --skip-test-eval
