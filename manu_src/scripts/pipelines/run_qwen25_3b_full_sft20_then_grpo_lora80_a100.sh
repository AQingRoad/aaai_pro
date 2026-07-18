#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/user/aaai_pro}
SFT_SCRIPT=${SFT_SCRIPT:-$ROOT/manu_src/scripts/train/run_qwen25_3b_full_sft20_a100.sh}
GRPO_SCRIPT=${GRPO_SCRIPT:-$ROOT/manu_src/scripts/train/run_qwen25_3b_grpo_lora_from_full_sft20_a100.sh}

SFT_OUT=${SFT_OUT:-$ROOT/manu_src/model_outputs/CDs_and_Vinyl/sft/qwen25_3b_full_sft20_disjoint_time_title_rating_store_categories_desc256_details256_v1_bs16_ga1_lr2e5_ep1_len4096_seed42}
GRPO_OUT=${GRPO_OUT:-$ROOT/manu_src/model_outputs/CDs_and_Vinyl/grpo/qwen25_3b_fullsft20_bs16_lr2e5_grpolora80_cottrained_logsoftmaxsim_w0p8_ndcg100_w0p2_g4_genbs16_bs4_ga1_vllmsleep1_vllm0p10_lr2e5_ep1_vllmlen4608_clen512_seed42}

SFT_BATCH_SIZE=${SFT_BATCH_SIZE:-16}
SFT_LEARNING_RATE=${SFT_LEARNING_RATE:-2e-5}
GRPO_BATCH_SIZE=${GRPO_BATCH_SIZE:-4}
GRPO_GENERATION_BATCH_SIZE=${GRPO_GENERATION_BATCH_SIZE:-16}
GRPO_LEARNING_RATE=${GRPO_LEARNING_RATE:-2e-5}
SEED=${SEED:-42}
ALLOW_EXISTING_OUTPUT=${ALLOW_EXISTING_OUTPUT:-0}
CONFIRM_RUN=${CONFIRM_RUN:-0}

print_param() {
  local name=$1 value=$2 description=$3
  printf '  %-38s %s\n      %s\n' "$name=$value" "" "$description"
}

for dependency in "$ROOT/AGENTS.md" "$SFT_SCRIPT" "$GRPO_SCRIPT"; do
  if [[ ! -e "$dependency" ]]; then
    echo "缺少调度依赖：$dependency" >&2
    exit 1
  fi
done
if [[ "$SEED" != "42" ]]; then
  echo "项目随机种子必须为 42。" >&2
  exit 1
fi
if [[ "$ALLOW_EXISTING_OUTPUT" != "0" && "$ALLOW_EXISTING_OUTPUT" != "1" ]]; then
  echo "ALLOW_EXISTING_OUTPUT 只能为 0 或 1。" >&2
  exit 1
fi

echo "Qwen2.5-3B 全参数 SFT 20% → LoRA GRPO 80% 顺序调度参数："
print_param STAGE_1 full_parameter_sft "先更新 Qwen2.5-3B 全部参数，训练数据为固定前 20% 的 2,144 条样本。"
print_param STAGE_2 lora_grpo "SFT 完成后，以完整 SFT checkpoint 为基座，新建 GRPO LoRA 并训练后 80% 的 8,578 条样本。"
print_param SPLIT disjoint_example20_80 "两阶段按 example_id 分割，训练样本交集为 0；随机种子固定为 42。"
print_param INPUT_SCHEMA time_title_rating_store_categories_desc256_details256_v1 "SFT、GRPO、reward embedding 和后续评测统一使用该 history 口径。"
print_param SFT_BATCH_SIZE "$SFT_BATCH_SIZE" "全参数 SFT 单卡 micro batch；梯度累积固定为 1。"
print_param SFT_LEARNING_RATE "$SFT_LEARNING_RATE" "全参数 SFT AdamW 峰值学习率。"
print_param SFT_EPOCHS 1 "前 20% 数据训练 1 个完整 epoch。"
print_param SFT_MAX_LENGTH 4096 "SFT chat 序列最大 token 数，采用左截断。"
print_param GRPO_BATCH_SIZE "$GRPO_BATCH_SIZE" "每次反向传播 4 条 completion，即 1 个完整四候选组。"
print_param GRPO_GENERATION_BATCH_SIZE "$GRPO_GENERATION_BATCH_SIZE" "每轮 rollout 生成 16 条 completion，即 4 个四候选组。"
print_param GRPO_LEARNING_RATE "$GRPO_LEARNING_RATE" "新建 GRPO LoRA 的 AdamW 峰值学习率。"
print_param GRPO_EPOCHS 1 "后 80% 数据训练 1 个完整 epoch。"
print_param GRPO_REWARD "0.8 similarity + 0.2 NDCG@100" "两项分别做组内 z-score；不加入格式、margin 或 rubric 奖励。"
print_param GRPO_CONTEXT "4096 + 512 = 4608" "policy prompt 上限 4096 tokens，completion 上限 512 tokens。"
print_param GRPO_REFERENCE full_sft_model "关闭 GRPO adapter 后即为完整 SFT policy，KL 系数为 0.04。"
print_param SFT_OUT "$SFT_OUT" "完整 SFT checkpoint 输出根目录。"
print_param GRPO_OUT "$GRPO_OUT" "GRPO LoRA checkpoint 与 reward 日志输出根目录。"
print_param ALLOW_EXISTING_OUTPUT "$ALLOW_EXISTING_OUTPUT" "默认为 0，阻止误混旧 checkpoint；断点处理时需显式设为 1。"
print_param SEED "$SEED" "数据、模型、DataLoader 与 rollout 随机种子固定为 42。"
print_param CONFIRM_RUN "$CONFIRM_RUN" "必须为 1 才顺序启动两阶段长任务。"

if [[ "$CONFIRM_RUN" != "1" ]]; then
  echo "未启动训练。"
  exit 0
fi

if [[ "$ALLOW_EXISTING_OUTPUT" == "0" ]]; then
  for output_dir in "$SFT_OUT" "$GRPO_OUT"; do
    if [[ -e "$output_dir" ]]; then
      echo "输出目录已经存在，拒绝覆盖或混入旧结果：$output_dir" >&2
      exit 1
    fi
  done
fi

cd "$ROOT"
mkdir -p "$(dirname "$SFT_OUT")" "$(dirname "$GRPO_OUT")"

echo "[$(date -Iseconds)] 启动阶段 1：前 20% 数据全参数 SFT。"
ROOT="$ROOT" \
OUT="$SFT_OUT" \
BATCH_SIZE="$SFT_BATCH_SIZE" \
LEARNING_RATE="$SFT_LEARNING_RATE" \
SEED="$SEED" \
CONFIRM_RUN=1 \
bash "$SFT_SCRIPT"

latest_sft_run=$(find "$SFT_OUT" -mindepth 1 -maxdepth 1 -type d -name 'v*' -printf '%T@ %p\n' \
  | sort -nr | head -n 1 | cut -d' ' -f2-)
if [[ -z "$latest_sft_run" ]]; then
  echo "全参数 SFT 完成后没有找到 Swift run 目录：$SFT_OUT" >&2
  exit 1
fi
sft_checkpoint=$(find "$latest_sft_run" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint-*' \
  | sort -V | tail -n 1)
if [[ -z "$sft_checkpoint" || ! -e "$sft_checkpoint/config.json" ]]; then
  echo "没有找到可加载的完整 SFT checkpoint：$latest_sft_run" >&2
  exit 1
fi
if [[ -e "$sft_checkpoint/adapter_config.json" ]]; then
  echo "阶段 1 产物仍是 LoRA adapter，停止调度：$sft_checkpoint" >&2
  exit 1
fi

printf '%s\n' "$sft_checkpoint" > "$SFT_OUT/selected_full_sft_checkpoint.txt"
echo "[$(date -Iseconds)] 阶段 1 完成，GRPO 基座：$sft_checkpoint"
echo "[$(date -Iseconds)] 启动阶段 2：后 80% 数据 LoRA GRPO。"

ROOT="$ROOT" \
MODEL="$sft_checkpoint" \
OUT="$GRPO_OUT" \
BATCH_SIZE="$GRPO_BATCH_SIZE" \
GENERATION_BATCH_SIZE="$GRPO_GENERATION_BATCH_SIZE" \
LEARNING_RATE="$GRPO_LEARNING_RATE" \
SEED="$SEED" \
CONFIRM_RUN=1 \
bash "$GRPO_SCRIPT"

echo "[$(date -Iseconds)] 两阶段训练全部完成。"
