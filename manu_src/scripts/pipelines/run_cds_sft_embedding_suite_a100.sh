#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/user/aaai_pro}
VENV=${VENV:-/home/user/.conda/envs/aaai_pro}
CONFIRM_RUN=${CONFIRM_RUN:-0}
SEED=${SEED:-42}

INPUT_DIR=$ROOT/manu_src/datas/CDs_and_Vinyl/train_datas/time_title_rating_store_categories_desc256_details256_v1.0
COT_DIR=$ROOT/manu_src/datas/CDs_and_Vinyl/cot/api/history_only_next_item_feature_cot__input_time_title_rating_store_categories_desc256_details256_v1
COT_FILE=$COT_DIR/cot_non_target_glm52_train_full_seed42_temp1.jsonl
SFT_DATA=$ROOT/manu_src/datas/CDs_and_Vinyl/sft/history_only_next_item_feature_cot__input_time_title_rating_store_categories_desc256_details256_v1/sft_messages_full_seed42.jsonl
EMBED_DATA=$ROOT/manu_src/datas/CDs_and_Vinyl/embedding/history_plus_non_target_cot__input_time_title_rating_store_categories_desc256_details256_v1/train.jsonl

SFT_OUT=$ROOT/manu_src/model_outputs/CDs_and_Vinyl/sft/qwen25_3b_lora_history_only_next_item_feature_cot_time_title_rating_store_categories_desc256_details256_v1_bs32_ga1_lr2e5_ep5_len4096_seed42
EMBED_FULL_OUT=$ROOT/manu_src/model_outputs/CDs_and_Vinyl/embedding/qwen3emb06b_history_plus_glm52_non_target_cot_input_time_title_rating_store_categories_desc256_details256_v1_bs64_ga1_lr2e5_ep5_len4096_seed42
EMBED_MASK_OUT=$ROOT/manu_src/model_outputs/CDs_and_Vinyl/embedding/qwen3emb06b_history_plus_glm52_non_target_cot_whole_cot_mask_p0p5_input_time_title_rating_store_categories_desc256_details256_v1_bs64_ga1_lr2e5_ep5_len4096_seed42
EVAL_ROOT=$ROOT/manu_src/model_outputs/CDs_and_Vinyl/eval/qwen25_3b_lora_history_only_next_item_feature_cot_time_title_rating_store_categories_desc256_details256_v1
STATE_DIR=$ROOT/manu_src/model_outputs/CDs_and_Vinyl/scheduler_state/cds_sft_embedding_suite_seed42

print_param() {
  local name=$1 value=$2 description=$3
  printf '  %-34s %s\n      %s\n' "$name=$value" "" "$description"
}

if [[ "$SEED" != "42" ]]; then
  echo "项目随机种子必须为 42。" >&2
  exit 1
fi
for path in "$ROOT/AGENTS.md" "$VENV/bin/python" "$INPUT_DIR/train.jsonl" "$INPUT_DIR/test.jsonl" "$COT_FILE"; do
  if [[ ! -e "$path" ]]; then
    echo "缺少调度依赖：$path" >&2
    exit 1
  fi
done

echo "CDs_and_Vinyl 统一实验调度参数："
print_param STAGE_1 "SFT LoRA" "Qwen2.5-3B-Instruct，batch=32，grad_accum=1，max_length=4096 左截断，lr=2e-5，5 epoch。"
print_param STAGE_2 "vLLM test CoT" "对 5 个 SFT checkpoint 逐一生成 1341 条 test CoT；prompt 与 API 训练数据一致。"
print_param STAGE_3 "embedding history+CoT" "Qwen3-Embedding-0.6B，batch=64，grad_accum=1，max_length=4096，lr=2e-5，5 epoch。"
print_param STAGE_4 "embedding CoT mask p=0.5" "与 stage 3 参数一致，每个 epoch 按样本重新随机删除完整 CoT。"
print_param STAGE_5 "5 checkpoint retrieval eval" "固定使用 stage 3 的 checkpoint-epoch-05，在 12000 个候选上计算 HR/NDCG@5,10,20。"
print_param INPUT_SCHEMA "time_title_rating_store_categories_desc256_details256_v1" "history 保留时间、标题、评分、store、类别、Description256 和 Details256。"
print_param SFT_DATA "$SFT_DATA" "SFT messages 数据；缺失时由 10722 条 API CoT 重建。"
print_param EMBED_DATA "$EMBED_DATA" "结构化 history、完整 CoT、full positive 训练 pair；缺失时自动重建。"
print_param SFT_OUT "$SFT_OUT" "SFT LoRA checkpoint 目录。"
print_param EMBED_FULL_OUT "$EMBED_FULL_OUT" "不 mask CoT 的 embedding checkpoint 目录。"
print_param EMBED_MASK_OUT "$EMBED_MASK_OUT" "整段 CoT mask p=0.5 的 embedding checkpoint 目录。"
print_param EVAL_ROOT "$EVAL_ROOT" "5 组生成审计、rank 与检索指标目录。"
print_param RESUME_POLICY "stage markers + artifact audit" "已完整产出的阶段自动跳过；不完整阶段停止并保留现场。"
print_param SEED "$SEED" "所有数据、训练、mask、生成和评测随机种子。"
print_param CONFIRM_RUN "$CONFIRM_RUN" "必须为 1 才启动长任务。"

if [[ "$CONFIRM_RUN" != "1" ]]; then
  echo "未启动。"
  exit 0
fi

source /opt/miniforge3/bin/activate "$VENV"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
mkdir -p "$STATE_DIR"

checkpoint_count() {
  local directory=$1
  find "$directory" -type d -name 'checkpoint-*' 2>/dev/null | wc -l | tr -d ' '
}

mark_if_complete() {
  local stage=$1 directory=$2 expected=$3
  if [[ ! -f "$STATE_DIR/$stage.done" && -d "$directory" ]]; then
    local count
    count=$(checkpoint_count "$directory")
    if [[ "$count" == "$expected" ]]; then
      touch "$STATE_DIR/$stage.done"
    elif [[ "$count" != "0" ]]; then
      echo "$stage 存在 $count/$expected 个 checkpoint；为避免覆盖未完成训练，调度停止。" >&2
      exit 1
    fi
  fi
}

if [[ ! -f "$STATE_DIR/prepare_data.done" ]]; then
  "$VENV/bin/python" manu_src/scripts/cot/build_non_target_cot_sft_dataset.py \
    --input "$COT_FILE" \
    --output "$SFT_DATA" \
    --item-type "CD or vinyl release" \
    --language en \
    --seed "$SEED"
  "$VENV/bin/python" manu_src/scripts/cot/build_history_cot_embedding_dataset.py \
    --pairs "$INPUT_DIR/train.jsonl" \
    --cot "$COT_FILE" \
    --output "$EMBED_DATA" \
    --seed "$SEED"
  [[ $(wc -l < "$SFT_DATA" | tr -d ' ') == 10722 ]]
  [[ $(wc -l < "$EMBED_DATA" | tr -d ' ') == 10722 ]]
  touch "$STATE_DIR/prepare_data.done"
fi

mark_if_complete sft "$SFT_OUT" 5
if [[ ! -f "$STATE_DIR/sft.done" ]]; then
  CONFIRM_RUN=1 ROOT="$ROOT" OUT="$SFT_OUT" \
    bash manu_src/scripts/train/run_qwen25_3b_non_target_cot_sft_a100.sh
  [[ $(checkpoint_count "$SFT_OUT") == 5 ]]
  touch "$STATE_DIR/sft.done"
fi

if [[ ! -f "$STATE_DIR/vllm_generate.done" ]]; then
  CONFIRM_RUN=1 MODE=generate ROOT="$ROOT" SFT_OUT="$SFT_OUT" EVAL_ROOT="$EVAL_ROOT" \
    bash manu_src/scripts/eval/run_qwen25_sft_checkpoints_vllm_eval_a100.sh
  [[ $(find "$EVAL_ROOT" -name test_generated_cot.jsonl -type f | wc -l | tr -d ' ') == 5 ]]
  touch "$STATE_DIR/vllm_generate.done"
fi

mark_if_complete embedding_full "$EMBED_FULL_OUT" 5
if [[ ! -f "$STATE_DIR/embedding_full.done" ]]; then
  CONFIRM_RUN=1 ROOT="$ROOT" TRAIN_FILE="$EMBED_DATA" OUT="$EMBED_FULL_OUT" COT_MASK_PROB=0.0 \
    bash manu_src/scripts/train/run_cds_history_cot_embedding_a100.sh
  [[ $(checkpoint_count "$EMBED_FULL_OUT") == 5 ]]
  touch "$STATE_DIR/embedding_full.done"
fi

mark_if_complete embedding_mask "$EMBED_MASK_OUT" 5
if [[ ! -f "$STATE_DIR/embedding_mask.done" ]]; then
  CONFIRM_RUN=1 ROOT="$ROOT" TRAIN_FILE="$EMBED_DATA" OUT="$EMBED_MASK_OUT" COT_MASK_PROB=0.5 \
    bash manu_src/scripts/train/run_cds_history_cot_embedding_a100.sh
  [[ $(checkpoint_count "$EMBED_MASK_OUT") == 5 ]]
  touch "$STATE_DIR/embedding_mask.done"
fi

if [[ ! -f "$STATE_DIR/sft_eval.done" ]]; then
  CONFIRM_RUN=1 MODE=evaluate ROOT="$ROOT" SFT_OUT="$SFT_OUT" EVAL_ROOT="$EVAL_ROOT" \
    EMBEDDING_SCORER="$EMBED_FULL_OUT/checkpoint-epoch-05" \
    bash manu_src/scripts/eval/run_qwen25_sft_checkpoints_vllm_eval_a100.sh
  [[ -s "$EVAL_ROOT/all_checkpoint_metrics.json" ]]
  touch "$STATE_DIR/sft_eval.done"
fi

echo "统一实验调度全部完成：$EVAL_ROOT/all_checkpoint_metrics.json"
