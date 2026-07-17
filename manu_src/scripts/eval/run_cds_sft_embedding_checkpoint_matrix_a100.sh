#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/user/aaai_pro}
VENV=${VENV:-/home/user/.conda/envs/aaai_pro}
FULL_OUT=${FULL_OUT:-$ROOT/manu_src/model_outputs/CDs_and_Vinyl/embedding/qwen3emb06b_history_plus_glm52_non_target_cot_input_time_title_rating_store_categories_desc256_details256_v1_bs128_ga1_lr2e5_ep5_len4096_seed42}
MASK_OUT=${MASK_OUT:-$ROOT/manu_src/model_outputs/CDs_and_Vinyl/embedding/qwen3emb06b_history_plus_glm52_non_target_cot_whole_cot_mask_p0p5_input_time_title_rating_store_categories_desc256_details256_v1_bs128_ga1_lr2e5_ep5_len4096_seed42}
SFT_EVAL_ROOT=${SFT_EVAL_ROOT:-$ROOT/manu_src/model_outputs/CDs_and_Vinyl/eval/qwen25_3b_lora_history_only_next_item_feature_cot_time_title_rating_store_categories_desc256_details256_v1}
ITEM_INFO=${ITEM_INFO:-$ROOT/manu_src/datas/CDs_and_Vinyl/arrow_to_jsonls/item_info.jsonl}
OUTPUT_ROOT=${OUTPUT_ROOT:-$SFT_EVAL_ROOT/embedding_checkpoint_matrix}

MAX_LENGTH=${MAX_LENGTH:-4096}
ITEM_BATCH_SIZE=${ITEM_BATCH_SIZE:-128}
QUERY_BATCH_SIZE=${QUERY_BATCH_SIZE:-64}
SCORE_BATCH_SIZE=${SCORE_BATCH_SIZE:-128}
SEED=${SEED:-42}
CONFIRM_RUN=${CONFIRM_RUN:-0}

print_param() {
  local name=$1 value=$2 description=$3
  printf '  %-34s %s\n      %s\n' "$name=$value" "" "$description"
}

if [[ "$SEED" != "42" ]]; then
  echo "项目随机种子必须为 42。" >&2
  exit 1
fi
for path in "$ROOT/AGENTS.md" "$VENV/bin/python" "$FULL_OUT" "$MASK_OUT" "$SFT_EVAL_ROOT" "$ITEM_INFO"; do
  if [[ ! -e "$path" ]]; then
    echo "缺少矩阵评测依赖：$path" >&2
    exit 1
  fi
done

full_checkpoints=$(find "$FULL_OUT" -maxdepth 1 -type d -name 'checkpoint-epoch-*' | wc -l | tr -d ' ')
mask_checkpoints=$(find "$MASK_OUT" -maxdepth 1 -type d -name 'checkpoint-epoch-*' | wc -l | tr -d ' ')
sft_outputs=$(find "$SFT_EVAL_ROOT" -mindepth 2 -maxdepth 2 -type f -name 'test_generated_cot.jsonl' | wc -l | tr -d ' ')
if [[ "$full_checkpoints" != "5" || "$mask_checkpoints" != "5" || "$sft_outputs" != "5" ]]; then
  echo "评测输入尚未完整：full embedding=$full_checkpoints/5，mask embedding=$mask_checkpoints/5，SFT CoT=$sft_outputs/5。" >&2
  exit 1
fi

echo "CDs_and_Vinyl embedding × SFT checkpoint 矩阵评测参数："
print_param EMBEDDING_FULL "$FULL_OUT/checkpoint-epoch-{01..05}" "完整 CoT embedding 的每轮 checkpoint。"
print_param EMBEDDING_MASK "$MASK_OUT/checkpoint-epoch-{01..05}" "整段 CoT mask p=0.5 embedding 的每轮 checkpoint。"
print_param SFT_TEST_COT "$SFT_EVAL_ROOT/checkpoint-{336,672,1008,1344,1680}/test_generated_cot.jsonl" "5 个 SFT checkpoint 各自生成的 1341 条 history+CoT test query。"
print_param COMBINATIONS "2×5×5=50" "每个 embedding checkpoint 分别评测全部 5 份 SFT 结果。"
print_param CANDIDATES "12000 full items" "使用完整候选库；每个 embedding checkpoint 只编码一次候选并复用于 5 份 SFT query。"
print_param SEEN_ITEM_MASK "enabled" "排序前屏蔽历史已交互物品，同时保留监督 target。"
print_param METRICS "MRR, mean/median rank, HR/NDCG@5,10,20" "所有组合使用相同检索指标。"
print_param MAX_LENGTH "$MAX_LENGTH" "query 与候选编码上限；候选文本禁止截断，query 沿用字段级截断口径。"
print_param ITEM_BATCH_SIZE "$ITEM_BATCH_SIZE" "候选物品 embedding 编码批量。"
print_param QUERY_BATCH_SIZE "$QUERY_BATCH_SIZE" "SFT test query embedding 编码批量。"
print_param SCORE_BATCH_SIZE "$SCORE_BATCH_SIZE" "与 12000 个候选计算排名时的 query 批量。"
print_param OUTPUT_ROOT "$OUTPUT_ROOT" "按 embedding 版本、embedding epoch、SFT checkpoint 三层保存指标和 ranks。"
print_param RESUME_POLICY "validated combination artifacts" "已有指标必须匹配 scorer、test 文件和 1341 条 ranks 才会跳过。"
print_param SEED "$SEED" "模型加载、编码和评测随机种子。"
print_param CONFIRM_RUN "$CONFIRM_RUN" "必须为 1 才执行长耗时评测。"

if [[ "$CONFIRM_RUN" != "1" ]]; then
  echo "未执行。"
  exit 0
fi

source /opt/miniforge3/bin/activate "$VENV"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
mkdir -p "$OUTPUT_ROOT"

"$VENV/bin/python" manu_src/scripts/eval/evaluate_sft_embedding_checkpoint_matrix.py \
  --embedding-run "history_plus_cot=$FULL_OUT" \
  --embedding-run "whole_cot_mask_p0p5=$MASK_OUT" \
  --sft-eval-root "$SFT_EVAL_ROOT" \
  --item-info "$ITEM_INFO" \
  --output-root "$OUTPUT_ROOT" \
  --expected-embedding-checkpoints 5 \
  --expected-sft-checkpoints 5 \
  --expected-test-rows 1341 \
  --max-length "$MAX_LENGTH" \
  --item-batch-size "$ITEM_BATCH_SIZE" \
  --query-batch-size "$QUERY_BATCH_SIZE" \
  --score-batch-size "$SCORE_BATCH_SIZE" \
  --ks 5,10,20 \
  --seed "$SEED" \
  --attn-implementation flash_attention_2
