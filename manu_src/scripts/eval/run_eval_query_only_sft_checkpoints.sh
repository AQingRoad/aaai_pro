#!/usr/bin/env bash
set -euo pipefail

# SFT 代码和 checkpoint 位于独立 manu_src 仓库；检索模型及旧基准位于正式仓库。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MANU_ROOT=${MANU_ROOT:-"$(cd "$SCRIPT_DIR/../.." && pwd)"}
PROJECT_ROOT=${PROJECT_ROOT:-"$(cd "$MANU_ROOT/.." && pwd)"}
RETRIEVAL_ROOT=${RETRIEVAL_ROOT:-/home/user/aaai_pro}
VENV=${VENV:-/home/user/.conda/envs/aaai_pro}
PYTHON_BIN=${PYTHON_BIN:-$VENV/bin/python}

BASE_MODEL=${BASE_MODEL:-/home/user/models_hf/Qwen2.5-3B-Instruct}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-$MANU_ROOT/model_outputs/CDs_and_Vinyl/sft/qwen25_3b_lora_glm52_v1p12_train_head20_metadata_rich_len4096_bs16_ga1_paddingfree_lr1e5_ep3_seed42/v0-20260714-104315}
CHECKPOINT_STEPS=${CHECKPOINT_STEPS:-"134 268 402"}

# 生成阶段读取和 SFT 训练一致的富字段 QUERY；检索阶段读取严格 title_store_categories QUERY。
GENERATION_INPUT=${GENERATION_INPUT:-$MANU_ROOT/datas/CDs_and_Vinyl/eval_inputs/test_time_title_rating_store_categories_desc256_details256_seed42.jsonl}
RETRIEVAL_INPUT=${RETRIEVAL_INPUT:-$RETRIEVAL_ROOT/ablantion/datas/processed_datas/cds_query_ablation_test/cds_test_query_title_store_categories.jsonl}
ITEM_INFO=${ITEM_INFO:-$RETRIEVAL_ROOT/github_artifacts/CDs_and_Vinyl/rrec_eval/item_info.jsonl}
EMBEDDING_MODEL=${EMBEDDING_MODEL:-$RETRIEVAL_ROOT/checkpoints/rrec_amazon_CDs_and_Vinyl/cds_query_ablation_cot_lr2e-5_epoch5/title_store_categories_no_trunc_plus_cot/checkpoint-83}
REFERENCE_RESULT=${REFERENCE_RESULT:-$RETRIEVAL_ROOT/outputs/rrec_amazon/eval/CDs_and_Vinyl/cds_query_ablation_cot_lr2e-5_epoch5/title_store_categories_no_trunc_plus_cot/checkpoint-83_test.json}

RUN_NAME=${RUN_NAME:-qwen25_3b_lora_glm52_v1p12_checkpoints134_268_402_test_history_plus_cot_seed42}
OUT_DIR=${OUT_DIR:-$MANU_ROOT/eval_results/CDs_and_Vinyl/sft/$RUN_NAME}
GENERATION_BATCH_SIZE=${GENERATION_BATCH_SIZE:-32}
EMBEDDING_BATCH_SIZE=${EMBEDDING_BATCH_SIZE:-32}
MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS:-3328}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-768}
EMBEDDING_MAX_LENGTH=${EMBEDDING_MAX_LENGTH:-4096}
SEED=${SEED:-42}
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
  printf '  %-34s\n      %s\n' "$name=$value" "$description"
}

require_path "Python 环境" "$PYTHON_BIN"
require_path "生成测试集" "$GENERATION_INPUT"
require_path "严格检索测试集" "$RETRIEVAL_INPUT"
require_path "全量 item_info" "$ITEM_INFO"
require_path "Embedding checkpoint" "$EMBEDDING_MODEL"
require_path "GLM teacher CoT 参考结果" "$REFERENCE_RESULT"
for step in $CHECKPOINT_STEPS; do
  require_path "SFT checkpoint-$step" "$CHECKPOINT_ROOT/checkpoint-$step"
done

if [[ "$SEED" != "42" ]]; then
  echo "项目随机种子固定为 42" >&2
  exit 1
fi
if (( MAX_PROMPT_TOKENS + MAX_NEW_TOKENS != 4096 )); then
  echo "输入与生成 token 预算之和必须为 4096" >&2
  exit 1
fi

echo "即将评测三个 QUERY-only SFT checkpoint："
print_param "CHECKPOINT_STEPS" "$CHECKPOINT_STEPS" "依次评测第 1、2、3 个 epoch 保存的 LoRA adapter。"
print_param "GENERATION_INPUT" "$GENERATION_INPUT" "CoT 生成只读取带评分的富字段 test QUERY，共 1341 条。"
print_param "RETRIEVAL_INPUT" "$RETRIEVAL_INPUT" "排序使用严格 title_store_categories 历史 query。"
print_param "GENERATION_BATCH_SIZE" "$GENERATION_BATCH_SIZE" "Qwen2.5 LoRA 贪心生成的单卡 batch size。"
print_param "EMBEDDING_BATCH_SIZE" "$EMBEDDING_BATCH_SIZE" "Qwen3 Embedding 编码 item 和 query 的 batch size。"
print_param "MAX_PROMPT_TOKENS" "$MAX_PROMPT_TOKENS" "输入最多保留的 tokens；超长时逐步压缩最早历史。"
print_param "MAX_NEW_TOKENS" "$MAX_NEW_TOKENS" "完整 <think>/<answer> 最多生成的 tokens。"
print_param "EMBEDDING_MAX_LENGTH" "$EMBEDDING_MAX_LENGTH" "拼接 query 和 tagged CoT 后的检索编码长度。"
print_param "EMBEDDING_MODEL" "$EMBEDDING_MODEL" "使用既有 title_store_categories+CoT 检索模型 checkpoint-83。"
print_param "ITEM_INFO" "$ITEM_INFO" "目标 item 在 12000 个候选中排序。"
print_param "QUERY_MODE" "history_plus_cot" "严格 query 后拼接 Recommendation reasoning 和 tagged CoT。"
print_param "SEEN_ITEM_MASK" "true" "屏蔽用户历史中已经交互的候选物品。"
print_param "SEED" "$SEED" "固定测试顺序和模型生成随机状态。"
print_param "OUT_DIR" "$OUT_DIR" "保存三组 CoT、审计、逐样本 rank 和汇总指标。"
print_param "CONFIRM_RUN" "$CONFIRM_RUN" "值为 1 时执行完整评测。"

if [[ "$CONFIRM_RUN" != "1" ]]; then
  echo "参数已打印，评测尚未启动。"
  exit 0
fi

mkdir -p "$OUT_DIR"
export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$PROJECT_ROOT:$RETRIEVAL_ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

for step in $CHECKPOINT_STEPS; do
  checkpoint="checkpoint-$step"
  run_dir="$OUT_DIR/$checkpoint"
  mkdir -p "$run_dir"
  echo "开始生成 $checkpoint 的测试集 CoT。"
  "$PYTHON_BIN" "$SCRIPT_DIR/generate_query_only_sft_cot.py" \
    --generation-input "$GENERATION_INPUT" \
    --retrieval-input "$RETRIEVAL_INPUT" \
    --output "$run_dir/generated_cot.jsonl" \
    --audit-output "$run_dir/generated_cot.audit.json" \
    --base-model "$BASE_MODEL" \
    --adapter "$CHECKPOINT_ROOT/$checkpoint" \
    --batch-size "$GENERATION_BATCH_SIZE" \
    --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --seed "$SEED"

  echo "开始计算 $checkpoint 的 12000-item 排名。"
  "$PYTHON_BIN" "$RETRIEVAL_ROOT/scripts/eval/evaluate_rrec_jsonl_fullset.py" \
    --examples "$run_dir/generated_cot.jsonl" \
    --item-info "$ITEM_INFO" \
    --category CDs_and_Vinyl \
    --split test \
    --query-mode history_plus_cot \
    --cot-text-mode tagged \
    --require-cot \
    --ks 5,10,20 \
    --scorer qwen3_embedding \
    --embedding-model "$EMBEDDING_MODEL" \
    --embedding-max-length "$EMBEDDING_MAX_LENGTH" \
    --embedding-batch-size "$EMBEDDING_BATCH_SIZE" \
    --max-item-chars 0 \
    --torch-dtype bfloat16 \
    --device cuda:0 \
    --mask-history-items \
    --mask-pad-item \
    --output "$run_dir/embedding_eval.json" \
    --ranks-output "$run_dir/ranks.jsonl"
done

"$PYTHON_BIN" "$SCRIPT_DIR/summarize_sft_checkpoint_eval.py" \
  --output-dir "$OUT_DIR" \
  --reference "$REFERENCE_RESULT" \
  --checkpoints "$(tr ' ' ',' <<<"$CHECKPOINT_STEPS")" \
  --output "$OUT_DIR/summary.json"

echo "三个 checkpoint 评测完成：$OUT_DIR/summary.json"
