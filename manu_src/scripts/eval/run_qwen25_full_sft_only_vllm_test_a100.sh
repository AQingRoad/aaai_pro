#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/user/aaai_pro}
VENV=${VENV:-/home/user/.conda/envs/aaai_pro}
SFT_MODEL=${SFT_MODEL:-$ROOT/manu_src/model_outputs/CDs_and_Vinyl/sft/qwen25_3b_full_sft20_disjoint_time_title_rating_store_categories_desc256_details256_v1_bs16_ga1_lr2e5_ep1_len4096_seed42/v0-20260718-080839/checkpoint-134}
TEST_FILE=${TEST_FILE:-$ROOT/manu_src/datas/CDs_and_Vinyl/train_datas/time_title_rating_store_categories_desc256_details256_v1.0/test.jsonl}
ITEM_INFO=${ITEM_INFO:-$ROOT/manu_src/datas/CDs_and_Vinyl/arrow_to_jsonls/item_info.jsonl}
EMBEDDING_SCORER=${EMBEDDING_SCORER:-$ROOT/manu_src/model_outputs/CDs_and_Vinyl/embedding/qwen3emb06b_history_plus_glm52_non_target_cot_input_time_title_rating_store_categories_desc256_details256_v1_bs128_ga1_lr2e5_ep5_len4096_seed42/checkpoint-epoch-01}
EVAL_ROOT=${EVAL_ROOT:-$ROOT/manu_src/model_outputs/CDs_and_Vinyl/eval/qwen25_3b_fullsft20_only_test_time_title_rating_store_categories_desc256_details256_v1_cottrained_epoch01_seed42}

MODE=${MODE:-all}
GENERATION_BATCH_SIZE=${GENERATION_BATCH_SIZE:-32}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_K=${TOP_K:-200}
TOP_P=${TOP_P:-1.0}
MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS:-4096}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-512}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-4608}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
KS=${KS:-5,10,20,50,100}
SEED=${SEED:-42}
EXPECTED_ROWS=${EXPECTED_ROWS:-1341}
CONFIRM_RUN=${CONFIRM_RUN:-0}

print_param() {
  local name=$1 value=$2 description=$3
  printf '  %-34s %s\n      %s\n' "$name=$value" "" "$description"
}

if [[ "$MODE" != "generate" && "$MODE" != "evaluate" && "$MODE" != "all" ]]; then
  echo "MODE 必须为 generate、evaluate 或 all。" >&2
  exit 1
fi
if [[ "$SEED" != "42" ]]; then
  echo "项目随机种子必须为 42。" >&2
  exit 1
fi
for path in "$ROOT/AGENTS.md" "$VENV/bin/python" "$SFT_MODEL" "$TEST_FILE" "$ITEM_INFO" "$EMBEDDING_SCORER"; do
  if [[ ! -e "$path" ]]; then
    echo "缺少评测依赖：$path" >&2
    exit 1
  fi
done
if [[ ! -s "$SFT_MODEL/config.json" || ! -s "$SFT_MODEL/model.safetensors.index.json" ]]; then
  echo "SFT_MODEL 不是完整的全参数 checkpoint：$SFT_MODEL" >&2
  exit 1
fi
if [[ "$(wc -l < "$TEST_FILE" | tr -d ' ')" != "$EXPECTED_ROWS" ]]; then
  echo "test 应为 $EXPECTED_ROWS 条。" >&2
  exit 1
fi

echo "Qwen2.5-3B full-SFT-only test 评测参数："
print_param MODE "$MODE" "all 表示先用 full-SFT 模型生成 CoT，再执行完整候选检索评测。"
print_param SFT_MODEL "$SFT_MODEL" "GRPO 各版本共同使用的 checkpoint-134；直接加载完整权重，不挂载 GRPO LoRA。"
print_param TEST_FILE "$TEST_FILE" "$EXPECTED_ROWS 条 test history；target 字段不进入生成 prompt。"
print_param PROMPT_TEMPLATE "general_recommendation_cot_en" "与 GRPO checkpoint 和 API 非 target CoT 评测使用同一英文提示词。"
print_param EMBEDDING_SCORER "$EMBEDDING_SCORER" "固定使用 GRPO reward 与现有结果对应的 CoT-trained embedding epoch-01。"
print_param ITEM_INFO "$ITEM_INFO" "12000 个真实候选；沿用训练 positive formatter，并屏蔽历史物品。"
print_param COMPLETION_POLICY "one_shot_raw_completion" "每条 history 采样一次；非空但格式不完整的原始 completion 仍进入检索，并单独审计格式。"
print_param GENERATION_BATCH_SIZE "$GENERATION_BATCH_SIZE" "vLLM 单批生成 32 条 query。"
print_param TEMPERATURE "$TEMPERATURE" "与当前 GRPO checkpoint test 评测一致。"
print_param TOP_K "$TOP_K" "每步保留概率最高的 200 个 token。"
print_param TOP_P "$TOP_P" "不额外裁剪 nucleus 概率质量。"
print_param MAX_PROMPT_TOKENS "$MAX_PROMPT_TOKENS" "prompt 上限 4096 token；超长 history 左截断。"
print_param MAX_NEW_TOKENS "$MAX_NEW_TOKENS" "每条 CoT 最多生成 512 token。"
print_param VLLM_MAX_MODEL_LEN "$VLLM_MAX_MODEL_LEN" "总上下文覆盖 4096-token prompt 与 512-token completion。"
print_param KS "$KS" "输出 HR/NDCG@5、10、20、50、100，并输出 MRR、mean rank 和 median rank。"
print_param SEED "$SEED" "生成、embedding 编码和排序随机种子固定为 42。"
print_param EVAL_ROOT "$EVAL_ROOT" "保存生成 CoT、格式审计、逐样本排名和完整指标。"
print_param CONFIRM_RUN "$CONFIRM_RUN" "必须为 1 才开始长耗时评测。"

if [[ "$CONFIRM_RUN" != "1" ]]; then
  echo "未执行。"
  exit 0
fi

source /opt/miniforge3/bin/activate "$VENV"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
mkdir -p "$EVAL_ROOT/checkpoint-134"

checkpoint_dir="$EVAL_ROOT/checkpoint-134"
predictions="$checkpoint_dir/test_generated_cot.jsonl"
audit="$checkpoint_dir/test_generated_cot.audit.json"
metrics="$checkpoint_dir/retrieval_metrics.json"
ranks="$checkpoint_dir/retrieval_ranks.jsonl"

if [[ "$MODE" == "generate" || "$MODE" == "all" ]]; then
  "$VENV/bin/python" manu_src/scripts/inference/vllm_lora_non_target_cot.py \
    --input "$TEST_FILE" \
    --output "$predictions" \
    --audit-output "$audit" \
    --model "$SFT_MODEL" \
    --item-type "CD or vinyl release" \
    --language en \
    --generation-batch-size "$GENERATION_BATCH_SIZE" \
    --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --temperature "$TEMPERATURE" \
    --top-k "$TOP_K" \
    --top-p "$TOP_P" \
    --max-output-words 512 \
    --vllm-max-model-len "$VLLM_MAX_MODEL_LEN" \
    --vllm-max-num-seqs "$GENERATION_BATCH_SIZE" \
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
    --max-attempts 1 \
    --allow-noncanonical-output \
    --expected-split test \
    --seed "$SEED"
fi

if [[ "$MODE" == "evaluate" || "$MODE" == "all" ]]; then
  if [[ ! -s "$predictions" ]]; then
    echo "缺少推理结果：$predictions" >&2
    exit 1
  fi
  "$VENV/bin/python" manu_src/scripts/eval/evaluate_embedding_fullset.py \
    --checkpoint "$EMBEDDING_SCORER" \
    --test-file "$predictions" \
    --expected-split test \
    --item-info "$ITEM_INFO" \
    --output "$metrics" \
    --ranks-output "$ranks" \
    --max-length 4096 \
    --item-batch-size 128 \
    --query-batch-size 64 \
    --score-batch-size 128 \
    --ks "$KS" \
    --seed "$SEED" \
    --attn-implementation flash_attention_2
fi

if [[ "$MODE" != "generate" ]]; then
  "$VENV/bin/python" - "$checkpoint_dir" "$EVAL_ROOT" <<'PY'
import json
import sys
from pathlib import Path

checkpoint_dir = Path(sys.argv[1])
root = Path(sys.argv[2])
result = json.loads((checkpoint_dir / "retrieval_metrics.json").read_text(encoding="utf-8"))
audit = json.loads((checkpoint_dir / "test_generated_cot.audit.json").read_text(encoding="utf-8"))
row = {
    "checkpoint": "checkpoint-134",
    "split": "test",
    "format_valid_rate": audit["format_valid_rows"] / audit["output_rows"],
    "format_valid_rows": audit["format_valid_rows"],
    "invalid_format_rows": audit["invalid_format_rows"],
    "mean_cot_words": audit["mean_cot_words"],
    "mean_generation_tokens": audit["mean_generation_tokens"],
    "finish_reason_counts": audit["finish_reason_counts"],
    **result["metrics"],
}
(root / "all_checkpoint_metrics.json").write_text(
    json.dumps([row], ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)
print(json.dumps(row, ensure_ascii=False, indent=2))
PY
fi
