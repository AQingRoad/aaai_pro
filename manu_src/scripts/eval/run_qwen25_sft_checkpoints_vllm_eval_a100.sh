#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/user/aaai_pro}
VENV=${VENV:-/home/user/.conda/envs/aaai_pro}
BASE_MODEL=${BASE_MODEL:-/home/user/models_hf/Qwen2.5-3B-Instruct}
SFT_OUT=${SFT_OUT:-$ROOT/manu_src/model_outputs/CDs_and_Vinyl/sft/qwen25_3b_lora_history_only_next_item_feature_cot_time_title_rating_store_categories_desc256_details256_v1_bs32_ga1_lr2e5_ep5_len4096_seed42}
TEST_FILE=${TEST_FILE:-$ROOT/manu_src/datas/CDs_and_Vinyl/train_datas/time_title_rating_store_categories_desc256_details256_v1.0/test.jsonl}
ITEM_INFO=${ITEM_INFO:-$ROOT/manu_src/datas/CDs_and_Vinyl/arrow_to_jsonls/item_info.jsonl}
EMBEDDING_SCORER=${EMBEDDING_SCORER:-$ROOT/manu_src/model_outputs/CDs_and_Vinyl/embedding/qwen3emb06b_history_plus_glm52_non_target_cot_input_time_title_rating_store_categories_desc256_details256_v1_bs64_ga1_lr2e5_ep5_len4096_seed42/checkpoint-epoch-05}
EVAL_ROOT=${EVAL_ROOT:-$ROOT/manu_src/model_outputs/CDs_and_Vinyl/eval/qwen25_3b_lora_history_only_next_item_feature_cot_time_title_rating_store_categories_desc256_details256_v1}

MODE=${MODE:-all}
GENERATION_BATCH_SIZE=${GENERATION_BATCH_SIZE:-32}
TEMPERATURE=${TEMPERATURE:-1.0}
TOP_P=${TOP_P:-0.9}
MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS:-4096}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
SEED=${SEED:-42}
CONFIRM_RUN=${CONFIRM_RUN:-0}

print_param() {
  local name=$1 value=$2 description=$3
  printf '  %-34s %s\n      %s\n' "$name=$value" "" "$description"
}

if [[ "$MODE" != "generate" && "$MODE" != "evaluate" && "$MODE" != "all" ]]; then
  echo "MODE 必须是 generate、evaluate 或 all。" >&2
  exit 1
fi
if [[ "$SEED" != "42" ]]; then
  echo "项目随机种子必须为 42。" >&2
  exit 1
fi
for path in "$ROOT/AGENTS.md" "$VENV/bin/python" "$BASE_MODEL" "$SFT_OUT" "$TEST_FILE" "$ITEM_INFO"; do
  if [[ ! -e "$path" ]]; then
    echo "缺少评测依赖：$path" >&2
    exit 1
  fi
done
if [[ "$MODE" != "generate" && ! -e "$EMBEDDING_SCORER" ]]; then
  echo "缺少固定 embedding scorer：$EMBEDDING_SCORER" >&2
  exit 1
fi

mapfile -t CHECKPOINTS < <(find "$SFT_OUT" -type d -name 'checkpoint-*' | sort -V)
if [[ ${#CHECKPOINTS[@]} -ne 5 ]]; then
  echo "SFT 输出应有 5 个 checkpoint，当前找到 ${#CHECKPOINTS[@]} 个。" >&2
  printf '%s\n' "${CHECKPOINTS[@]}" >&2
  exit 1
fi

echo "SFT checkpoint vLLM 生成与检索评测参数："
print_param MODE "$MODE" "generate 只生成；evaluate 只排序；all 连续执行两者。"
print_param BASE_MODEL "$BASE_MODEL" "vLLM 加载的 Qwen2.5-3B-Instruct 基座。"
print_param SFT_OUT "$SFT_OUT" "包含 5 个 LoRA checkpoint 的正式 SFT 输出。"
print_param TEST_FILE "$TEST_FILE" "1341 条 test history；target 字段仅保留给指标脚本。"
print_param EMBEDDING_SCORER "$EMBEDDING_SCORER" "固定的 history+CoT embedding 第 5 轮 checkpoint，不按本次 test 指标选模。"
print_param ITEM_INFO "$ITEM_INFO" "12000 个真实候选物品，文本 formatter 与训练 positive 一致。"
print_param TEMPERATURE "$TEMPERATURE" "vLLM 生成温度，与 API CoT 数据的 1.0 一致。"
print_param TOP_P "$TOP_P" "核采样阈值，与 API CoT 数据的 0.9 一致。"
print_param MAX_PROMPT_TOKENS "$MAX_PROMPT_TOKENS" "prompt 最多保留 4096 token；超长时左截断。"
print_param MAX_NEW_TOKENS "$MAX_NEW_TOKENS" "每条 CoT 最多生成 2048 token，并额外检查 512 words。"
print_param GENERATION_BATCH_SIZE "$GENERATION_BATCH_SIZE" "vLLM 单批请求数。"
print_param SEED "$SEED" "生成、embedding 编码与排序随机种子。"
print_param CONFIRM_RUN "$CONFIRM_RUN" "必须为 1 才执行。"

if [[ "$CONFIRM_RUN" != "1" ]]; then
  echo "未执行。"
  exit 0
fi

source /opt/miniforge3/bin/activate "$VENV"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
mkdir -p "$EVAL_ROOT"

for checkpoint in "${CHECKPOINTS[@]}"; do
  checkpoint_name=$(basename "$checkpoint")
  checkpoint_dir="$EVAL_ROOT/$checkpoint_name"
  predictions="$checkpoint_dir/test_generated_cot.jsonl"
  audit="$checkpoint_dir/test_generated_cot.audit.json"
  metrics="$checkpoint_dir/retrieval_metrics.json"
  ranks="$checkpoint_dir/retrieval_ranks.jsonl"
  mkdir -p "$checkpoint_dir"

  if [[ "$MODE" == "generate" || "$MODE" == "all" ]]; then
    "$VENV/bin/python" manu_src/scripts/inference/vllm_lora_non_target_cot.py \
      --input "$TEST_FILE" \
      --output "$predictions" \
      --audit-output "$audit" \
      --model "$BASE_MODEL" \
      --adapter "$checkpoint" \
      --item-type "CD or vinyl release" \
      --language en \
      --generation-batch-size "$GENERATION_BATCH_SIZE" \
      --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
      --max-new-tokens "$MAX_NEW_TOKENS" \
      --temperature "$TEMPERATURE" \
      --top-p "$TOP_P" \
      --max-output-words 512 \
      --vllm-max-model-len 6144 \
      --vllm-max-num-seqs "$GENERATION_BATCH_SIZE" \
      --gpu-memory-utilization 0.85 \
      --max-attempts 3 \
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
      --item-info "$ITEM_INFO" \
      --output "$metrics" \
      --ranks-output "$ranks" \
      --max-length 4096 \
      --item-batch-size 128 \
      --query-batch-size 64 \
      --score-batch-size 128 \
      --ks 5,10,20 \
      --seed "$SEED" \
      --attn-implementation flash_attention_2
  fi
done

if [[ "$MODE" != "generate" ]]; then
  "$VENV/bin/python" - "$EVAL_ROOT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted(root.glob("checkpoint-*/retrieval_metrics.json")):
    result = json.loads(path.read_text(encoding="utf-8"))
    rows.append({"checkpoint": path.parent.name, **result["metrics"]})
(root / "all_checkpoint_metrics.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(rows, ensure_ascii=False, indent=2))
PY
fi
