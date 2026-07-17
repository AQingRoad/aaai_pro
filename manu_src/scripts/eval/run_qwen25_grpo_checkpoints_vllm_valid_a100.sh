#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/user/aaai_pro}
VENV=${VENV:-/home/user/.conda/envs/aaai_pro}
BASE_MODEL=${BASE_MODEL:-/home/user/models_hf/Qwen2.5-3B-Instruct}
GRPO_RUN=${GRPO_RUN:-$ROOT/manu_src/model_outputs/CDs_and_Vinyl/grpo/qwen25_3b_lora_sft20_grpo80_cottrained_logsoftmaxsim_w0p8_ndcg100_w0p2_g4_genbs16_bs4_ga1_vllmsleep1_vllm0p10_lr2e5_ep1_vllmlen4608_clen512_seed42/v0-20260717-155416}
VALID_FILE=${VALID_FILE:-$ROOT/manu_src/datas/CDs_and_Vinyl/train_datas/time_title_rating_store_categories_desc256_details256_v1.0/val.jsonl}
ITEM_INFO=${ITEM_INFO:-$ROOT/manu_src/datas/CDs_and_Vinyl/arrow_to_jsonls/item_info.jsonl}
EMBEDDING_SCORER=${EMBEDDING_SCORER:-$ROOT/manu_src/model_outputs/CDs_and_Vinyl/embedding/qwen3emb06b_history_plus_glm52_non_target_cot_input_time_title_rating_store_categories_desc256_details256_v1_bs128_ga1_lr2e5_ep5_len4096_seed42/checkpoint-epoch-01}
EVAL_ROOT=${EVAL_ROOT:-$ROOT/manu_src/model_outputs/CDs_and_Vinyl/eval/qwen25_3b_standard_grpo_sft20_valid_cottrained_epoch01_seed42}

MODE=${MODE:-all}
CHECKPOINT_STEPS=${CHECKPOINT_STEPS:-300,600,900}
EXPECTED_ROWS=${EXPECTED_ROWS:-1340}
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
for path in "$ROOT/AGENTS.md" "$VENV/bin/python" "$BASE_MODEL" "$GRPO_RUN" "$VALID_FILE" "$ITEM_INFO" "$EMBEDDING_SCORER"; do
  if [[ ! -e "$path" ]]; then
    echo "缺少评测依赖：$path" >&2
    exit 1
  fi
done
if [[ "$(wc -l < "$VALID_FILE" | tr -d ' ')" != "$EXPECTED_ROWS" ]]; then
  echo "valid 应为 $EXPECTED_ROWS 条。" >&2
  exit 1
fi

IFS=',' read -r -a STEPS <<< "$CHECKPOINT_STEPS"
CHECKPOINTS=()
for step in "${STEPS[@]}"; do
  checkpoint="$GRPO_RUN/checkpoint-$step"
  if [[ ! -s "$checkpoint/adapter_model.safetensors" ]]; then
    echo "缺少完整 checkpoint：$checkpoint" >&2
    exit 1
  fi
  CHECKPOINTS+=("$checkpoint")
done

echo "Qwen2.5-3B 标准 GRPO checkpoint valid 评测参数："
print_param MODE "$MODE" "all 表示依次生成 CoT 并执行完整候选检索评测。"
print_param CHECKPOINTS "$CHECKPOINT_STEPS" "评测已保存的三个标准 GRPO LoRA checkpoint。"
print_param VALID_FILE "$VALID_FILE" "1340 条 valid history；target 字段不进入生成 prompt。"
print_param EMBEDDING_SCORER "$EMBEDDING_SCORER" "固定使用 GRPO reward 对应的 CoT-trained embedding epoch-01。"
print_param ITEM_INFO "$ITEM_INFO" "12000 个真实候选；沿用训练 positive formatter，并屏蔽历史物品。"
print_param GRPO_UPDATE "all_group_completions" "这些 checkpoint 使用标准 GRPO；同组 4 条 completion 均按组相对优势参与更新。"
print_param GENERATION_BATCH_SIZE "$GENERATION_BATCH_SIZE" "vLLM 单批生成的 valid query 数。"
print_param TEMPERATURE "$TEMPERATURE" "与 GRPO rollout 一致的生成温度。"
print_param TOP_K "$TOP_K" "与 GRPO rollout 一致，每步保留概率最高的 200 个 token。"
print_param TOP_P "$TOP_P" "与 GRPO rollout 一致，不额外裁剪 nucleus 概率质量。"
print_param MAX_PROMPT_TOKENS "$MAX_PROMPT_TOKENS" "prompt 上限 4096 token；超长时左截断。"
print_param MAX_NEW_TOKENS "$MAX_NEW_TOKENS" "每条 CoT 最多生成 512 token。"
print_param VLLM_MAX_MODEL_LEN "$VLLM_MAX_MODEL_LEN" "总上下文覆盖 4096-token prompt 与 512-token completion。"
print_param KS "$KS" "输出 HR/NDCG@5、10、20、50、100，并输出 MRR、mean rank 和 median rank。"
print_param SEED "$SEED" "生成、embedding 编码和排序随机种子固定为 42。"
print_param EVAL_ROOT "$EVAL_ROOT" "三个 checkpoint 的生成、审计、rank 和指标输出目录。"
print_param CONFIRM_RUN "$CONFIRM_RUN" "必须为 1 才开始长耗时评测。"

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
  predictions="$checkpoint_dir/valid_generated_cot.jsonl"
  audit="$checkpoint_dir/valid_generated_cot.audit.json"
  metrics="$checkpoint_dir/retrieval_metrics.json"
  ranks="$checkpoint_dir/retrieval_ranks.jsonl"
  mkdir -p "$checkpoint_dir"

  if [[ "$MODE" == "generate" || "$MODE" == "all" ]]; then
    "$VENV/bin/python" manu_src/scripts/inference/vllm_lora_non_target_cot.py \
      --input "$VALID_FILE" \
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
      --top-k "$TOP_K" \
      --top-p "$TOP_P" \
      --max-output-words 512 \
      --vllm-max-model-len "$VLLM_MAX_MODEL_LEN" \
      --vllm-max-num-seqs "$GENERATION_BATCH_SIZE" \
      --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
      --max-attempts 3 \
      --expected-split valid \
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
      --expected-split valid \
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
