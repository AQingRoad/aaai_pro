#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/user/aaai_pro}
VENV=${VENV:-/home/user/.conda/envs/aaai_pro}
BASE_MODEL=${BASE_MODEL:-}
GRPO_RUN=${GRPO_RUN:-$ROOT/manu_src/model_outputs/CDs_and_Vinyl/grpo/qwen25_3b_lora_sft20_grpo80_cottrained_logsoftmaxsim_w0p8_ndcg100_w0p2_g4_genbs16_bs4_ga1_vllmsleep1_vllm0p10_lr2e5_ep1_vllmlen4608_clen512_seed42/v0-20260717-155416}
CHECKPOINT_TYPE=${CHECKPOINT_TYPE:-lora}
ITEM_INFO=${ITEM_INFO:-$ROOT/manu_src/datas/CDs_and_Vinyl/arrow_to_jsonls/item_info.jsonl}
EMBEDDING_SCORER=${EMBEDDING_SCORER:-$ROOT/manu_src/model_outputs/CDs_and_Vinyl/embedding/qwen3emb06b_history_plus_glm52_non_target_cot_input_time_title_rating_store_categories_desc256_details256_v1_bs128_ga1_lr2e5_ep5_len4096_seed42/checkpoint-epoch-01}

MODE=${MODE:-all}
EVAL_SPLIT=${EVAL_SPLIT:-valid}
CHECKPOINT_STEPS=${CHECKPOINT_STEPS:-300,600,900}
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

case "$EVAL_SPLIT" in
  valid)
    DEFAULT_EVAL_FILE=$ROOT/manu_src/datas/CDs_and_Vinyl/train_datas/time_title_rating_store_categories_desc256_details256_v1.0/val.jsonl
    DEFAULT_EXPECTED_ROWS=1340
    DEFAULT_EVAL_ROOT=$ROOT/manu_src/model_outputs/CDs_and_Vinyl/eval/qwen25_3b_standard_grpo_sft20_valid_oneshot_raw_completion_cottrained_epoch01_seed42
    ;;
  test)
    DEFAULT_EVAL_FILE=$ROOT/manu_src/datas/CDs_and_Vinyl/train_datas/time_title_rating_store_categories_desc256_details256_v1.0/test.jsonl
    DEFAULT_EXPECTED_ROWS=1341
    DEFAULT_EVAL_ROOT=$ROOT/manu_src/model_outputs/CDs_and_Vinyl/eval/qwen25_3b_standard_grpo_sft20_test_oneshot_raw_completion_cottrained_epoch01_seed42
    ;;
  *)
    echo "EVAL_SPLIT 必须为 valid 或 test。" >&2
    exit 1
    ;;
esac

EVAL_FILE=${EVAL_FILE:-$DEFAULT_EVAL_FILE}
EXPECTED_ROWS=${EXPECTED_ROWS:-$DEFAULT_EXPECTED_ROWS}
EVAL_ROOT=${EVAL_ROOT:-$DEFAULT_EVAL_ROOT}

print_param() {
  local name=$1 value=$2 description=$3
  printf '  %-34s %s\n      %s\n' "$name=$value" "" "$description"
}

if [[ "$MODE" != "generate" && "$MODE" != "evaluate" && "$MODE" != "all" ]]; then
  echo "MODE 必须为 generate、evaluate 或 all。" >&2
  exit 1
fi
if [[ "$CHECKPOINT_TYPE" != "lora" && "$CHECKPOINT_TYPE" != "full" ]]; then
  echo "CHECKPOINT_TYPE 必须为 lora 或 full。" >&2
  exit 1
fi
if [[ "$SEED" != "42" ]]; then
  echo "项目随机种子必须为 42。" >&2
  exit 1
fi
for path in "$ROOT/AGENTS.md" "$VENV/bin/python" "$GRPO_RUN" "$EVAL_FILE" "$ITEM_INFO" "$EMBEDDING_SCORER"; do
  if [[ ! -e "$path" ]]; then
    echo "缺少评测依赖：$path" >&2
    exit 1
  fi
done
if [[ "$(wc -l < "$EVAL_FILE" | tr -d ' ')" != "$EXPECTED_ROWS" ]]; then
  echo "$EVAL_SPLIT 应为 $EXPECTED_ROWS 条。" >&2
  exit 1
fi

IFS=',' read -r -a STEPS <<< "$CHECKPOINT_STEPS"
CHECKPOINTS=()
for step in "${STEPS[@]}"; do
  checkpoint="$GRPO_RUN/checkpoint-$step"
  if [[ "$CHECKPOINT_TYPE" == "lora" ]]; then
    if [[ ! -s "$checkpoint/adapter_model.safetensors" || ! -s "$checkpoint/adapter_config.json" ]]; then
      echo "缺少完整 LoRA checkpoint：$checkpoint" >&2
      exit 1
    fi
  else
    if [[ ! -s "$checkpoint/config.json" ]]; then
      echo "完整模型 checkpoint 缺少 config.json：$checkpoint" >&2
      exit 1
    fi
    if [[ ! -s "$checkpoint/model.safetensors" && ! -s "$checkpoint/model.safetensors.index.json" ]]; then
      echo "完整模型 checkpoint 缺少 safetensors 权重：$checkpoint" >&2
      exit 1
    fi
    if [[ -e "$checkpoint/adapter_config.json" ]]; then
      echo "full 模式检测到 LoRA 配置，拒绝混用：$checkpoint" >&2
      exit 1
    fi
  fi
  CHECKPOINTS+=("$checkpoint")
done

if [[ "$CHECKPOINT_TYPE" == "lora" ]]; then
  DERIVED_BASE_MODEL=$("$VENV/bin/python" - "${CHECKPOINTS[@]}" <<'PY'
import json
import sys
from pathlib import Path

base_models = []
for checkpoint_text in sys.argv[1:]:
    checkpoint = Path(checkpoint_text)
    config = json.loads((checkpoint / "adapter_config.json").read_text(encoding="utf-8"))
    base_model = str(config.get("base_model_name_or_path") or "").strip()
    if not base_model:
        raise SystemExit(f"{checkpoint}/adapter_config.json 缺少 base_model_name_or_path")
    base_path = Path(base_model).expanduser()
    base_models.append(str(base_path.resolve()) if base_path.exists() else base_model.rstrip("/"))

if len(set(base_models)) != 1:
    raise SystemExit(f"LoRA checkpoint 的基座不一致：{base_models}")
print(base_models[0])
PY
  )
  if [[ -z "$BASE_MODEL" ]]; then
    BASE_MODEL=$DERIVED_BASE_MODEL
  fi
  if [[ ! -e "$BASE_MODEL" ]]; then
    echo "缺少 LoRA 声明的基座模型：$BASE_MODEL" >&2
    exit 1
  fi
  if [[ "$(readlink -f "$BASE_MODEL")" != "$(readlink -f "$DERIVED_BASE_MODEL")" ]]; then
    echo "BASE_MODEL 与 LoRA adapter_config 不一致。" >&2
    echo "  BASE_MODEL=$BASE_MODEL" >&2
    echo "  adapter base=$DERIVED_BASE_MODEL" >&2
    exit 1
  fi
else
  "$VENV/bin/python" - "${CHECKPOINTS[@]}" <<'PY'
import json
import sys
from pathlib import Path

for checkpoint_text in sys.argv[1:]:
    checkpoint = Path(checkpoint_text)
    index_path = checkpoint / "model.safetensors.index.json"
    if index_path.is_file():
        payload = json.loads(index_path.read_text(encoding="utf-8"))
        shards = sorted(set(payload.get("weight_map", {}).values()))
        if not shards:
            raise SystemExit(f"权重索引为空：{index_path}")
        missing = [name for name in shards if not (checkpoint / name).is_file()]
        if missing:
            raise SystemExit(f"checkpoint 缺少权重分片：{checkpoint}: {missing}")
PY
fi

echo "Qwen2.5-3B 标准 GRPO checkpoint $EVAL_SPLIT 评测参数："
print_param MODE "$MODE" "all 表示依次生成 CoT 并执行完整候选检索评测。"
print_param EVAL_SPLIT "$EVAL_SPLIT" "当前评测数据划分。"
print_param CHECKPOINTS "$CHECKPOINT_STEPS" "按给定顺序评测这些标准 GRPO checkpoint。"
print_param CHECKPOINT_TYPE "$CHECKPOINT_TYPE" "lora 使用固定基座挂载 adapter；full 直接加载每个 checkpoint 的完整权重且不执行 LoRA 合并。"
if [[ "$CHECKPOINT_TYPE" == "lora" ]]; then
  print_param BASE_MODEL "$BASE_MODEL" "从各 checkpoint 的 adapter_config.json 自动读取并交叉验证；必须是 GRPO 训练使用的 full-SFT 基座。"
else
  print_param FULL_MODEL_SOURCE checkpoint "vLLM 的 model 参数直接指向当前 full checkpoint；不读取 adapter，不合并权重。"
fi
print_param EVAL_FILE "$EVAL_FILE" "$EXPECTED_ROWS 条 $EVAL_SPLIT history；target 字段不进入生成 prompt。"
print_param EMBEDDING_SCORER "$EMBEDDING_SCORER" "固定使用 GRPO reward 对应的 CoT-trained embedding epoch-01。"
print_param ITEM_INFO "$ITEM_INFO" "12000 个真实候选；沿用训练 positive formatter，并屏蔽历史物品。"
print_param GRPO_UPDATE "all_group_completions" "这些 checkpoint 使用标准 GRPO；同组 4 条 completion 均按组相对优势参与更新。"
print_param COMPLETION_POLICY "one_shot_raw_completion" "每条 $EVAL_SPLIT history 只采样一次；格式不完整的非空 completion 仍原样进入检索，并单独统计格式合规率。"
print_param GENERATION_BATCH_SIZE "$GENERATION_BATCH_SIZE" "vLLM 单批生成的 $EVAL_SPLIT query 数。"
print_param TEMPERATURE "$TEMPERATURE" "与 GRPO rollout 一致的生成温度。"
print_param TOP_K "$TOP_K" "与 GRPO rollout 一致，每步保留概率最高的 200 个 token。"
print_param TOP_P "$TOP_P" "与 GRPO rollout 一致，不额外裁剪 nucleus 概率质量。"
print_param MAX_PROMPT_TOKENS "$MAX_PROMPT_TOKENS" "prompt 上限 4096 token；超长时左截断。"
print_param MAX_NEW_TOKENS "$MAX_NEW_TOKENS" "每条 CoT 最多生成 512 token。"
print_param VLLM_MAX_MODEL_LEN "$VLLM_MAX_MODEL_LEN" "总上下文覆盖 4096-token prompt 与 512-token completion。"
print_param KS "$KS" "输出 HR/NDCG@5、10、20、50、100，并输出 MRR、mean rank 和 median rank。"
print_param SEED "$SEED" "生成、embedding 编码和排序随机种子固定为 42。"
print_param EVAL_ROOT "$EVAL_ROOT" "全部 checkpoint 的生成、审计、rank 和指标输出目录。"
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
  predictions="$checkpoint_dir/${EVAL_SPLIT}_generated_cot.jsonl"
  audit="$checkpoint_dir/${EVAL_SPLIT}_generated_cot.audit.json"
  metrics="$checkpoint_dir/retrieval_metrics.json"
  ranks="$checkpoint_dir/retrieval_ranks.jsonl"
  mkdir -p "$checkpoint_dir"

  if [[ "$MODE" == "generate" || "$MODE" == "all" ]]; then
    model_args=()
    if [[ "$CHECKPOINT_TYPE" == "lora" ]]; then
      model_args+=(--model "$BASE_MODEL" --adapter "$checkpoint")
    else
      model_args+=(--model "$checkpoint")
    fi
    "$VENV/bin/python" manu_src/scripts/inference/vllm_lora_non_target_cot.py \
      --input "$EVAL_FILE" \
      --output "$predictions" \
      --audit-output "$audit" \
      "${model_args[@]}" \
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
      --expected-split "$EVAL_SPLIT" \
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
      --expected-split "$EVAL_SPLIT" \
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
  "$VENV/bin/python" - "$EVAL_ROOT" "$EVAL_SPLIT" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
eval_split = sys.argv[2]
rows = []
for path in sorted(
    root.glob("checkpoint-*/retrieval_metrics.json"),
    key=lambda value: int(value.parent.name.rsplit("-", 1)[1]),
):
    result = json.loads(path.read_text(encoding="utf-8"))
    audit_path = path.parent / f"{eval_split}_generated_cot.audit.json"
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    rows.append(
        {
            "checkpoint": path.parent.name,
            "split": eval_split,
            "format_valid_rate": audit["format_valid_rows"] / audit["output_rows"],
            "format_valid_rows": audit["format_valid_rows"],
            "invalid_format_rows": audit["invalid_format_rows"],
            "mean_cot_words": audit["mean_cot_words"],
            "mean_generation_tokens": audit["mean_generation_tokens"],
            "finish_reason_counts": audit["finish_reason_counts"],
            **result["metrics"],
        }
    )
(root / "all_checkpoint_metrics.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
print(json.dumps(rows, ensure_ascii=False, indent=2))
PY
fi
