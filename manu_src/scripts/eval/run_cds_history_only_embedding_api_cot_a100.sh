#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/user/aaai_pro}
VENV=${VENV:-/home/user/.conda/envs/aaai_pro}
HISTORY_RUN=${HISTORY_RUN:-$ROOT/manu_src/model_outputs/CDs_and_Vinyl/embedding/qwen3emb06b_time_title_rating_store_categories_desc256_details256_fieldtrim_bs128_ga1_lr2e5_ep5_len4096_seed42}
API_COT_TEST=${API_COT_TEST:-$ROOT/manu_src/datas/CDs_and_Vinyl/cot/eval/history_plus_api_non_target_cot__input_time_title_rating_store_categories_desc256_details256_v1/test.jsonl}
API_COT_TEST_MANIFEST=${API_COT_TEST_MANIFEST:-$ROOT/manu_src/datas/CDs_and_Vinyl/cot/eval/history_plus_api_non_target_cot__input_time_title_rating_store_categories_desc256_details256_v1/test.manifest.json}
ITEM_INFO=${ITEM_INFO:-$ROOT/manu_src/datas/CDs_and_Vinyl/arrow_to_jsonls/item_info.jsonl}
HISTORY_BASELINE_ROOT=${HISTORY_BASELINE_ROOT:-$ROOT/manu_src/eval_results/CDs_and_Vinyl/embedding/qwen3emb06b_time_title_rating_store_categories_desc256_details256_fieldtrim_bs128_ga1_lr2e5_ep5_len4096_seed42_test_fullset}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT/manu_src/model_outputs/CDs_and_Vinyl/eval/history_only_embedding_on_api_glm52_non_target_cot_time_title_rating_store_categories_desc256_details256_v1}

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

for path in \
  "$ROOT/AGENTS.md" \
  "$VENV/bin/python" \
  "$HISTORY_RUN" \
  "$API_COT_TEST" \
  "$API_COT_TEST_MANIFEST" \
  "$ITEM_INFO" \
  "$HISTORY_BASELINE_ROOT"; do
  if [[ ! -e "$path" ]]; then
    echo "缺少 History-only × API-CoT 评测依赖：$path" >&2
    exit 1
  fi
done

mapfile -t checkpoints < <(
  find "$HISTORY_RUN" -maxdepth 1 -type d -name 'checkpoint-epoch-*' | sort -V
)
if [[ ${#checkpoints[@]} != 5 ]]; then
  echo "History-only embedding checkpoint 应为 5 个，当前为 ${#checkpoints[@]}。" >&2
  exit 1
fi
if [[ $(wc -l < "$API_COT_TEST" | tr -d ' ') != "1341" ]]; then
  echo "API-CoT test 应为 1341 条。" >&2
  exit 1
fi

echo "CDs_and_Vinyl History-only embedding × API-CoT 评测参数："
print_param HISTORY_RUN "$HISTORY_RUN/checkpoint-epoch-{01..05}" "5 个 embedding checkpoint 的训练 query 仅含 History，没有使用 CoT。"
print_param API_COT_TEST "$API_COT_TEST" "1341 条 History+GLM-5.2 API-CoT test query；target 只用于指标计算。"
print_param QUERY "history + Recommendation reasoning + API CoT" "只改变测试 query，不训练或更新 embedding 参数。"
print_param HISTORY_BASELINE_ROOT "$HISTORY_BASELINE_ROOT" "复用已完成的纯 History 指标，按相同 epoch 计算配对差值。"
print_param CANDIDATES "12000 full items" "候选文本 formatter 与 embedding positive 保持一致。"
print_param SEEN_ITEM_MASK "enabled" "屏蔽历史已交互物品，同时保留监督 target。"
print_param METRICS "MRR, mean/median rank, HR/NDCG@5,10,20" "输出全指标、逐样本 rank 和相对纯 History 的差值。"
print_param MAX_LENGTH "$MAX_LENGTH" "超长时只裁剪最旧历史字段，API-CoT 保持完整。"
print_param ITEM_BATCH_SIZE "$ITEM_BATCH_SIZE" "候选物品编码批量。"
print_param QUERY_BATCH_SIZE "$QUERY_BATCH_SIZE" "History+API-CoT query 编码批量。"
print_param SCORE_BATCH_SIZE "$SCORE_BATCH_SIZE" "完整候选打分批量。"
print_param OUTPUT_ROOT "$OUTPUT_ROOT" "5 组指标、逐样本 ranks、汇总和配对比较输出目录。"
print_param SEED "$SEED" "评测随机种子固定为项目统一值 42。"
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

combination_complete() {
  local metrics=$1 ranks=$2 checkpoint=$3
  [[ -s "$metrics" && -s "$ranks" ]] || return 1
  [[ $(wc -l < "$ranks" | tr -d ' ') == "1341" ]] || return 1
  jq -e --arg checkpoint "$checkpoint" --arg test_file "$API_COT_TEST" \
    '.checkpoint == $checkpoint and .test_file == $test_file and
     .evaluated == 1341 and .num_candidates == 12000 and .seed == 42' \
    "$metrics" >/dev/null
}

for checkpoint in "${checkpoints[@]}"; do
  checkpoint_name=$(basename "$checkpoint")
  combo_dir=$OUTPUT_ROOT/$checkpoint_name
  metrics=$combo_dir/retrieval_metrics.json
  ranks=$combo_dir/retrieval_ranks.jsonl
  mkdir -p "$combo_dir"
  if combination_complete "$metrics" "$ranks" "$checkpoint"; then
    echo "已验证并跳过：$checkpoint_name"
    continue
  fi
  "$VENV/bin/python" manu_src/scripts/eval/evaluate_embedding_fullset.py \
    --checkpoint "$checkpoint" \
    --test-file "$API_COT_TEST" \
    --item-info "$ITEM_INFO" \
    --output "$metrics" \
    --ranks-output "$ranks" \
    --max-length "$MAX_LENGTH" \
    --item-batch-size "$ITEM_BATCH_SIZE" \
    --query-batch-size "$QUERY_BATCH_SIZE" \
    --score-batch-size "$SCORE_BATCH_SIZE" \
    --ks 5,10,20 \
    --seed "$SEED" \
    --attn-implementation flash_attention_2
  echo "组合完成：$checkpoint_name"
done

"$VENV/bin/python" - "$OUTPUT_ROOT" "$HISTORY_BASELINE_ROOT" <<'PY'
import json
import statistics
import sys
from pathlib import Path

output_root = Path(sys.argv[1])
baseline_root = Path(sys.argv[2])
metric_names = [
    "MRR",
    "mean_rank",
    "median_rank",
    "HR@5",
    "NDCG@5",
    "HR@10",
    "NDCG@10",
    "HR@20",
    "NDCG@20",
]

rows = []
comparisons = []
for epoch in range(1, 6):
    checkpoint_name = f"checkpoint-epoch-{epoch:02d}"
    cot_result = json.loads(
        (output_root / checkpoint_name / "retrieval_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    baseline_result = json.loads(
        (baseline_root / f"{checkpoint_name}_test_fullset.json").read_text(
            encoding="utf-8"
        )
    )
    cot_metrics = cot_result["metrics"]
    baseline_metrics = baseline_result["metrics"]
    rows.append({"embedding_epoch": epoch, **cot_metrics})
    comparisons.append(
        {
            "embedding_epoch": epoch,
            "history_only": baseline_metrics,
            "history_plus_api_cot": cot_metrics,
            "delta_api_cot_minus_history": {
                name: cot_metrics[name] - baseline_metrics[name]
                for name in metric_names
            },
        }
    )

averages = {}
for name in metric_names:
    baseline_mean = statistics.mean(
        row["history_only"][name] for row in comparisons
    )
    cot_mean = statistics.mean(
        row["history_plus_api_cot"][name] for row in comparisons
    )
    averages[name] = {
        "history_only_mean": baseline_mean,
        "history_plus_api_cot_mean": cot_mean,
        "delta": cot_mean - baseline_mean,
        "relative_percent": (
            (cot_mean - baseline_mean) / baseline_mean * 100.0
            if baseline_mean
            else None
        ),
    }

(output_root / "all_metrics.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
(output_root / "all_metrics.jsonl").write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    encoding="utf-8",
)
(output_root / "comparison_vs_history_only.json").write_text(
    json.dumps(
        {"per_epoch": comparisons, "five_epoch_means": averages},
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
print(json.dumps({"per_epoch": comparisons, "five_epoch_means": averages}, ensure_ascii=False, indent=2))
PY

echo "History-only embedding × API-CoT 的 5 组评测完成：$OUTPUT_ROOT/comparison_vs_history_only.json"
