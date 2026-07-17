#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/user/aaai_pro}
VENV=${VENV:-/home/user/.conda/envs/aaai_pro}
TEST_PAIRS=${TEST_PAIRS:-$ROOT/manu_src/datas/CDs_and_Vinyl/train_datas/time_title_rating_store_categories_desc256_details256_v1.0/test.jsonl}
API_COT_DIR=${API_COT_DIR:-$ROOT/manu_src/datas/CDs_and_Vinyl/cot/api/history_only_next_item_feature_cot__input_time_title_rating_store_categories_desc256_details256_v1}
API_COT=${API_COT:-$API_COT_DIR/cot_non_target_glm52_test_full_seed42_temp1.jsonl}
API_COT_MANIFEST=${API_COT_MANIFEST:-$API_COT_DIR/cot_non_target_glm52_test_full_seed42_temp1.manifest.json}
EVAL_TEST_DIR=${EVAL_TEST_DIR:-$ROOT/manu_src/datas/CDs_and_Vinyl/cot/eval/history_plus_api_non_target_cot__input_time_title_rating_store_categories_desc256_details256_v1}
EVAL_TEST=${EVAL_TEST:-$EVAL_TEST_DIR/test.jsonl}
EVAL_TEST_MANIFEST=${EVAL_TEST_MANIFEST:-$EVAL_TEST_DIR/test.manifest.json}
ITEM_INFO=${ITEM_INFO:-$ROOT/manu_src/datas/CDs_and_Vinyl/arrow_to_jsonls/item_info.jsonl}
FULL_OUT=${FULL_OUT:-$ROOT/manu_src/model_outputs/CDs_and_Vinyl/embedding/qwen3emb06b_history_plus_glm52_non_target_cot_input_time_title_rating_store_categories_desc256_details256_v1_bs128_ga1_lr2e5_ep5_len4096_seed42}
MASK_OUT=${MASK_OUT:-$ROOT/manu_src/model_outputs/CDs_and_Vinyl/embedding/qwen3emb06b_history_plus_glm52_non_target_cot_whole_cot_mask_p0p5_input_time_title_rating_store_categories_desc256_details256_v1_bs128_ga1_lr2e5_ep5_len4096_seed42}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT/manu_src/model_outputs/CDs_and_Vinyl/eval/api_glm52_non_target_cot_time_title_rating_store_categories_desc256_details256_v1/embedding_checkpoint_matrix}

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
for path in "$ROOT/AGENTS.md" "$VENV/bin/python" "$TEST_PAIRS" "$API_COT" "$API_COT_MANIFEST" "$ITEM_INFO" "$FULL_OUT" "$MASK_OUT"; do
  if [[ ! -e "$path" ]]; then
    echo "缺少 API-CoT 评测依赖：$path" >&2
    exit 1
  fi
done

api_rows=$(wc -l < "$API_COT" | tr -d ' ')
full_checkpoints=$(find "$FULL_OUT" -maxdepth 1 -type d -name 'checkpoint-epoch-*' | wc -l | tr -d ' ')
mask_checkpoints=$(find "$MASK_OUT" -maxdepth 1 -type d -name 'checkpoint-epoch-*' | wc -l | tr -d ' ')
if [[ "$api_rows" != "1341" || "$full_checkpoints" != "5" || "$mask_checkpoints" != "5" ]]; then
  echo "评测输入不完整：API CoT=$api_rows/1341，full=$full_checkpoints/5，mask=$mask_checkpoints/5。" >&2
  exit 1
fi

echo "CDs_and_Vinyl API-CoT × embedding checkpoint 评测参数："
print_param TEST_PAIRS "$TEST_PAIRS" "1341 条原始 test history 与监督 target；target 只用于指标计算。"
print_param API_COT "$API_COT" "GLM-5.2 API 非 target CoT；1341 条，temperature=1.0，top_p=0.9。"
print_param API_PROVIDER_MIX "ks_tokenverse=770, glm_official=571" "两类 provider 均调用 GLM-5.2，沿用已完成生成结果。"
print_param QUERY "history + Recommendation reasoning + API CoT" "与 SFT-CoT 评测使用相同分隔符和 embedding query formatter。"
print_param EMBEDDING_FULL "$FULL_OUT/checkpoint-epoch-{01..05}" "完整 CoT embedding 的每轮 checkpoint。"
print_param EMBEDDING_MASK "$MASK_OUT/checkpoint-epoch-{01..05}" "整段 CoT mask p=0.5 embedding 的每轮 checkpoint。"
print_param COMBINATIONS "2×5×1=10" "每个 embedding checkpoint 评测同一份 API test CoT。"
print_param CANDIDATES "12000 full items" "候选 formatter 与训练 positive 一致。"
print_param SEEN_ITEM_MASK "enabled" "屏蔽历史已交互物品，同时保留监督 target。"
print_param METRICS "MRR, mean/median rank, HR/NDCG@5,10,20" "使用完整候选排序指标。"
print_param MAX_LENGTH "$MAX_LENGTH" "保持与训练及 SFT-CoT 评测一致。"
print_param ITEM_BATCH_SIZE "$ITEM_BATCH_SIZE" "候选物品编码批量。"
print_param QUERY_BATCH_SIZE "$QUERY_BATCH_SIZE" "API-CoT test query 编码批量。"
print_param SCORE_BATCH_SIZE "$SCORE_BATCH_SIZE" "完整候选打分批量。"
print_param OUTPUT_ROOT "$OUTPUT_ROOT" "10 组指标、逐样本 ranks 与全局汇总目录。"
print_param RESUME_POLICY "validated combination artifacts" "仅跳过 scorer、test 和 1341 条 ranks 均匹配的组合。"
print_param SEED "$SEED" "数据构造、模型编码和评测随机种子。"
print_param CONFIRM_RUN "$CONFIRM_RUN" "必须为 1 才构造评测输入并执行长耗时评测。"

if [[ "$CONFIRM_RUN" != "1" ]]; then
  echo "未执行。"
  exit 0
fi

source /opt/miniforge3/bin/activate "$VENV"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

"$VENV/bin/python" manu_src/scripts/cot/build_api_cot_test_eval_dataset.py \
  --pairs "$TEST_PAIRS" \
  --cot "$API_COT" \
  --cot-manifest "$API_COT_MANIFEST" \
  --output "$EVAL_TEST" \
  --manifest-output "$EVAL_TEST_MANIFEST" \
  --expected-rows 1341 \
  --max-output-words 512 \
  --seed "$SEED"

mkdir -p "$OUTPUT_ROOT"

combination_complete() {
  local metrics=$1 ranks=$2 checkpoint=$3
  [[ -s "$metrics" && -s "$ranks" ]] || return 1
  [[ $(wc -l < "$ranks" | tr -d ' ') == "1341" ]] || return 1
  jq -e --arg checkpoint "$checkpoint" --arg test_file "$EVAL_TEST" \
    '.checkpoint == $checkpoint and .test_file == $test_file and
     .evaluated == 1341 and .num_candidates == 12000 and .seed == 42' \
    "$metrics" >/dev/null
}

for variant in history_plus_cot whole_cot_mask_p0p5; do
  if [[ "$variant" == "history_plus_cot" ]]; then
    run_dir=$FULL_OUT
  else
    run_dir=$MASK_OUT
  fi
  mapfile -t checkpoints < <(find "$run_dir" -maxdepth 1 -type d -name 'checkpoint-epoch-*' | sort -V)
  for checkpoint in "${checkpoints[@]}"; do
    checkpoint_name=$(basename "$checkpoint")
    combo_dir=$OUTPUT_ROOT/$variant/$checkpoint_name
    metrics=$combo_dir/retrieval_metrics.json
    ranks=$combo_dir/retrieval_ranks.jsonl
    mkdir -p "$combo_dir"
    if combination_complete "$metrics" "$ranks" "$checkpoint"; then
      echo "已验证并跳过：$variant/$checkpoint_name"
      continue
    fi
    "$VENV/bin/python" manu_src/scripts/eval/evaluate_embedding_fullset.py \
      --checkpoint "$checkpoint" \
      --test-file "$EVAL_TEST" \
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
    echo "组合完成：$variant/$checkpoint_name"
  done
done

"$VENV/bin/python" - "$OUTPUT_ROOT" <<'PY'
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in root.glob("*/checkpoint-epoch-*/retrieval_metrics.json"):
    result = json.loads(path.read_text(encoding="utf-8"))
    epoch = int(re.search(r"(\d+)$", path.parent.name).group(1))
    rows.append(
        {
            "embedding_variant": path.parent.parent.name,
            "embedding_checkpoint": path.parent.name,
            "embedding_epoch": epoch,
            **result["metrics"],
        }
    )
rows.sort(key=lambda row: (row["embedding_variant"], row["embedding_epoch"]))
if len(rows) != 10:
    raise RuntimeError(f"API-CoT 评测应有 10 组，当前为 {len(rows)}")
(root / "all_metrics.json").write_text(
    json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
(root / "all_metrics.jsonl").write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
    encoding="utf-8",
)
print(json.dumps(rows, ensure_ascii=False, indent=2))
PY

echo "API-CoT 全部 10 组评测完成：$OUTPUT_ROOT/all_metrics.json"
