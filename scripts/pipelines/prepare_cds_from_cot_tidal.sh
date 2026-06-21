#!/usr/bin/env bash
set -euo pipefail

# Rebuild CDs_and_Vinyl training data from existing CoT candidate lists and
# rubric scores on the Tidal server. This script does not generate or rejudge CoT.

ROOT=${ROOT:-/mnt/tidal-sh01/usr/xiayu6/xiayu/aaai_pro}
VENV=${VENV:-/root/miniconda3/envs/swift}
PYTHON_BIN=${PYTHON_BIN:-$VENV/bin/python}

CATEGORY=${CATEGORY:-CDs_and_Vinyl}
RUN_NAME=${RUN_NAME:-deepseek_v4_pro_cds_remerged}
MODEL=${MODEL:-/mnt/tidal-sh01/usr/xiayu6/xiayu/checkpoint/Qwen3/4B}

COT_ARTIFACT_DIR=${COT_ARTIFACT_DIR:-$ROOT/github_artifacts/CDs_and_Vinyl/cot}
CANDIDATE_LISTS=${CANDIDATE_LISTS:-$COT_ARTIFACT_DIR/cot_candidate_lists_deepseek_v4_pro_low.jsonl}
RUBRIC_SCORES=${RUBRIC_SCORES:-$COT_ARTIFACT_DIR/cot_candidate_lists_deepseek_v4_pro_low.rubric_deepseek_v4_pro.jsonl}
RREC_EVAL_DIR=${RREC_EVAL_DIR:-$ROOT/github_artifacts/CDs_and_Vinyl/rrec_eval}
GAIN_ITEM_INFO=${GAIN_ITEM_INFO:-$RREC_EVAL_DIR/item_info.jsonl}
HISTORY_METADATA_MODE=${HISTORY_METADATA_MODE:-none}
HISTORY_MAX_ITEM_CHARS=${HISTORY_MAX_ITEM_CHARS:-320}
RREC_DATA_ROOT=${RREC_DATA_ROOT:-$ROOT/data}
PHASE0_TRAIN_DATASET=${PHASE0_TRAIN_DATASET:-$ROOT/github_artifacts/CDs_and_Vinyl/phase0/phase0_embedder_rrec_amazon_cds_and_vinyl_train.jsonl}

OUT_DIR=${OUT_DIR:-$ROOT/outputs/rrec_amazon/$CATEGORY}
COT_JUDGED=${COT_JUDGED:-$OUT_DIR/cot_judged_${RUN_NAME}.jsonl}
COT_SCORED=${COT_SCORED:-$OUT_DIR/cot_scored_${RUN_NAME}.jsonl}
SCORED_EXAMPLES=${SCORED_EXAMPLES:-$OUT_DIR/examples_${RUN_NAME}.jsonl}
FILTERED_COT=${FILTERED_COT:-$OUT_DIR/filtered_high_quality_cot_${RUN_NAME}.jsonl}
REJECTED_COT=${REJECTED_COT:-$OUT_DIR/rejected_cot_${RUN_NAME}.jsonl}
FINAL_FILTERED_COT=${FINAL_FILTERED_COT:-$OUT_DIR/filtered_high_quality_cot_${RUN_NAME}.final_sft.jsonl}
COT_SCORE_PLOT=${COT_SCORE_PLOT:-$OUT_DIR/cot_gain_distribution_${RUN_NAME}.svg}
COT_SCORE_SUMMARY=${COT_SCORE_SUMMARY:-$OUT_DIR/cot_gain_distribution_${RUN_NAME}.json}
SFT_DATASET=${SFT_DATASET:-$OUT_DIR/sft_${RUN_NAME}.jsonl}
GRPO_DATASET=${GRPO_DATASET:-$OUT_DIR/grpo_${RUN_NAME}.jsonl}
GRPO_INPUT=${GRPO_INPUT:-$ROOT/data/rrec_amazon/$CATEGORY/examples.jsonl}

EMBEDDER_OUT=${EMBEDDER_OUT:-$ROOT/checkpoints/rrec_amazon_CDs_and_Vinyl/qwen3_embedding_cds_tidal}
QWEN3_EMBEDDING_MODEL=${QWEN3_EMBEDDING_MODEL:-}
GAIN_CUDA_VISIBLE_DEVICES=${GAIN_CUDA_VISIBLE_DEVICES:-0}
GAIN_PARALLEL_DEVICES=${GAIN_PARALLEL_DEVICES:-$GAIN_CUDA_VISIBLE_DEVICES}
GAIN_NUM_SHARDS=${GAIN_NUM_SHARDS:-auto}
GAIN_ROW_BATCH_SIZE=${GAIN_ROW_BATCH_SIZE:-32}
GAIN_BASELINE_CACHE=${GAIN_BASELINE_CACHE:-$COT_SCORED.baseline_ndcg.jsonl}
GAIN_EMBEDDER_MODE=${GAIN_EMBEDDER_MODE:-qwen3_embedding}
GAIN_MODE=${GAIN_MODE:-ndcg}
GAIN_NDCG_K=${GAIN_NDCG_K:-100}
GRPO_BASELINE_EMBEDDER_MODE=${GRPO_BASELINE_EMBEDDER_MODE:-$GAIN_EMBEDDER_MODE}
GAIN_EMBEDDING_BATCH_SIZE=${GAIN_EMBEDDING_BATCH_SIZE:-8}
GAIN_EMBEDDING_MAX_LENGTH=${GAIN_EMBEDDING_MAX_LENGTH:-8192}
GAIN_EMBEDDING_DEVICE=${GAIN_EMBEDDING_DEVICE:-cuda:0}
GRPO_BASELINE_EMBEDDING_BATCH_SIZE=${GRPO_BASELINE_EMBEDDING_BATCH_SIZE:-8}
GRPO_BASELINE_EMBEDDING_MAX_LENGTH=${GRPO_BASELINE_EMBEDDING_MAX_LENGTH:-8192}
GRPO_BASELINE_EMBEDDING_DEVICE=${GRPO_BASELINE_EMBEDDING_DEVICE:-cuda:0}

MIN_RUBRIC=${MIN_RUBRIC:-0.5}
MIN_GAIN=${MIN_GAIN:-0.0}
FINAL_MIN_GAIN=${FINAL_MIN_GAIN:-0.0}
COT_SCORE_FIELD=${COT_SCORE_FIELD:-cot_gain}
COT_SCORE_BINS=${COT_SCORE_BINS:-40}
TOP_K=${TOP_K:-1}
FALLBACK_WHEN_EMPTY=${FALLBACK_WHEN_EMPTY:-1}
GRPO_EXCLUDE_SFT=${GRPO_EXCLUDE_SFT:-1}

RUN_MERGE=${RUN_MERGE:-1}
RUN_GAIN=${RUN_GAIN:-1}
RUN_SELECT=${RUN_SELECT:-1}
RUN_DATASETS=${RUN_DATASETS:-1}
CLEAN_INTERMEDIATE=${CLEAN_INTERMEDIATE:-0}
CLEAN_GAIN_BASELINE_CACHE=${CLEAN_GAIN_BASELINE_CACHE:-0}
CLEAN_REJECTED_COT=${CLEAN_REJECTED_COT:-0}

require_file() {
  local label="$1"
  local path="$2"
  if [[ ! -s "$path" ]]; then
    echo "Missing or empty $label: $path" >&2
    exit 1
  fi
}

require_path() {
  local label="$1"
  local path="$2"
  if [[ ! -e "$path" ]]; then
    echo "Missing $label: $path" >&2
    exit 1
  fi
}

latest_checkpoint() {
  local dir="$1"
  find "$dir" -type d -name 'checkpoint-*' -print 2>/dev/null | sort -V | tail -n 1
}

prepare_grpo_input() {
  if [[ -s "$GRPO_INPUT" ]]; then
    echo "Using full GRPO train examples: $GRPO_INPUT"
    return
  fi

  local rrec_dataset_dir="$RREC_DATA_ROOT/${CATEGORY}_0_2022-10-2023-10"
  if [[ -d "$rrec_dataset_dir" ]]; then
    echo "Preparing full GRPO train examples from RRec dataset -> $GRPO_INPUT"
    "$PYTHON_BIN" scripts/prepare_rrec_amazon_examples.py \
      --data-root "$RREC_DATA_ROOT" \
      --category "$CATEGORY" \
      --split train \
      --output "$GRPO_INPUT" \
      --max-examples 0 \
      --max-history-items 20 \
      --history-metadata-mode "$HISTORY_METADATA_MODE" \
      --history-max-item-chars "$HISTORY_MAX_ITEM_CHARS"
    return
  fi

  require_file "phase0 train dataset for GRPO input fallback" "$PHASE0_TRAIN_DATASET"
  echo "Converting phase0 train dataset to full GRPO examples -> $GRPO_INPUT"
  "$PYTHON_BIN" - "$PHASE0_TRAIN_DATASET" "$GRPO_INPUT" <<'PY'
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
dst.parent.mkdir(parents=True, exist_ok=True)
count = 0
with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
    for line in fin:
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        category = row.get("category", "CDs_and_Vinyl")
        split = row.get("split", "train")
        interaction_id = row.get("interaction_id", count)
        user_id = row.get("user_id", "")
        out = {
            "example_id": f"{category}:{split}:{interaction_id}:{user_id}",
            "dataset": "rrec-amazon-2023",
            "category": category,
            "split": split,
            "user_id": user_id,
            "interaction_id": interaction_id,
            "target_item_id": row.get("target_item_id"),
            "target_item_title": row.get("target_item_title", ""),
            "target_item_text": row.get("positive", ""),
            "target_rating": row.get("target_rating", 0.0),
            "history_item_count": row.get("history_item_count", 0),
            "user_history": row.get("query", ""),
        }
        fout.write(json.dumps(out, ensure_ascii=False) + "\n")
        count += 1
print(f"wrote {count} full GRPO examples to {dst}")
PY
}

require_path "project root" "$ROOT"
require_path "python" "$PYTHON_BIN"
require_file "candidate lists" "$CANDIDATE_LISTS"
require_file "rubric scores" "$RUBRIC_SCORES"
if [[ "$RUN_GAIN" == "1" && "$GAIN_MODE" == "ndcg" ]]; then
  require_file "gain item_info" "$GAIN_ITEM_INFO"
fi

if [[ "$RUN_GAIN" == "1" || "$RUN_DATASETS" == "1" ]]; then
  if [[ -z "$QWEN3_EMBEDDING_MODEL" ]]; then
    QWEN3_EMBEDDING_MODEL=$(latest_checkpoint "$EMBEDDER_OUT")
  fi
  require_path "Qwen3 embedding checkpoint" "$QWEN3_EMBEDDING_MODEL"
fi

mkdir -p "$OUT_DIR"
cd "$ROOT"

export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export MODELSCOPE_CACHE=${MODELSCOPE_CACHE:-/mnt/tidal-sh01/usr/xiayu6/xiayu/modelscope_cache}
export QWEN3_EMBEDDING_MODEL

echo "ROOT=$ROOT"
echo "CANDIDATE_LISTS=$CANDIDATE_LISTS"
echo "RUBRIC_SCORES=$RUBRIC_SCORES"
echo "QWEN3_EMBEDDING_MODEL=$QWEN3_EMBEDDING_MODEL"
echo "RUN_NAME=$RUN_NAME"
echo "OUT_DIR=$OUT_DIR"
echo "GAIN_MODE=$GAIN_MODE"
echo "GAIN_NDCG_K=$GAIN_NDCG_K"
echo "GAIN_ITEM_INFO=$GAIN_ITEM_INFO"
echo "GAIN_EMBEDDING_DEVICE=$GAIN_EMBEDDING_DEVICE"
echo "GAIN_PARALLEL_DEVICES=$GAIN_PARALLEL_DEVICES"
echo "GAIN_NUM_SHARDS=$GAIN_NUM_SHARDS"
echo "GAIN_ROW_BATCH_SIZE=$GAIN_ROW_BATCH_SIZE"
echo "GAIN_BASELINE_CACHE=$GAIN_BASELINE_CACHE"
echo "FINAL_FILTERED_COT=$FINAL_FILTERED_COT"
echo "COT_SCORE_PLOT=$COT_SCORE_PLOT"
echo "FINAL_MIN_GAIN=$FINAL_MIN_GAIN"
echo "GRPO_INPUT=$GRPO_INPUT"
echo "GRPO_EXCLUDE_SFT=$GRPO_EXCLUDE_SFT"

if [[ "$RUN_MERGE" == "1" ]]; then
  "$PYTHON_BIN" scripts/merge_candidate_list_rubric.py \
    --candidate-lists "$CANDIDATE_LISTS" \
    --rubric-scores "$RUBRIC_SCORES" \
    --output "$COT_JUDGED" \
    --scored-examples-output "$SCORED_EXAMPLES"
else
  echo "Skipping merge: $COT_JUDGED"
fi

if [[ "$RUN_GAIN" == "1" ]]; then
  require_file "merged CoT" "$COT_JUDGED"
  IFS=',' read -r -a gain_devices <<< "$GAIN_PARALLEL_DEVICES"
  if [[ "${#gain_devices[@]}" -eq 0 || -z "${gain_devices[0]}" ]]; then
    gain_devices=(0)
  fi
  if [[ "$GAIN_NUM_SHARDS" == "auto" ]]; then
    gain_num_shards=${#gain_devices[@]}
  else
    gain_num_shards=$GAIN_NUM_SHARDS
  fi

  if [[ "$gain_num_shards" -gt 1 ]]; then
    gain_part_dir=${GAIN_PART_DIR:-$COT_SCORED.parts}
    mkdir -p "$gain_part_dir"
    rm -f "$gain_part_dir"/input-*.jsonl "$gain_part_dir"/scored-*.jsonl

    "$PYTHON_BIN" - "$COT_JUDGED" "$gain_part_dir" "$gain_num_shards" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
out_dir = Path(sys.argv[2])
num_shards = int(sys.argv[3])
files = []
counts = [0 for _ in range(num_shards)]
for idx in range(num_shards):
    path = out_dir / f"input-{idx:05d}-of-{num_shards:05d}.jsonl"
    files.append(path.open("w", encoding="utf-8"))
try:
    with src.open("r", encoding="utf-8") as fin:
        for line_no, line in enumerate(fin, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            key = str(row.get("example_id") or row.get("user_id") or row.get("candidate_id") or row.get("interaction_id") or line_no)
            digest = hashlib.md5(key.encode("utf-8")).digest()
            shard = int.from_bytes(digest[:8], "big") % num_shards
            files[shard].write(json.dumps(row, ensure_ascii=False) + "\n")
            counts[shard] += 1
finally:
    for f in files:
        f.close()
for idx, count in enumerate(counts):
    print(f"gain_shard_input[{idx}]={count}", flush=True)
PY

    pids=()
    for ((shard_idx=0; shard_idx<gain_num_shards; shard_idx++)); do
      device=${gain_devices[$((shard_idx % ${#gain_devices[@]}))]}
      input_part="$gain_part_dir/input-$(printf '%05d' "$shard_idx")-of-$(printf '%05d' "$gain_num_shards").jsonl"
      output_part="$gain_part_dir/scored-$(printf '%05d' "$shard_idx")-of-$(printf '%05d' "$gain_num_shards").jsonl"
      echo "Starting gain shard $shard_idx/$gain_num_shards on CUDA_VISIBLE_DEVICES=$device -> $output_part"
      CUDA_VISIBLE_DEVICES="$device" \
      "$PYTHON_BIN" scripts/compute_cot_gain.py \
        --input "$input_part" \
        --output "$output_part" \
        --embedder-mode "$GAIN_EMBEDDER_MODE" \
        --gain-mode "$GAIN_MODE" \
        --item-info "$GAIN_ITEM_INFO" \
        --ndcg-k "$GAIN_NDCG_K" \
        --model "$MODEL" \
        --embedding-model "$QWEN3_EMBEDDING_MODEL" \
        --embedding-batch-size "$GAIN_EMBEDDING_BATCH_SIZE" \
        --embedding-max-length "$GAIN_EMBEDDING_MAX_LENGTH" \
        --row-batch-size "$GAIN_ROW_BATCH_SIZE" \
        --baseline-cache "$gain_part_dir/baseline-$(printf '%05d' "$shard_idx")-of-$(printf '%05d' "$gain_num_shards").jsonl" \
        --num-shards 1 \
        --shard-index 0 \
        --device "$GAIN_EMBEDDING_DEVICE" &
      pids+=("$!")
    done

    failed=0
    for pid in "${pids[@]}"; do
      if ! wait "$pid"; then
        failed=1
      fi
    done
    if [[ "$failed" != "0" ]]; then
      echo "At least one gain shard failed. Check $gain_part_dir." >&2
      exit 1
    fi

    : > "$COT_SCORED"
    : > "$GAIN_BASELINE_CACHE"
    for ((shard_idx=0; shard_idx<gain_num_shards; shard_idx++)); do
      output_part="$gain_part_dir/scored-$(printf '%05d' "$shard_idx")-of-$(printf '%05d' "$gain_num_shards").jsonl"
      require_file "gain shard output" "$output_part"
      cat "$output_part" >> "$COT_SCORED"
      baseline_part="$gain_part_dir/baseline-$(printf '%05d' "$shard_idx")-of-$(printf '%05d' "$gain_num_shards").jsonl"
      if [[ -s "$baseline_part" ]]; then
        cat "$baseline_part" >> "$GAIN_BASELINE_CACHE"
      fi
    done
    echo "Merged gain shards -> $COT_SCORED"
    echo "Merged baseline cache -> $GAIN_BASELINE_CACHE"
  else
    CUDA_VISIBLE_DEVICES="$GAIN_CUDA_VISIBLE_DEVICES" \
    "$PYTHON_BIN" scripts/compute_cot_gain.py \
      --input "$COT_JUDGED" \
      --output "$COT_SCORED" \
      --embedder-mode "$GAIN_EMBEDDER_MODE" \
      --gain-mode "$GAIN_MODE" \
      --item-info "$GAIN_ITEM_INFO" \
      --ndcg-k "$GAIN_NDCG_K" \
      --model "$MODEL" \
      --embedding-model "$QWEN3_EMBEDDING_MODEL" \
      --embedding-batch-size "$GAIN_EMBEDDING_BATCH_SIZE" \
      --embedding-max-length "$GAIN_EMBEDDING_MAX_LENGTH" \
      --row-batch-size "$GAIN_ROW_BATCH_SIZE" \
      --baseline-cache "$GAIN_BASELINE_CACHE" \
      --device "$GAIN_EMBEDDING_DEVICE"
  fi
else
  echo "Skipping gain: $COT_SCORED"
fi

if [[ "$RUN_SELECT" == "1" ]]; then
  require_file "gain-scored CoT" "$COT_SCORED"
  select_args=()
  if [[ "$FALLBACK_WHEN_EMPTY" == "1" ]]; then
    select_args+=(--fallback-when-empty)
  fi
  "$PYTHON_BIN" scripts/select_filtered_cot.py \
    --input "$COT_SCORED" \
    --output "$FILTERED_COT" \
    --rejected-output "$REJECTED_COT" \
    --top-k "$TOP_K" \
    --min-rubric "$MIN_RUBRIC" \
    --min-gain "$MIN_GAIN" \
    "${select_args[@]}"
else
  echo "Skipping select: $FILTERED_COT"
fi

if [[ "$RUN_DATASETS" == "1" ]]; then
  require_file "filtered CoT" "$FILTERED_COT"
  prepare_grpo_input
  require_file "full GRPO train examples" "$GRPO_INPUT"

  "$PYTHON_BIN" scripts/finalize_cot_selection.py \
    --input "$FILTERED_COT" \
    --output "$FINAL_FILTERED_COT" \
    --plot-output "$COT_SCORE_PLOT" \
    --summary-output "$COT_SCORE_SUMMARY" \
    --score-field "$COT_SCORE_FIELD" \
    --gain-field cot_gain \
    --min-gain "$FINAL_MIN_GAIN" \
    --bins "$COT_SCORE_BINS"

  require_file "final SFT CoT" "$FINAL_FILTERED_COT"
  "$PYTHON_BIN" scripts/make_sft_dataset.py \
    --input "$FINAL_FILTERED_COT" \
    --output "$SFT_DATASET"

  grpo_exclude_args=()
  if [[ "$GRPO_EXCLUDE_SFT" == "1" ]]; then
    grpo_exclude_args+=(--exclude-prompts-from "$SFT_DATASET")
  fi
  "$PYTHON_BIN" scripts/make_grpo_dataset.py \
    --input "$GRPO_INPUT" \
    --output "$GRPO_DATASET" \
    --baseline-mode "$GRPO_BASELINE_EMBEDDER_MODE" \
    --embedding-model "$QWEN3_EMBEDDING_MODEL" \
    --embedding-batch-size "$GRPO_BASELINE_EMBEDDING_BATCH_SIZE" \
    --embedding-max-length "$GRPO_BASELINE_EMBEDDING_MAX_LENGTH" \
    --device "$GRPO_BASELINE_EMBEDDING_DEVICE" \
    "${grpo_exclude_args[@]}"
else
  echo "Skipping dataset build: $SFT_DATASET / $GRPO_DATASET"
fi

if [[ "$CLEAN_INTERMEDIATE" == "1" ]]; then
  gain_part_dir=${GAIN_PART_DIR:-$COT_SCORED.parts}
  if [[ -d "$gain_part_dir" ]]; then
    rm -rf "$gain_part_dir"
    echo "Removed gain shard parts: $gain_part_dir"
  fi
  if [[ "$CLEAN_GAIN_BASELINE_CACHE" == "1" && -f "$GAIN_BASELINE_CACHE" ]]; then
    rm -f "$GAIN_BASELINE_CACHE"
    echo "Removed gain baseline cache: $GAIN_BASELINE_CACHE"
  fi
  if [[ "$CLEAN_REJECTED_COT" == "1" && -f "$REJECTED_COT" ]]; then
    rm -f "$REJECTED_COT"
    echo "Removed rejected CoT file: $REJECTED_COT"
  fi
fi

echo "Rebuilt CoT artifacts:"
echo "  judged:   $COT_JUDGED"
echo "  scored:   $COT_SCORED"
echo "  filtered: $FILTERED_COT"
echo "  final:    $FINAL_FILTERED_COT"
echo "  rejected: $REJECTED_COT"
echo "  examples: $SCORED_EXAMPLES"
echo "  grpo_in:  $GRPO_INPUT"
echo "  plot:     $COT_SCORE_PLOT"
echo "  summary:  $COT_SCORE_SUMMARY"
echo "  sft:      $SFT_DATASET"
echo "  grpo:     $GRPO_DATASET"
