#!/usr/bin/env bash
set -euo pipefail

# Upload the prepared CDs_and_Vinyl training datasets to a dataset-specific
# ModelScope repo. Run this on the machine that already has the artifacts.
#
# Required:
#   MS_DATA_REPO_ID=<namespace>/rrec_amazon_cds_and_vinyl_training_data
#
# Optional:
#   MODELSCOPE_API_TOKEN=<token>   # or MS_TOKEN=<token>; omit after `modelscope login`
#   MS_REPO_ID=<namespace>/repo     # backward-compatible alias for MS_DATA_REPO_ID
#   UPLOAD_EMBEDDING=1             # also upload the trained embedding checkpoint
#   MS_MODEL_REPO_ID=<namespace>/rrec_amazon_cds_and_vinyl_qwen3_embedding_0_6b
#   MS_ENDPOINT=https://www.modelscope.cn
#   UPLOAD_INTERMEDIATE=1          # also upload scored/filtered CoT trace files

ROOT=${ROOT:-/root/autodl-tmp/rec/aaai_pro}
MS_CLI=${MS_CLI:-/root/autodl-tmp/rec/ms-swift-312-cu124-venv/bin/modelscope}
MS_REPO_ID=${MS_REPO_ID:-}
MS_DATA_REPO_ID=${MS_DATA_REPO_ID:-$MS_REPO_ID}
UPLOAD_EMBEDDING=${UPLOAD_EMBEDDING:-0}
MS_MODEL_REPO_ID=${MS_MODEL_REPO_ID:-}
if [[ -z "$MS_DATA_REPO_ID" ]]; then
  echo "Set MS_DATA_REPO_ID, for example: your_namespace/rrec_amazon_cds_and_vinyl_training_data" >&2
  exit 1
fi
if [[ "$UPLOAD_EMBEDDING" == "1" && -z "$MS_MODEL_REPO_ID" ]]; then
  echo "Set MS_MODEL_REPO_ID when UPLOAD_EMBEDDING=1." >&2
  exit 1
fi
MS_ENDPOINT=${MS_ENDPOINT:-https://www.modelscope.cn}
MS_TOKEN=${MS_TOKEN:-${MODELSCOPE_API_TOKEN:-}}
TOKEN_ARGS=()
if [[ -n "$MS_TOKEN" ]]; then
  TOKEN_ARGS=(--token "$MS_TOKEN")
fi

EMBEDDER_DATASET=${EMBEDDER_DATASET:-$ROOT/data/rrec_amazon/phase0_embedder_CDs_and_Vinyl_train.jsonl}
EMBEDDING_CKPT=${EMBEDDING_CKPT:-$ROOT/checkpoints/phase0_qwen3_embedding_CDs_and_Vinyl_len2048_bs128_20260616_150058/checkpoint-83}
EMBEDDING_ARGS=${EMBEDDING_ARGS:-$ROOT/checkpoints/phase0_qwen3_embedding_CDs_and_Vinyl_len2048_bs128_20260616_150058/phase0_args.json}
SFT_DATASET=${SFT_DATASET:-$ROOT/outputs/rrec_amazon/CDs_and_Vinyl/sft_deepseek_v4_pro_cds_embedder_partial.jsonl}
GRPO_DATASET=${GRPO_DATASET:-$ROOT/outputs/rrec_amazon/CDs_and_Vinyl/grpo_deepseek_v4_pro_cds_embedder_disjoint_full.jsonl}
FILTERED_COT=${FILTERED_COT:-$ROOT/outputs/rrec_amazon/CDs_and_Vinyl/filtered_high_quality_cot_deepseek_v4_pro_cds_embedder_partial.jsonl}
COT_SCORED=${COT_SCORED:-$ROOT/outputs/rrec_amazon/CDs_and_Vinyl/cot_scored_deepseek_v4_pro_cds_embedder_partial.jsonl}
MANIFEST=${MANIFEST:-$ROOT/outputs/rrec_amazon/CDs_and_Vinyl/modelscope_upload_manifest_cds.json}
UPLOAD_INTERMEDIATE=${UPLOAD_INTERMEDIATE:-0}

require_path() {
  local path="$1"
  if [[ ! -e "$path" ]]; then
    echo "Required path does not exist: $path" >&2
    exit 1
  fi
}

create_repo_if_needed() {
  local repo_id="$1"
  local repo_type="$2"
  "$MS_CLI" create "$repo_id" \
    --repo_type "$repo_type" \
    --visibility "${MS_VISIBILITY:-private}" \
    --exist_ok \
    --endpoint "$MS_ENDPOINT" \
    "${TOKEN_ARGS[@]}"
}

upload_path() {
  local repo_id="$1"
  local repo_type="$2"
  local local_path="$3"
  local path_in_repo="$4"
  local message="$5"
  require_path "$local_path"
  "$MS_CLI" upload "$repo_id" "$local_path" "$path_in_repo" \
    --repo-type "$repo_type" \
    "${TOKEN_ARGS[@]}" \
    --endpoint "$MS_ENDPOINT" \
    --max-workers "${MS_MAX_WORKERS:-4}" \
    --commit-message "$message"
}

require_path "$MS_CLI"
require_path "$EMBEDDER_DATASET"
require_path "$SFT_DATASET"
require_path "$GRPO_DATASET"
if [[ "$UPLOAD_EMBEDDING" == "1" ]]; then
  require_path "$EMBEDDING_CKPT"
  require_path "$EMBEDDING_ARGS"
fi

mkdir -p "$(dirname "$MANIFEST")"
embedder_rows=$(wc -l < "$EMBEDDER_DATASET" | tr -d ' ')
sft_rows=$(wc -l < "$SFT_DATASET" | tr -d ' ')
grpo_rows=$(wc -l < "$GRPO_DATASET" | tr -d ' ')
cat > "$MANIFEST" <<EOF
{
  "category": "CDs_and_Vinyl",
  "data_repo": "$MS_DATA_REPO_ID",
  "embedder_dataset": "phase0/phase0_embedder_rrec_amazon_cds_and_vinyl_train.jsonl",
  "sft_dataset": "sft/sft_rrec_amazon_cds_and_vinyl_deepseek_v4_pro_cds_embedder_partial.jsonl",
  "grpo_dataset": "grpo/grpo_rrec_amazon_cds_and_vinyl_deepseek_v4_pro_cds_embedder_disjoint_full.jsonl",
  "embedding_training": {
    "base_model": "Qwen/Qwen3-Embedding-0.6B",
    "max_length": 2048,
    "batch_size": 128,
    "epochs": 1,
    "final_step": 83
  },
  "dataset_rows": {
    "embedder": $embedder_rows,
    "sft": $sft_rows,
    "grpo": $grpo_rows
  }
}
EOF

create_repo_if_needed "$MS_DATA_REPO_ID" dataset

upload_path "$MS_DATA_REPO_ID" dataset "$EMBEDDER_DATASET" "phase0/phase0_embedder_rrec_amazon_cds_and_vinyl_train.jsonl" "upload RRec Amazon CDs_and_Vinyl phase0 embedder dataset"
upload_path "$MS_DATA_REPO_ID" dataset "$SFT_DATASET" "sft/sft_rrec_amazon_cds_and_vinyl_deepseek_v4_pro_cds_embedder_partial.jsonl" "upload RRec Amazon CDs_and_Vinyl SFT dataset"
upload_path "$MS_DATA_REPO_ID" dataset "$GRPO_DATASET" "grpo/grpo_rrec_amazon_cds_and_vinyl_deepseek_v4_pro_cds_embedder_disjoint_full.jsonl" "upload RRec Amazon CDs_and_Vinyl GRPO dataset"

if [[ "$UPLOAD_EMBEDDING" == "1" ]]; then
  create_repo_if_needed "$MS_MODEL_REPO_ID" model
  upload_path "$MS_MODEL_REPO_ID" model "$EMBEDDING_CKPT" "qwen3_embedding_0_6b_rrec_amazon_cds_and_vinyl/checkpoint-83" "upload RRec Amazon CDs_and_Vinyl Qwen3 embedding checkpoint"
  upload_path "$MS_MODEL_REPO_ID" model "$EMBEDDING_ARGS" "qwen3_embedding_0_6b_rrec_amazon_cds_and_vinyl/phase0_args.json" "upload RRec Amazon CDs_and_Vinyl embedding training args"
fi

if [[ "$UPLOAD_INTERMEDIATE" == "1" ]]; then
  if [[ -e "$FILTERED_COT" ]]; then
    upload_path "$MS_DATA_REPO_ID" dataset "$FILTERED_COT" "intermediate/filtered_high_quality_cot_rrec_amazon_cds_and_vinyl_deepseek_v4_pro_cds_embedder_partial.jsonl" "upload RRec Amazon CDs_and_Vinyl filtered CoT data"
  fi
  if [[ -e "$COT_SCORED" ]]; then
    upload_path "$MS_DATA_REPO_ID" dataset "$COT_SCORED" "intermediate/cot_scored_rrec_amazon_cds_and_vinyl_deepseek_v4_pro_cds_embedder_partial.jsonl" "upload RRec Amazon CDs_and_Vinyl scored CoT data"
  fi
fi

upload_path "$MS_DATA_REPO_ID" dataset "$MANIFEST" "MANIFEST.rrec_amazon_cds_and_vinyl.json" "upload RRec Amazon CDs_and_Vinyl artifact manifest"

echo "Uploaded CDs datasets to ModelScope dataset repo: $MS_DATA_REPO_ID"
if [[ "$UPLOAD_EMBEDDING" == "1" ]]; then
  echo "Uploaded CDs embedding to ModelScope model repo: $MS_MODEL_REPO_ID"
fi
