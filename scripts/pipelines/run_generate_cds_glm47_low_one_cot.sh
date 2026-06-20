#!/usr/bin/env bash
set -euo pipefail

# Generate one GLM CodePlan CoT candidate for every CDs_and_Vinyl train example.
# The generator writes an append-only candidate checkpoint next to the final JSONL,
# so interrupted runs can resume without repeating completed API calls.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -d "$SCRIPT_DIR/../rubric_cot_pipeline" ]]; then
  REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
else
  REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
fi

ROOT=${ROOT:-$REPO_ROOT}
VENV=${VENV:-}
if [[ -z "${PYTHON_BIN:-}" ]]; then
  if [[ -n "$VENV" && -x "$VENV/bin/python" ]]; then
    PYTHON_BIN="$VENV/bin/python"
  else
    PYTHON_BIN=python3
  fi
fi
CONFIG_FILE=${CONFIG_FILE:-$ROOT/configs/glm_codeplan.env}

CATEGORY=${CATEGORY:-CDs_and_Vinyl}
RREC_DATA_ROOT=${RREC_DATA_ROOT:-$ROOT/data}
DATA_DIR=${DATA_DIR:-$ROOT/data/rrec_amazon/$CATEGORY}
OUT_DIR=${OUT_DIR:-$ROOT/outputs/rrec_amazon/$CATEGORY}
EXAMPLES_FILE=${EXAMPLES_FILE:-$DATA_DIR/examples.jsonl}
PHASE0_DATASET=${PHASE0_DATASET:-$ROOT/github_artifacts/CDs_and_Vinyl/phase0/phase0_embedder_rrec_amazon_cds_and_vinyl_train.jsonl}
OUTPUT=${OUTPUT:-$OUT_DIR/cot_candidate_lists_glm47_low_one_train.jsonl}

API_PROVIDER=${API_PROVIDER:-glm_codeplan}
API_BASE_URL=${API_BASE_URL:-https://open.bigmodel.cn/api/coding/paas/v4}
API_MODEL=${API_MODEL:-glm-4.7}
API_THINKING=${API_THINKING:-enabled}
API_REASONING_EFFORT=${API_REASONING_EFFORT:-low}
API_TIMEOUT=${API_TIMEOUT:-300}
API_MAX_RETRIES=${API_MAX_RETRIES:-3}
API_MIN_INTERVAL=${API_MIN_INTERVAL:-0.2}
MAX_WORKERS=${MAX_WORKERS:-8}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS:-2048}
TOP_P=${TOP_P:-0.9}
TEMPERATURES=${TEMPERATURES:-0.6}
AGGREGATE_EVERY=${AGGREGATE_EVERY:-100}
MAX_EXAMPLES=${MAX_EXAMPLES:-0}
RESUME_FLAG=${RESUME_FLAG:---resume}

require_path() {
  local label="$1"
  local path="$2"
  if [[ ! -e "$path" ]]; then
    echo "Missing $label: $path" >&2
    exit 1
  fi
}

require_file() {
  local label="$1"
  local path="$2"
  if [[ ! -s "$path" ]]; then
    echo "Missing or empty $label: $path" >&2
    exit 1
  fi
}

require_path "project root" "$ROOT"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing python executable: $PYTHON_BIN" >&2
  exit 1
fi

if [[ -f "$CONFIG_FILE" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$CONFIG_FILE"
  set +a
fi

if [[ -z "${BIGMODEL_API_KEY:-}" && -z "${COT_GENERATION_API_KEY:-}" ]]; then
  echo "Missing API key. Put BIGMODEL_API_KEY in $CONFIG_FILE or export COT_GENERATION_API_KEY." >&2
  exit 2
fi

mkdir -p "$DATA_DIR" "$OUT_DIR"
cd "$ROOT"

if [[ -n "$VENV" ]]; then
  export PATH="$VENV/bin:$PATH"
fi
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"

if [[ ! -s "$EXAMPLES_FILE" ]]; then
  rrec_dataset_dir="$RREC_DATA_ROOT/${CATEGORY}_0_2022-10-2023-10"
  if [[ -d "$rrec_dataset_dir" ]]; then
    echo "Preparing all train examples -> $EXAMPLES_FILE"
    "$PYTHON_BIN" scripts/prepare_rrec_amazon_examples.py \
      --data-root "$RREC_DATA_ROOT" \
      --category "$CATEGORY" \
      --split train \
      --output "$EXAMPLES_FILE" \
      --max-examples 0 \
      --max-history-items 20
  else
    require_file "phase0 train dataset" "$PHASE0_DATASET"
    echo "Converting phase0 train dataset -> $EXAMPLES_FILE"
    "$PYTHON_BIN" - "$PHASE0_DATASET" "$EXAMPLES_FILE" <<'PY'
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
print(f"wrote {count} examples to {dst}")
PY
  fi
else
  echo "Using existing examples: $EXAMPLES_FILE"
fi

require_file "train examples" "$EXAMPLES_FILE"

echo "Generating one CoT candidate per train example"
echo "  model:        $API_MODEL"
echo "  effort:       $API_REASONING_EFFORT"
echo "  examples:     $EXAMPLES_FILE"
echo "  output:       $OUTPUT"
echo "  max_workers:  $MAX_WORKERS"
echo "  max_examples: $MAX_EXAMPLES"

"$PYTHON_BIN" scripts/generate_cot_candidate_lists.py \
  --input "$EXAMPLES_FILE" \
  --output "$OUTPUT" \
  --max-examples "$MAX_EXAMPLES" \
  --num-candidates 1 \
  --temperatures "$TEMPERATURES" \
  --max-workers "$MAX_WORKERS" \
  --aggregate-every "$AGGREGATE_EVERY" \
  $RESUME_FLAG \
  --api-provider "$API_PROVIDER" \
  --api-base-url "$API_BASE_URL" \
  --api-model "$API_MODEL" \
  --api-timeout "$API_TIMEOUT" \
  --api-max-retries "$API_MAX_RETRIES" \
  --api-min-interval "$API_MIN_INTERVAL" \
  --api-thinking "$API_THINKING" \
  --api-reasoning-effort "$API_REASONING_EFFORT" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --max-prompt-tokens "$MAX_PROMPT_TOKENS" \
  --top-p "$TOP_P"

echo "Done: $OUTPUT"
