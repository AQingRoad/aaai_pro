#!/usr/bin/env bash
set -euo pipefail

# Build Video_Games no-all-ratings metadata examples, then generate one tagged
# GLM-4.7 CoT candidate per example through the API.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

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
if [[ -f "$CONFIG_FILE" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$CONFIG_FILE"
  set +a
fi

CATEGORY=${CATEGORY:-Video_Games}
SPLIT=${SPLIT:-train}
if [[ -z "${RREC_DATA_ROOT:-}" ]]; then
  if [[ -d /root/autodl-tmp/rec/RRec_official/data ]]; then
    RREC_DATA_ROOT=/root/autodl-tmp/rec/RRec_official/data
  else
    RREC_DATA_ROOT=$ROOT/data
  fi
fi
DATASET_DIR=${DATASET_DIR:-}
SOURCE_EXAMPLES_JSONL=${SOURCE_EXAMPLES_JSONL:-}
ITEM_INFO=${ITEM_INFO:-}
DATA_DIR=${DATA_DIR:-$ROOT/data/rrec_amazon/$CATEGORY}
OUT_DIR=${OUT_DIR:-$ROOT/outputs/rrec_amazon/$CATEGORY}
EXAMPLES_FILE=${EXAMPLES_FILE:-$DATA_DIR/examples_meta_compact_no_all_ratings.jsonl}
OUTPUT=${OUTPUT:-$OUT_DIR/cot_candidate_lists_glm47_meta_compact_no_all_ratings_one_${SPLIT}_raw.jsonl}
REBUILD_EXAMPLES=${REBUILD_EXAMPLES:-0}
PREPARE_ONLY=${PREPARE_ONLY:-0}
DRY_RUN=${DRY_RUN:-0}

API_PROVIDER=${API_PROVIDER:-glm_codeplan}
API_BASE_URL=${API_BASE_URL:-https://open.bigmodel.cn/api/coding/paas/v4}
API_MODEL=${API_MODEL:-glm-4.7}
API_THINKING=${API_THINKING:-disabled}
API_REASONING_EFFORT=${API_REASONING_EFFORT:-}
API_TIMEOUT=${API_TIMEOUT:-300}
API_MAX_RETRIES=${API_MAX_RETRIES:-3}
API_MIN_INTERVAL=${API_MIN_INTERVAL:-0.2}
MAX_WORKERS=${MAX_WORKERS:-8}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
TOP_P=${TOP_P:-0.9}
TEMPERATURES=${TEMPERATURES:-0.6}
AGGREGATE_EVERY=${AGGREGATE_EVERY:-100}
MAX_EXAMPLES=${MAX_EXAMPLES:-0}
RESUME_FLAG=${RESUME_FLAG:---resume}

COT_OUTPUT_FORMAT=${COT_OUTPUT_FORMAT:-tagged}
MAX_OUTPUT_WORDS=${MAX_OUTPUT_WORDS:-1024}
RATING_CONTEXT=${RATING_CONTEXT:-no_rating}
MIN_ANSWER_WORDS=${MIN_ANSWER_WORDS:-0}
MAX_ANSWER_WORDS=${MAX_ANSWER_WORDS:-0}
RECORD_API_RAW=${RECORD_API_RAW:-0}
REQUIRE_LITERAL_TAGS=${REQUIRE_LITERAL_TAGS:-0}

MAX_HISTORY_ITEMS=${MAX_HISTORY_ITEMS:-20}
MIN_HISTORY=${MIN_HISTORY:-1}
MIN_RATING=${MIN_RATING:-0}
MAX_TARGET_CHARS=${MAX_TARGET_CHARS:-0}
HISTORY_METADATA_MODE=${HISTORY_METADATA_MODE:-compact}
HISTORY_MAX_ITEM_CHARS=${HISTORY_MAX_ITEM_CHARS:-0}
HISTORY_INCLUDE_RATINGS=${HISTORY_INCLUDE_RATINGS:-0}
HISTORY_INCLUDE_CATALOG_STATS=${HISTORY_INCLUDE_CATALOG_STATS:-0}
STRIP_RATING_FIELDS=${STRIP_RATING_FIELDS:-1}
ITEM_METADATA_SUMMARY=${ITEM_METADATA_SUMMARY:-}

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

bool_arg() {
  local value="$1"
  local yes_arg="$2"
  local no_arg="$3"
  local lc
  lc="$(printf '%s' "$value" | tr '[:upper:]' '[:lower:]')"
  case "$lc" in
    1|true|yes|on) printf '%s\n' "$yes_arg" ;;
    *) printf '%s\n' "$no_arg" ;;
  esac
}

print_param() {
  local key="$1"
  local value="$2"
  local note="$3"
  printf '%-32s %s\n    %s\n' "$key=$value" "" "$note"
}

require_path "project root" "$ROOT"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1 && [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Missing python executable: $PYTHON_BIN" >&2
  exit 1
fi
if [[ -z "${BIGMODEL_API_KEY:-}" && -z "${COT_GENERATION_API_KEY:-}" ]]; then
  echo "Missing API key. Put BIGMODEL_API_KEY in $CONFIG_FILE or export COT_GENERATION_API_KEY." >&2
  exit 2
fi

mkdir -p "$DATA_DIR" "$OUT_DIR"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
if [[ -n "$VENV" ]]; then
  export PATH="$VENV/bin:$PATH"
fi

history_rating_arg="$(bool_arg "$HISTORY_INCLUDE_RATINGS" "--history-include-ratings" "--no-history-include-ratings")"
catalog_stats_arg="$(bool_arg "$HISTORY_INCLUDE_CATALOG_STATS" "--history-include-catalog-stats" "--no-history-include-catalog-stats")"
strip_rating_arg="$(bool_arg "$STRIP_RATING_FIELDS" "--strip-rating-fields" "--no-strip-rating-fields")"
record_api_raw_arg="$(bool_arg "$RECORD_API_RAW" "--record-api-raw" "--no-record-api-raw")"
require_literal_tags_arg="$(bool_arg "$REQUIRE_LITERAL_TAGS" "--require-literal-tags" "--no-require-literal-tags")"

echo "Video_Games no-all-ratings API CoT generation parameters"
print_param "ROOT" "$ROOT" "项目根目录；脚本从这里读取代码、配置、data 和 outputs。"
print_param "CATEGORY" "$CATEGORY" "数据集类别；本脚本默认构建 Amazon Video_Games。"
print_param "SPLIT" "$SPLIT" "构建和生成的 split；默认 train。"
print_param "RREC_DATA_ROOT" "$RREC_DATA_ROOT" "RRec official HuggingFace dataset 根目录；用于重建带 compact metadata 的 examples。"
print_param "DATASET_DIR" "${DATASET_DIR:-auto}" "可选单类别 dataset 目录；为空时使用 RREC_DATA_ROOT/CATEGORY_0_2022-10-2023-10。"
print_param "SOURCE_EXAMPLES_JSONL" "${SOURCE_EXAMPLES_JSONL:-auto}" "当 official dataset 不存在时的 JSONL fallback；需要配合 ITEM_INFO。"
print_param "ITEM_INFO" "${ITEM_INFO:-auto}" "JSONL fallback 的 item_info；用于 history metadata 和 target item text。"
print_param "EXAMPLES_FILE" "$EXAMPLES_FILE" "输出的 no-all-ratings examples；API prompt 只读取这里的 user_history 和 category。"
print_param "OUTPUT" "$OUTPUT" "API 生成的 CoT candidate list 输出；旁边会生成 .candidates 和 .failures checkpoint。"
print_param "REBUILD_EXAMPLES" "$REBUILD_EXAMPLES" "为 1 时强制重建 examples；为 0 时复用已有非空 examples。"
print_param "MAX_HISTORY_ITEMS" "$MAX_HISTORY_ITEMS" "每个用户最多保留最近多少个历史 item。"
print_param "HISTORY_METADATA_MODE" "$HISTORY_METADATA_MODE" "history item 文本格式；compact 对齐 CDs no-all-ratings 参考文件。"
print_param "HISTORY_MAX_ITEM_CHARS" "$HISTORY_MAX_ITEM_CHARS" "history 单个 item 字符截断；0 表示不做字符级截断。"
print_param "HISTORY_INCLUDE_RATINGS" "$HISTORY_INCLUDE_RATINGS" "是否在 history 中写入用户星级评分；no-all-ratings 固定为 0。"
print_param "HISTORY_INCLUDE_CATALOG_STATS" "$HISTORY_INCLUDE_CATALOG_STATS" "是否写入 avg_rating/rating_count 等 catalog 统计；no-all-ratings 固定为 0。"
print_param "STRIP_RATING_FIELDS" "$STRIP_RATING_FIELDS" "是否删除 target_rating/history_rating/rating 字段；no-all-ratings 固定为 1。"
print_param "MAX_TARGET_CHARS" "$MAX_TARGET_CHARS" "target_item_text 字符截断；0 表示不截断，便于后续 embedding positive 保持完整。"
print_param "API_PROVIDER" "$API_PROVIDER" "API provider；glm_codeplan 走 BigModel CodePlan 兼容接口。"
print_param "API_MODEL" "$API_MODEL" "生成 CoT 的模型；默认 glm-4.7，对齐 CDs GLM-4.7 配置。"
print_param "API_THINKING" "$API_THINKING" "API thinking 开关；disabled 表示只使用显式 tagged 输出。"
print_param "COT_OUTPUT_FORMAT" "$COT_OUTPUT_FORMAT" "CoT 保存格式；tagged 会保存 <think>/<answer> 规范化字段。"
print_param "RATING_CONTEXT" "$RATING_CONTEXT" "prompt 和输出校验的评分上下文；no_rating 禁止 rating/review/popularity 语言。"
print_param "RECORD_API_RAW" "$RECORD_API_RAW" "是否落盘 API 原始 request/response；默认 0，减少字段泄漏排查以外的落盘风险。"
print_param "MAX_WORKERS" "$MAX_WORKERS" "并发 API 请求数；过大可能触发限流或增加失败率。"
print_param "API_MIN_INTERVAL" "$API_MIN_INTERVAL" "单进程 API 请求最小间隔秒数；配合并发控制请求速率。"
print_param "MAX_EXAMPLES" "$MAX_EXAMPLES" "最多生成多少条；0 表示当前 split 全量，调试可设为 20。"
print_param "DRY_RUN" "$DRY_RUN" "为 1 时只打印参数，不构建 examples 或调用 API。"

if [[ "$DRY_RUN" == "1" ]]; then
  exit 0
fi

prepare_args=(
  scripts/data/prepare_rrec_amazon_examples.py
  --category "$CATEGORY"
  --split "$SPLIT"
  --output "$EXAMPLES_FILE"
  --max-examples 0
  --max-history-items "$MAX_HISTORY_ITEMS"
  --min-history "$MIN_HISTORY"
  --min-rating "$MIN_RATING"
  --max-target-chars "$MAX_TARGET_CHARS"
  --history-metadata-mode "$HISTORY_METADATA_MODE"
  --history-max-item-chars "$HISTORY_MAX_ITEM_CHARS"
  --item-summary "$ITEM_METADATA_SUMMARY"
  "$history_rating_arg"
  "$catalog_stats_arg"
  "$strip_rating_arg"
)
if [[ -n "$DATASET_DIR" ]]; then
  prepare_args+=(--dataset-dir "$DATASET_DIR")
else
  prepare_args+=(--data-root "$RREC_DATA_ROOT")
fi
if [[ -n "$SOURCE_EXAMPLES_JSONL" ]]; then
  prepare_args+=(--examples-jsonl "$SOURCE_EXAMPLES_JSONL")
fi
if [[ -n "$ITEM_INFO" ]]; then
  prepare_args+=(--item-info "$ITEM_INFO")
fi

if [[ "$REBUILD_EXAMPLES" == "1" || ! -s "$EXAMPLES_FILE" ]]; then
  echo "Preparing no-all-ratings examples -> $EXAMPLES_FILE"
  "$PYTHON_BIN" "${prepare_args[@]}"
else
  echo "Using existing no-all-ratings examples: $EXAMPLES_FILE"
fi
require_file "no-all-ratings examples" "$EXAMPLES_FILE"

if [[ "$PREPARE_ONLY" == "1" ]]; then
  echo "PREPARE_ONLY=1, skip API generation."
  exit 0
fi

echo "Generating one GLM CoT candidate per example -> $OUTPUT"
"$PYTHON_BIN" scripts/cot/generate_cot_candidate_lists.py \
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
  --cot-output-format "$COT_OUTPUT_FORMAT" \
  --max-output-words "$MAX_OUTPUT_WORDS" \
  --rating-context "$RATING_CONTEXT" \
  --min-answer-words "$MIN_ANSWER_WORDS" \
  --max-answer-words "$MAX_ANSWER_WORDS" \
  "$record_api_raw_arg" \
  "$require_literal_tags_arg" \
  --max-new-tokens "$MAX_NEW_TOKENS" \
  --top-p "$TOP_P"

echo "Done: $OUTPUT"
