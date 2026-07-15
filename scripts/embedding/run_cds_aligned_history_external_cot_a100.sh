#!/usr/bin/env bash
set -euo pipefail

# Strict paired experiment. Both retrievers share clean title/store/categories
# histories, rows, positives, order, optimizer, loss, epochs, and evaluation.
# The only query-side change is the appended target-free tagged CoT.

ROOT=${ROOT:-/home/user/aaai_pro}
VENV=${VENV:-/home/user/.conda/envs/aaai_pro}
PYTHON_BIN=${PYTHON_BIN:-$VENV/bin/python}
CATEGORY=${CATEGORY:-CDs_and_Vinyl}
BASE_MODEL=${BASE_MODEL:-/home/user/models_hf/Qwen3-Embedding-0.6B}
CONFIG_FILE=${CONFIG_FILE:-$ROOT/configs/glm_codeplan.env}
if [[ -f "$CONFIG_FILE" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$CONFIG_FILE"
  set +a
fi

TRAIN_SOURCE=${TRAIN_SOURCE:-$ROOT/data/rrec_amazon/CDs_and_Vinyl/examples.jsonl}
VALID_SOURCE=${VALID_SOURCE:-$ROOT/github_artifacts/CDs_and_Vinyl/rrec_eval/valid.jsonl}
TEST_SOURCE=${TEST_SOURCE:-$ROOT/github_artifacts/CDs_and_Vinyl/rrec_eval/test.jsonl}
ITEM_INFO=${ITEM_INFO:-$ROOT/github_artifacts/CDs_and_Vinyl/rrec_eval/item_info.jsonl}

RUN_ID=${RUN_ID:-cds_aligned_tsc_glm47_singlepos_lr2e5_ep4_seed42_v1}
DATA_ROOT=${DATA_ROOT:-$ROOT/ablantion/datas/processed_datas/$RUN_ID}
HISTORY_DIR=${HISTORY_DIR:-$DATA_ROOT/history}
GEN_INPUT_DIR=${GEN_INPUT_DIR:-$DATA_ROOT/cot_generation_inputs}
COT_API_DIR=${COT_API_DIR:-$DATA_ROOT/cot/api}
COT_PAIR_DIR=${COT_PAIR_DIR:-$DATA_ROOT/cot/training}
AUDIT_DIR=${AUDIT_DIR:-$DATA_ROOT/audits}
REPORT_DIR=${REPORT_DIR:-$ROOT/experiments/results/$RUN_ID}

HISTORY_TRAIN=${HISTORY_TRAIN:-$HISTORY_DIR/train.jsonl}
HISTORY_VALID=${HISTORY_VALID:-$HISTORY_DIR/valid.jsonl}
HISTORY_TEST=${HISTORY_TEST:-$HISTORY_DIR/test.jsonl}
GEN_TRAIN=${GEN_TRAIN:-$GEN_INPUT_DIR/train.jsonl}
GEN_VALID=${GEN_VALID:-$GEN_INPUT_DIR/valid.jsonl}
GEN_TEST=${GEN_TEST:-$GEN_INPUT_DIR/test.jsonl}
COT_RAW_TRAIN=${COT_RAW_TRAIN:-$COT_API_DIR/train_glm47_tagged.jsonl}
COT_RAW_VALID=${COT_RAW_VALID:-$COT_API_DIR/valid_glm47_tagged.jsonl}
COT_RAW_TEST=${COT_RAW_TEST:-$COT_API_DIR/test_glm47_tagged.jsonl}
COT_TRAIN=${COT_TRAIN:-$COT_PAIR_DIR/train.jsonl}
COT_VALID=${COT_VALID:-$COT_PAIR_DIR/valid.jsonl}
COT_TEST=${COT_TEST:-$COT_PAIR_DIR/test.jsonl}

HISTORY_RUN_NAME=${HISTORY_RUN_NAME:-${RUN_ID}_history}
COT_RUN_NAME=${COT_RUN_NAME:-${RUN_ID}_external_cot}
HISTORY_OUT=${HISTORY_OUT:-$ROOT/checkpoints/rrec_amazon_CDs_and_Vinyl/$HISTORY_RUN_NAME}
COT_OUT=${COT_OUT:-$ROOT/checkpoints/rrec_amazon_CDs_and_Vinyl/$COT_RUN_NAME}
HISTORY_EVAL=${HISTORY_EVAL:-$ROOT/outputs/rrec_amazon/eval/$CATEGORY/$HISTORY_RUN_NAME}
COT_EVAL=${COT_EVAL:-$ROOT/outputs/rrec_amazon/eval/$CATEGORY/$COT_RUN_NAME}
HISTORY_BEST=${HISTORY_BEST:-$REPORT_DIR/history_best_valid.json}
COT_BEST=${COT_BEST:-$REPORT_DIR/cot_best_valid.json}

API_PROVIDER=${API_PROVIDER:-glm_codeplan}
API_BASE_URL=${API_BASE_URL:-https://open.bigmodel.cn/api/coding/paas/v4}
API_MODEL=${API_MODEL:-glm-4.7}
API_THINKING=${API_THINKING:-disabled}
API_TIMEOUT=${API_TIMEOUT:-180}
API_MAX_RETRIES=${API_MAX_RETRIES:-3}
API_MAX_WORKERS=${API_MAX_WORKERS:-4}
API_MIN_INTERVAL=${API_MIN_INTERVAL:-0}
COT_OUTPUT_FORMAT=${COT_OUTPUT_FORMAT:-tagged}
TEMPERATURE=${TEMPERATURE:-0.6}
TOP_P=${TOP_P:-0.9}
MAX_OUTPUT_WORDS=${MAX_OUTPUT_WORDS:-1024}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2048}
NUM_CANDIDATES=${NUM_CANDIDATES:-1}
RATING_CONTEXT=${RATING_CONTEXT:-no_rating}
RECORD_API_RAW=${RECORD_API_RAW:-0}

CUDA_DEVICES=${CUDA_DEVICES:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
MASTER_PORT_HISTORY=${MASTER_PORT_HISTORY:-29551}
MASTER_PORT_COT=${MASTER_PORT_COT:-29552}
BATCH_SIZE=${BATCH_SIZE:-128}
GRAD_ACCUM=${GRAD_ACCUM:-1}
MAX_LENGTH=${MAX_LENGTH:-4096}
EPOCHS=${EPOCHS:-4}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
WARMUP_RATIO=${WARMUP_RATIO:-0.03}
TEMPERATURE_LOSS=${TEMPERATURE_LOSS:-0.05}
SAVE_STEPS=${SAVE_STEPS:-83}
TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-auto}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-flash_attention_2}
MULTI_POSITIVE_TARGETS=${MULTI_POSITIVE_TARGETS:-0}
CROSS_GPU_NEGATIVES=${CROSS_GPU_NEGATIVES:-0}
SEED=${SEED:-42}

EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-256}
EVAL_MAX_LENGTH=${EVAL_MAX_LENGTH:-4096}
EVAL_DEVICE=${EVAL_DEVICE:-cuda:0}
SELECTION_METRIC=${SELECTION_METRIC:-NDCG@20}
KS=${KS:-5,10,20}
RUN_TEST=${RUN_TEST:-1}

FORCE_DATA=${FORCE_DATA:-0}
FORCE_GENERATION=${FORCE_GENERATION:-0}
FORCE_TRAIN=${FORCE_TRAIN:-0}
FORCE_EVAL=${FORCE_EVAL:-0}
RUN_GENERATION=${RUN_GENERATION:-1}
RUN_TRAIN=${RUN_TRAIN:-1}
PRINT_ARGS_ONLY=${PRINT_ARGS_ONLY:-0}
CONFIRM_RUN=${CONFIRM_RUN:-0}

NCCL_NET=${NCCL_NET:-Socket}
NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
NCCL_MNNVL_ENABLE=${NCCL_MNNVL_ENABLE:-0}
NCCL_COLLNET_ENABLE=${NCCL_COLLNET_ENABLE:-0}
NCCL_DEBUG=${NCCL_DEBUG:-WARN}
TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

enabled() {
  [[ "$1" == "1" || "$1" == "true" ]]
}

print_param() {
  printf '%s=%s # %s\n' "$1" "$2" "$3"
}

require_file() {
  if [[ ! -s "$2" ]]; then
    echo "缺少或为空 $1: $2" >&2
    exit 1
  fi
}

require_path() {
  if [[ ! -e "$2" ]]; then
    echo "缺少 $1: $2" >&2
    exit 1
  fi
}

print_plan() {
  echo "CDs history 与 external-CoT 严格对齐实验参数："
  print_param ROOT "$ROOT" "项目根目录；运行前必须读取 AGENTS.md。"
  print_param VENV "$VENV" "A100 Conda 环境。"
  print_param BASE_MODEL "$BASE_MODEL" "两组共同的 Qwen3-Embedding-0.6B 初始权重。"
  print_param CONFIG_FILE "$CONFIG_FILE" "GLM API 密钥配置文件；只读取，不输出密钥。"
  print_param TRAIN_SOURCE "$TRAIN_SOURCE" "官方 train split，10722 条。"
  print_param VALID_SOURCE "$VALID_SOURCE" "官方 valid split，1340 条，只用于选模。"
  print_param TEST_SOURCE "$TEST_SOURCE" "官方 test split，1341 条；方案冻结后评测一次。"
  print_param ITEM_INFO "$ITEM_INFO" "历史字段、完整 positive 和 12000 个候选物品来源。"
  print_param DATA_ROOT "$DATA_ROOT" "严格配对数据、CoT raw 和审计文件目录。"
  print_param QUERY_FIELDS "title,store,categories" "两组共同的 history 字段。"
  print_param HISTORY_MAX_ITEM_CHARS 0 "history item 文本不做字符截断。"
  print_param MAX_TARGET_CHARS 0 "positive 目标文本不做字符截断。"
  print_param MAX_ITEM_CHARS 0 "评测候选 item 文本不做字符截断。"
  print_param HISTORY_TRAIN "$HISTORY_TRAIN" "history-only train pair。"
  print_param HISTORY_VALID "$HISTORY_VALID" "history-only valid pair。"
  print_param HISTORY_TEST "$HISTORY_TEST" "history-only test pair。"
  print_param COT_TRAIN "$COT_TRAIN" "query 只比 HISTORY_TRAIN 多完整 tagged CoT。"
  print_param COT_VALID "$COT_VALID" "query 只比 HISTORY_VALID 多完整 tagged CoT。"
  print_param COT_TEST "$COT_TEST" "query 只比 HISTORY_TEST 多完整 tagged CoT。"
  print_param API_PROVIDER "$API_PROVIDER" "GLM OpenAI-compatible API provider。"
  print_param API_BASE_URL "$API_BASE_URL" "GLM API 地址。"
  print_param API_MODEL "$API_MODEL" "train、valid、test 共用的 external CoT 生成模型。"
  print_param API_KEY_CONFIGURED "$( [[ -n "${COT_GENERATION_API_KEY:-}" || -n "${BIGMODEL_API_KEY:-}" || -n "${ZAI_API_KEY:-}" ]] && echo yes || echo no )" "只显示密钥是否存在，不打印密钥内容。"
  print_param API_THINKING "$API_THINKING" "关闭 API 内置 thinking，只读取显式 tagged 输出。"
  print_param COT_OUTPUT_FORMAT "$COT_OUTPUT_FORMAT" "生成 <think>/<answer> 规范化 CoT。"
  print_param RATING_CONTEXT "$RATING_CONTEXT" "所有 split 都按无评分 observed history 生成。"
  print_param TEMPERATURE "$TEMPERATURE" "每条 history 的唯一 CoT 采样温度。"
  print_param TOP_P "$TOP_P" "GLM nucleus sampling 参数。"
  print_param NUM_CANDIDATES "$NUM_CANDIDATES" "每条 history 只生成一条 CoT。"
  print_param MAX_OUTPUT_WORDS "$MAX_OUTPUT_WORDS" "tagged think 与 answer 的总词数上限。"
  print_param MAX_NEW_TOKENS "$MAX_NEW_TOKENS" "API 最大输出 token 数。"
  print_param RECORD_API_RAW "$RECORD_API_RAW" "默认不保存完整 API raw response。"
  print_param HISTORY_OUT "$HISTORY_OUT" "clean history-only checkpoint 输出。"
  print_param COT_OUT "$COT_OUT" "严格配对 external-CoT checkpoint 输出。"
  print_param LOSS "single-positive InfoNCE" "两组每行只把同位置 positive 作为正例。"
  print_param MULTI_POSITIVE_TARGETS "$MULTI_POSITIVE_TARGETS" "固定为 0，保持 single-positive。"
  print_param BATCH_SIZE "$BATCH_SIZE" "两组共同的单卡 batch size。"
  print_param GRAD_ACCUM "$GRAD_ACCUM" "两组共同的梯度累积步数。"
  print_param GLOBAL_BATCH_SIZE "$((BATCH_SIZE * GRAD_ACCUM * NPROC_PER_NODE))" "每次更新覆盖的 query 数。"
  print_param MAX_LENGTH "$MAX_LENGTH" "训练 query 和 positive 的 token 上限；预审计要求超限数为 0。"
  print_param EPOCHS "$EPOCHS" "两组都训练 4 轮。"
  print_param LEARNING_RATE "$LEARNING_RATE" "两组共同的 AdamW 学习率。"
  print_param WEIGHT_DECAY "$WEIGHT_DECAY" "两组共同的权重衰减。"
  print_param WARMUP_RATIO "$WARMUP_RATIO" "两组共同的 warmup 比例。"
  print_param TEMPERATURE_LOSS "$TEMPERATURE_LOSS" "InfoNCE 相似度温度。"
  print_param SAVE_STEPS "$SAVE_STEPS" "每 83 step 保存一次，约每轮一个 checkpoint。"
  print_param SEED "$SEED" "数据顺序、DataLoader、初始化、训练和评测统一随机种子。"
  print_param ATTN_IMPLEMENTATION "$ATTN_IMPLEMENTATION" "两组共同使用 FlashAttention 2。"
  print_param PYTORCH_CUDA_ALLOC_CONF "$PYTORCH_CUDA_ALLOC_CONF" "减少长序列显存碎片。"
  print_param CUDA_DEVICES "$CUDA_DEVICES" "训练和评测使用的 A100 GPU。"
  print_param EVAL_BATCH_SIZE "$EVAL_BATCH_SIZE" "两组共同的评测编码 batch size。"
  print_param EVAL_MAX_LENGTH "$EVAL_MAX_LENGTH" "评测 token 上限，与训练一致。"
  print_param CANDIDATE_ITEMS 12000 "两组使用相同 item_info 和候选顺序。"
  print_param SEEN_ITEM_MASK 1 "两组排序前屏蔽 history item 和 pad item。"
  print_param SELECTION_METRIC "$SELECTION_METRIC" "两组分别只按 valid 指标选 checkpoint。"
  print_param RUN_TEST "$RUN_TEST" "valid 选模并冻结后是否各运行一次 development test。"
  print_param RUN_GENERATION "$RUN_GENERATION" "是否调用 GLM 生成 train、valid、test CoT。"
  print_param RUN_TRAIN "$RUN_TRAIN" "是否训练并评测两组 retriever。"
  print_param CONFIRM_RUN "$CONFIRM_RUN" "必须为 1 才执行数据生成、API、训练或评测。"
  print_param PRINT_ARGS_ONLY "$PRINT_ARGS_ONLY" "为 1 时只打印本参数表。"
}

build_history_split() {
  local split="$1" source="$2" output="$3" prefix
  prefix="cds_${split}_query"
  if enabled "$FORCE_DATA" || [[ ! -s "$output" ]]; then
    "$PYTHON_BIN" ablantion/scripts/build_cds_query_ablation.py \
      --examples "$source" \
      --item-info "$ITEM_INFO" \
      --output-dir "$HISTORY_DIR" \
      --output-prefix "$prefix" \
      --ablations title_store_categories \
      --max-item-chars 0
    mv "$HISTORY_DIR/${prefix}_title_store_categories.jsonl" "$output"
  fi
  require_file "$split history pair" "$output"
  "$PYTHON_BIN" scripts/data/audit_embedding_retriever_dataset.py \
    --dataset "$output" \
    --expected-split "$split" \
    --expected-query-fields title,store,categories \
    --forbid-cot-tags \
    --batch-size "$BATCH_SIZE" \
    --output "$AUDIT_DIR/${split}_history.json"
}

prepare_generation_split() {
  local split="$1" history="$2" output="$3"
  if enabled "$FORCE_DATA" || [[ ! -s "$output" ]]; then
    "$PYTHON_BIN" scripts/data/prepare_target_free_cot_generation_input.py \
      --input "$history" \
      --output "$output" \
      --audit-output "$AUDIT_DIR/${split}_generation_input.json" \
      --expected-split "$split"
  fi
  require_file "$split target-free generation input" "$output"
}

generate_cot_split() {
  local split="$1" input="$2" output="$3"
  if ! enabled "$RUN_GENERATION"; then
    require_file "$split existing CoT raw" "$output"
    return
  fi
  if enabled "$FORCE_GENERATION"; then
    rm -f "$output" "${output%.jsonl}.candidates.jsonl" "${output%.jsonl}.failures.jsonl"
  fi
  local raw_arg=--no-record-api-raw
  if enabled "$RECORD_API_RAW"; then
    raw_arg=--record-api-raw
  fi
  "$PYTHON_BIN" scripts/cot/generate_cot_candidate_lists.py \
    --input "$input" \
    --output "$output" \
    --num-candidates "$NUM_CANDIDATES" \
    --temperatures "$TEMPERATURE" \
    --max-workers "$API_MAX_WORKERS" \
    --aggregate-every 100 \
    --resume \
    --api-provider "$API_PROVIDER" \
    --api-base-url "$API_BASE_URL" \
    --api-model "$API_MODEL" \
    --api-timeout "$API_TIMEOUT" \
    --api-max-retries "$API_MAX_RETRIES" \
    --api-min-interval "$API_MIN_INTERVAL" \
    --api-thinking "$API_THINKING" \
    --cot-output-format "$COT_OUTPUT_FORMAT" \
    --max-output-words "$MAX_OUTPUT_WORDS" \
    --rating-context "$RATING_CONTEXT" \
    --require-literal-tags \
    --max-new-tokens "$MAX_NEW_TOKENS" \
    --max-prompt-tokens 0 \
    --top-p "$TOP_P" \
    --seed "$SEED" \
    "$raw_arg"
  require_file "$split generated CoT raw" "$output"
}

build_cot_pair_split() {
  local split="$1" history="$2" raw="$3" output="$4"
  if enabled "$FORCE_DATA" || enabled "$FORCE_GENERATION" || [[ ! -s "$output" ]]; then
    "$PYTHON_BIN" ablantion/scripts/build_cds_query_ablation_cot.py \
      --base-jsonl "$history" \
      --cot-jsonl "$raw" \
      --output "$output" \
      --audit-output "$AUDIT_DIR/${split}_cot_build.json" \
      --split "$split" \
      --candidate-index 0 \
      --cot-text-mode tagged \
      --max-cot-chars 0 \
      --require-cot \
      --fail-on-raw-asin \
      --fail-on-truncated-positive
  fi
  require_file "$split CoT training pair" "$output"
  "$PYTHON_BIN" scripts/data/audit_aligned_history_cot_pairs.py \
    --history "$history" \
    --cot "$output" \
    --expected-split "$split" \
    --output "$AUDIT_DIR/${split}_pair_alignment.json"
  "$PYTHON_BIN" scripts/data/audit_target_overlap_in_cot.py \
    --history "$history" \
    --cot "$output" \
    --expected-split "$split" \
    --output "$AUDIT_DIR/${split}_target_overlap_diagnostic.json"
}

audit_tokens() {
  local label="$1" dataset="$2" include_items="${3:-0}"
  local args=(scripts/data/audit_embedding_token_lengths.py \
    --dataset "$dataset" \
    --tokenizer "$BASE_MODEL" \
    --max-length "$MAX_LENGTH" \
    --output "$AUDIT_DIR/${label}_token_lengths.json")
  if enabled "$include_items"; then
    args+=(--item-info "$ITEM_INFO")
  fi
  "$PYTHON_BIN" "${args[@]}"
}

train_model() {
  local label="$1" dataset="$2" output="$3" port="$4"
  local done_file="$output/.train_done"
  if ! enabled "$FORCE_TRAIN" && [[ -s "$done_file" ]]; then
    echo "跳过已完成训练: $output"
    return
  fi
  local args=(
    scripts/embedding/train_phase0_embedder.py
    --model "$BASE_MODEL"
    --dataset "$dataset"
    --output-dir "$output"
    --max-length "$MAX_LENGTH"
    --batch-size "$BATCH_SIZE"
    --grad-accum "$GRAD_ACCUM"
    --epochs "$EPOCHS"
    --learning-rate "$LEARNING_RATE"
    --weight-decay "$WEIGHT_DECAY"
    --warmup-ratio "$WARMUP_RATIO"
    --temperature "$TEMPERATURE_LOSS"
    --save-steps "$SAVE_STEPS"
    --torch-dtype "$TORCH_DTYPE"
    --gradient-checkpointing "$GRADIENT_CHECKPOINTING"
    --attn-implementation "$ATTN_IMPLEMENTATION"
    --no-multi-positive-targets
    --no-cross-gpu-negatives
    --no-sync-barriers
    --seed "$SEED"
  )
  echo "开始训练 $label。"
  if ((NPROC_PER_NODE > 1)); then
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PYTHON_BIN" -m torch.distributed.run \
      --nproc_per_node "$NPROC_PER_NODE" --master_port "$port" "${args[@]}"
  else
    CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PYTHON_BIN" "${args[@]}"
  fi
  date -u +"%Y-%m-%dT%H:%M:%SZ" > "$done_file"
}

run_eval() {
  local split="$1" dataset="$2" checkpoints="$3" output="$4"
  ROOT="$ROOT" VENV="$VENV" PYTHON_BIN="$PYTHON_BIN" CATEGORY="$CATEGORY" \
  SPLIT="$split" MAX_EXAMPLES=0 CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
  CHECKPOINT_ROOT="$checkpoints" CHECKPOINT_PATTERN='checkpoint-*' EVAL_DIR="$output" \
  EVAL_EXAMPLES="$dataset" ITEM_INFO="$ITEM_INFO" EMBEDDING_MAX_LENGTH="$EVAL_MAX_LENGTH" \
  EMBEDDING_BATCH_SIZE="$EVAL_BATCH_SIZE" EMBEDDING_TORCH_DTYPE="$TORCH_DTYPE" \
  EMBEDDING_DEVICE="$EVAL_DEVICE" MAX_ITEM_CHARS=0 EVAL_QUERY_MODE=user_history \
  EVAL_REQUIRE_COT=0 EVAL_MASK_HISTORY_ITEMS=1 EVAL_MASK_PAD_ITEM=1 \
  EVAL_KEEP_TARGET_UNMASKED=0 KS="$KS" FORCE_EVAL="$FORCE_EVAL" \
  bash scripts/embedding/run_eval_cds_embedding_checkpoints_tidal.sh
}

select_best() {
  "$PYTHON_BIN" scripts/eval/select_best_embedding_checkpoint.py \
    --eval-dir "$1" --metric "$SELECTION_METRIC" --output "$2"
}

eval_selected_test() {
  local dataset="$1" best_json="$2" eval_root="$3"
  local checkpoint
  checkpoint=$(
    "$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["best_checkpoint"])' "$best_json"
  )
  require_path "valid 选出的 checkpoint" "$checkpoint"
  run_eval test "$dataset" "$checkpoint" "$eval_root/final_test"
}

main() {
  print_plan
  if enabled "$PRINT_ARGS_ONLY"; then
    echo "只打印参数，不生成数据、不调用 API、不训练、不评测。"
    exit 0
  fi
  require_file AGENTS.md "$ROOT/AGENTS.md"
  require_path Python "$PYTHON_BIN"
  require_path "base embedding model" "$BASE_MODEL"
  require_file "train source" "$TRAIN_SOURCE"
  require_file "valid source" "$VALID_SOURCE"
  require_file "test source" "$TEST_SOURCE"
  require_file item_info "$ITEM_INFO"
  if ! enabled "$CONFIRM_RUN"; then
    echo "未启动。确认参数后设置 CONFIRM_RUN=1。"
    exit 2
  fi
  if [[ "$SEED" != "42" ]]; then
    echo "SEED 必须为 42。" >&2
    exit 2
  fi
  if enabled "$MULTI_POSITIVE_TARGETS" || enabled "$CROSS_GPU_NEGATIVES"; then
    echo "本对齐实验固定 single-positive 且关闭 cross-GPU negatives。" >&2
    exit 2
  fi

  cd "$ROOT"
  mkdir -p "$HISTORY_DIR" "$GEN_INPUT_DIR" "$COT_API_DIR" "$COT_PAIR_DIR" \
    "$AUDIT_DIR" "$REPORT_DIR" "$HISTORY_OUT" "$COT_OUT" "$HISTORY_EVAL" "$COT_EVAL"
  export PATH="$VENV/bin:$PATH"
  export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
  export NCCL_NET NCCL_IB_DISABLE NCCL_P2P_DISABLE NCCL_NVLS_ENABLE NCCL_MNNVL_ENABLE
  export NCCL_COLLNET_ENABLE NCCL_DEBUG TORCH_NCCL_ASYNC_ERROR_HANDLING PYTORCH_CUDA_ALLOC_CONF

  build_history_split train "$TRAIN_SOURCE" "$HISTORY_TRAIN"
  build_history_split valid "$VALID_SOURCE" "$HISTORY_VALID"
  build_history_split test "$TEST_SOURCE" "$HISTORY_TEST"
  prepare_generation_split train "$HISTORY_TRAIN" "$GEN_TRAIN"
  prepare_generation_split valid "$HISTORY_VALID" "$GEN_VALID"
  prepare_generation_split test "$HISTORY_TEST" "$GEN_TEST"

  generate_cot_split train "$GEN_TRAIN" "$COT_RAW_TRAIN"
  generate_cot_split valid "$GEN_VALID" "$COT_RAW_VALID"
  generate_cot_split test "$GEN_TEST" "$COT_RAW_TEST"
  build_cot_pair_split train "$HISTORY_TRAIN" "$COT_RAW_TRAIN" "$COT_TRAIN"
  build_cot_pair_split valid "$HISTORY_VALID" "$COT_RAW_VALID" "$COT_VALID"
  build_cot_pair_split test "$HISTORY_TEST" "$COT_RAW_TEST" "$COT_TEST"

  audit_tokens history_train "$HISTORY_TRAIN" 1
  audit_tokens history_valid "$HISTORY_VALID"
  audit_tokens history_test "$HISTORY_TEST"
  audit_tokens cot_train "$COT_TRAIN"
  audit_tokens cot_valid "$COT_VALID"
  audit_tokens cot_test "$COT_TEST"

  if ! enabled "$RUN_TRAIN"; then
    echo "数据与 token 审计完成；RUN_TRAIN=0，未启动模型训练。"
    exit 0
  fi
  train_model history "$HISTORY_TRAIN" "$HISTORY_OUT" "$MASTER_PORT_HISTORY"
  train_model external_cot "$COT_TRAIN" "$COT_OUT" "$MASTER_PORT_COT"
  run_eval valid "$HISTORY_VALID" "$HISTORY_OUT" "$HISTORY_EVAL"
  run_eval valid "$COT_VALID" "$COT_OUT" "$COT_EVAL"
  select_best "$HISTORY_EVAL" "$HISTORY_BEST"
  select_best "$COT_EVAL" "$COT_BEST"

  if enabled "$RUN_TEST"; then
    eval_selected_test "$HISTORY_TEST" "$HISTORY_BEST" "$HISTORY_EVAL"
    eval_selected_test "$COT_TEST" "$COT_BEST" "$COT_EVAL"
  fi
  echo "严格对齐实验完成。结果目录: $REPORT_DIR"
}

main "$@"
