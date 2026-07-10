#!/usr/bin/env bash
set -euo pipefail

# Phase 3 pilot: build same-history CoT groups, listwise-judge a stratified
# subset without target text, and train one fold-disjoint pairwise scorer.

ROOT=${ROOT:-/home/user/aaai_pro}
VENV=${VENV:-/home/user/.conda/envs/aaai_pro}
PYTHON_BIN=${PYTHON_BIN:-$VENV/bin/python}
CONFIG_FILE=${CONFIG_FILE:-$ROOT/configs/glm_codeplan.env}

CATEGORY=${CATEGORY:-CDs_and_Vinyl}
COMPLETIONS=${COMPLETIONS:-$ROOT/checkpoints/rrec_amazon_CDs_and_Vinyl/qwen25_3b_lora_grpo_ref_soft_ndcg20_100_1000_equal_refbonus1p0_t0p7_vllm016_oomsafe_from_sft354_rejected80_lr1e-5_g4_b16_acc1_c512_len2048_ep3/v0-20260709-141638/completions.jsonl}
COMPONENTS=${COMPONENTS:-$ROOT/checkpoints/rrec_amazon_CDs_and_Vinyl/qwen25_3b_lora_grpo_ref_soft_ndcg20_100_1000_equal_refbonus1p0_t0p7_vllm016_oomsafe_from_sft354_rejected80_lr1e-5_g4_b16_acc1_c512_len2048_ep3/rubric_ref_soft_ndcg_components.jsonl}
SOURCE_SCORED=${SOURCE_SCORED:-$ROOT/outputs/rrec_amazon/$CATEGORY/sft_quality/cot_scored_glm47_tsc_no_trunc_rubric_gain_top20_ndcg100.jsonl}
DATA_DIR=${DATA_DIR:-$ROOT/outputs/rrec_amazon/$CATEGORY/preference_v2}
GROUPS=${GROUPS:-$DATA_DIR/grpo993_glm47_corruption_groups.jsonl}
GROUP_SUMMARY=${GROUP_SUMMARY:-$DATA_DIR/grpo993_glm47_corruption_groups.summary.json}
JUDGMENTS=${JUDGMENTS:-$DATA_DIR/grpo993_glm47_corruption_groups.glm52_pilot200.jsonl}

EMBEDDING_MODEL=${EMBEDDING_MODEL:-/home/user/models_hf/Qwen3-Embedding-0.6B}
RUN_NAME=${RUN_NAME:-cot_preference_v2_glm52_pilot200_fold0_20260710}
OUTPUT_DIR=${OUTPUT_DIR:-$ROOT/checkpoints/rrec_amazon_CDs_and_Vinyl/$RUN_NAME}
EMBEDDING_CACHE=${EMBEDDING_CACHE:-$OUTPUT_DIR/embedding_cache.pt}

NUM_FOLDS=${NUM_FOLDS:-5}
BUILD_MAX_GROUPS=${BUILD_MAX_GROUPS:-0}
MIN_CANDIDATES=${MIN_CANDIDATES:-4}
JUDGE_MAX_GROUPS=${JUDGE_MAX_GROUPS:-200}
JUDGE_BASE_URL=${JUDGE_BASE_URL:-https://open.bigmodel.cn/api/paas/v4}
JUDGE_MODEL=${JUDGE_MODEL:-glm-5.2}
JUDGE_WORKERS=${JUDGE_WORKERS:-4}
JUDGE_TIMEOUT=${JUDGE_TIMEOUT:-300}
JUDGE_MAX_RETRIES=${JUDGE_MAX_RETRIES:-5}
JUDGE_MAX_TOKENS=${JUDGE_MAX_TOKENS:-2048}
JUDGE_THINKING=${JUDGE_THINKING:-disabled}
JUDGE_TEMPERATURE=${JUDGE_TEMPERATURE:-0}
JUDGE_TOP_P=${JUDGE_TOP_P:-0.9}
JUDGE_SAVE_RAW=${JUDGE_SAVE_RAW:-0}

VALID_FOLD=${VALID_FOLD:-0}
MAX_LENGTH=${MAX_LENGTH:-4096}
ENCODE_BATCH_SIZE=${ENCODE_BATCH_SIZE:-64}
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-512}
EPOCHS=${EPOCHS:-20}
LEARNING_RATE=${LEARNING_RATE:-1e-3}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
WARMUP_RATIO=${WARMUP_RATIO:-0.05}
HIDDEN_DIM=${HIDDEN_DIM:-512}
DROPOUT=${DROPOUT:-0.1}
TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-flash_attention_2}
DEVICE=${DEVICE:-cuda:0}
SEED=${SEED:-42}

RUN_BUILD=${RUN_BUILD:-1}
RUN_JUDGE=${RUN_JUDGE:-1}
RUN_TRAIN=${RUN_TRAIN:-1}
FORCE_BUILD=${FORCE_BUILD:-0}
FORCE_REENCODE=${FORCE_REENCODE:-0}
CONFIRM_RUN=${CONFIRM_RUN:-0}
PRINT_ARGS_ONLY=${PRINT_ARGS_ONLY:-0}

PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

require_path() {
  local label="$1"
  local path="$2"
  if [[ ! -e "$path" ]]; then
    echo "缺少${label}: $path" >&2
    exit 1
  fi
}

require_file() {
  local label="$1"
  local path="$2"
  if [[ ! -s "$path" ]]; then
    echo "缺少或为空的${label}: $path" >&2
    exit 1
  fi
}

enabled() {
  [[ "$1" == "1" || "$1" == "true" ]]
}

print_param() {
  printf "%s=%s # %s\n" "$1" "$2" "$3"
}

if [[ -f "$CONFIG_FILE" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "$CONFIG_FILE"
  set +a
fi
JUDGE_API_KEY=${JUDGE_API_KEY:-${RUBRIC_JUDGE_API_KEY:-${BIGMODEL_API_KEY:-}}}

print_plan() {
  echo "第三阶段 CoT preference scorer pilot 参数核对："
  print_param "ROOT" "$ROOT" "项目根目录。"
  print_param "VENV" "$VENV" "A100 Conda 环境。"
  print_param "PYTHON_BIN" "$PYTHON_BIN" "数据构建、API judge 和 scorer 训练使用的 Python。"
  print_param "CATEGORY" "$CATEGORY" "Amazon 数据类目。"
  print_param "COMPLETIONS" "$COMPLETIONS" "已停止 GRPO 的 993 step completion 日志，每组包含 4 条 policy CoT。"
  print_param "COMPONENTS" "$COMPONENTS" "与 completion 逐行对齐的 rank、q_new、q_ref 元数据。"
  print_param "SOURCE_SCORED" "$SOURCE_SCORED" "外部 GLM CoT、无评分 history 和 baseline/cot rank 来源。"
  print_param "GROUPS" "$GROUPS" "同 history 多 CoT 分组输出；每组由 policy、external GLM 和扰动负例组成。"
  print_param "GROUP_SUMMARY" "$GROUP_SUMMARY" "候选来源、fold 和数据完整性统计。"
  print_param "NUM_FOLDS" "$NUM_FOLDS" "按 example_id hash 划分的 fold 数，保证同一 history 不跨 fold。"
  print_param "BUILD_MAX_GROUPS" "$BUILD_MAX_GROUPS" "最多构建多少组；0 表示使用日志中的全部 3972 组。"
  print_param "MIN_CANDIDATES" "$MIN_CANDIDATES" "去重后保留一个 history 组所需的最少候选数。"
  print_param "JUDGMENTS" "$JUDGMENTS" "GLM-5.2 listwise judge 输出，按 example_id 可恢复。"
  print_param "JUDGE_MAX_GROUPS" "$JUDGE_MAX_GROUPS" "pilot 最多 judge 200 个随机 history 组，约 200 次 API 请求。"
  print_param "JUDGE_BASE_URL" "$JUDGE_BASE_URL" "OpenAI-compatible GLM API 地址。"
  print_param "JUDGE_MODEL" "$JUDGE_MODEL" "listwise judge 模型。"
  print_param "JUDGE_API_KEY_PRESENT" "$([[ -n "$JUDGE_API_KEY" ]] && echo 1 || echo 0)" "仅检查密钥是否存在，不打印密钥内容。"
  print_param "JUDGE_WORKERS" "$JUDGE_WORKERS" "并发 API 请求数。"
  print_param "JUDGE_TIMEOUT" "$JUDGE_TIMEOUT" "单次 API 请求超时秒数。"
  print_param "JUDGE_MAX_RETRIES" "$JUDGE_MAX_RETRIES" "格式错误、限流或网络异常后的重试次数。"
  print_param "JUDGE_MAX_TOKENS" "$JUDGE_MAX_TOKENS" "judge JSON 响应最大 token 数。"
  print_param "JUDGE_THINKING" "$JUDGE_THINKING" "judge 内部 thinking 开关；disabled 降低延迟。"
  print_param "JUDGE_TEMPERATURE" "$JUDGE_TEMPERATURE" "judge 采样温度；0 保持排序稳定。"
  print_param "JUDGE_TOP_P" "$JUDGE_TOP_P" "judge nucleus sampling 参数。"
  print_param "JUDGE_USED_TARGET" "0" "judge prompt 不包含 target title、文本或 ID。"
  print_param "EMBEDDING_MODEL" "$EMBEDDING_MODEL" "冻结的原始 Qwen3-Embedding-0.6B；不复用训练样本上的 rank scorer。"
  print_param "OUTPUT_DIR" "$OUTPUT_DIR" "fold-0 pairwise scorer、指标和 embedding cache 输出目录。"
  print_param "EMBEDDING_CACHE" "$EMBEDDING_CACHE" "history 与 history+think 表示的缓存。"
  print_param "VALID_FOLD" "$VALID_FOLD" "只将该 fold 用作验证，其余 fold 训练。"
  print_param "TEXT_MODE" "think_only" "scorer 只读取 history 和 think；answer 不参与质量表示。"
  print_param "FEATURE_MODE" "joint_delta_product" "拼接 history+think 表示、相对 history 的差值及逐维乘积。"
  print_param "MAX_LENGTH" "$MAX_LENGTH" "embedding tokenizer 最大长度。"
  print_param "ENCODE_BATCH_SIZE" "$ENCODE_BATCH_SIZE" "冻结 encoder 的推理 batch size。"
  print_param "TRAIN_BATCH_SIZE" "$TRAIN_BATCH_SIZE" "Bradley-Terry pair batch size。"
  print_param "EPOCHS" "$EPOCHS" "pilot scorer head 训练轮数。"
  print_param "LEARNING_RATE" "$LEARNING_RATE" "scorer head AdamW 学习率。"
  print_param "WEIGHT_DECAY" "$WEIGHT_DECAY" "scorer head 权重衰减。"
  print_param "WARMUP_RATIO" "$WARMUP_RATIO" "cosine scheduler warmup 比例。"
  print_param "HIDDEN_DIM" "$HIDDEN_DIM" "pairwise scorer MLP 隐层维度。"
  print_param "DROPOUT" "$DROPOUT" "scorer head dropout。"
  print_param "TORCH_DTYPE" "$TORCH_DTYPE" "冻结 encoder 推理数据类型。"
  print_param "ATTN_IMPLEMENTATION" "$ATTN_IMPLEMENTATION" "冻结 encoder attention 后端。"
  print_param "DEVICE" "$DEVICE" "embedding 编码和 scorer head 训练设备。"
  print_param "RUN_BUILD" "$RUN_BUILD" "是否构建 preference groups。"
  print_param "RUN_JUDGE" "$RUN_JUDGE" "是否调用 GLM-5.2 listwise judge。"
  print_param "RUN_TRAIN" "$RUN_TRAIN" "是否训练 fold-0 pairwise scorer。"
  print_param "FORCE_BUILD" "$FORCE_BUILD" "是否覆盖已有 groups。"
  print_param "FORCE_REENCODE" "$FORCE_REENCODE" "是否忽略 embedding cache 重新编码。"
  print_param "CONFIRM_RUN" "$CONFIRM_RUN" "必须为 1 才执行数据生成、API 请求或 GPU 训练。"
  print_param "PRINT_ARGS_ONLY" "$PRINT_ARGS_ONLY" "为 1 时只打印参数。"
  print_param "PYTORCH_CUDA_ALLOC_CONF" "$PYTORCH_CUDA_ALLOC_CONF" "CUDA allocator 配置。"
  print_param "SEED" "$SEED" "候选打乱、fold 和训练随机种子。"
}

main() {
  require_path "项目根目录" "$ROOT"
  require_file "AGENTS.md" "$ROOT/AGENTS.md"
  require_path "Python" "$PYTHON_BIN"
  require_file "completion 日志" "$COMPLETIONS"
  require_file "reward component 日志" "$COMPONENTS"
  require_file "外部 GLM scored 数据" "$SOURCE_SCORED"
  require_path "冻结 embedding 模型" "$EMBEDDING_MODEL"
  print_plan
  if enabled "$PRINT_ARGS_ONLY"; then
    echo "只打印参数，不生成数据、不调用 API、不启动 GPU。"
    exit 0
  fi
  if ! enabled "$CONFIRM_RUN"; then
    echo "未执行。确认参数后设置 CONFIRM_RUN=1。"
    exit 2
  fi
  if enabled "$RUN_JUDGE" && [[ -z "$JUDGE_API_KEY" ]]; then
    echo "缺少 GLM judge API key。" >&2
    exit 2
  fi

  cd "$ROOT"
  mkdir -p "$DATA_DIR" "$OUTPUT_DIR"
  export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
  export PYTORCH_CUDA_ALLOC_CONF

  if enabled "$RUN_BUILD" && (enabled "$FORCE_BUILD" || [[ ! -s "$GROUPS" ]]); then
    "$PYTHON_BIN" scripts/datasets/build_cot_preference_groups_v2.py \
      --completions "$COMPLETIONS" \
      --components "$COMPONENTS" \
      --source-scored "$SOURCE_SCORED" \
      --output "$GROUPS" \
      --summary-output "$GROUP_SUMMARY" \
      --num-folds "$NUM_FOLDS" \
      --max-groups "$BUILD_MAX_GROUPS" \
      --min-candidates "$MIN_CANDIDATES"
  fi
  require_file "preference groups" "$GROUPS"

  if enabled "$RUN_JUDGE"; then
    judge_raw_arg=--no-save-raw
    if enabled "$JUDGE_SAVE_RAW"; then
      judge_raw_arg=--save-raw
    fi
    "$PYTHON_BIN" scripts/cot/judge_cot_preference_groups_v2.py \
      --input "$GROUPS" \
      --output "$JUDGMENTS" \
      --base-url "$JUDGE_BASE_URL" \
      --api-key "$JUDGE_API_KEY" \
      --judge-model "$JUDGE_MODEL" \
      --max-groups "$JUDGE_MAX_GROUPS" \
      --max-workers "$JUDGE_WORKERS" \
      --timeout "$JUDGE_TIMEOUT" \
      --max-retries "$JUDGE_MAX_RETRIES" \
      --max-tokens "$JUDGE_MAX_TOKENS" \
      --thinking "$JUDGE_THINKING" \
      --temperature "$JUDGE_TEMPERATURE" \
      --top-p "$JUDGE_TOP_P" \
      --seed "$SEED" \
      "$judge_raw_arg"
  fi
  require_file "listwise judgments" "$JUDGMENTS"

  if enabled "$RUN_TRAIN"; then
    force_reencode_arg=()
    if enabled "$FORCE_REENCODE"; then
      force_reencode_arg=(--force-reencode)
    fi
    "$PYTHON_BIN" scripts/train/train_cot_preference_scorer_v2.py \
      --groups "$GROUPS" \
      --judgments "$JUDGMENTS" \
      --embedding-model "$EMBEDDING_MODEL" \
      --embedding-cache "$EMBEDDING_CACHE" \
      --output-dir "$OUTPUT_DIR" \
      --valid-fold "$VALID_FOLD" \
      --max-length "$MAX_LENGTH" \
      --encode-batch-size "$ENCODE_BATCH_SIZE" \
      --train-batch-size "$TRAIN_BATCH_SIZE" \
      --epochs "$EPOCHS" \
      --learning-rate "$LEARNING_RATE" \
      --weight-decay "$WEIGHT_DECAY" \
      --warmup-ratio "$WARMUP_RATIO" \
      --hidden-dim "$HIDDEN_DIM" \
      --dropout "$DROPOUT" \
      --torch-dtype "$TORCH_DTYPE" \
      --attn-implementation "$ATTN_IMPLEMENTATION" \
      --device "$DEVICE" \
      --seed "$SEED" \
      "${force_reencode_arg[@]}"
  fi
}

main "$@"
