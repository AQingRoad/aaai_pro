#!/usr/bin/env bash
set -euo pipefail

# Phase 1: train a history-only retriever with multi-positive InfoNCE and
# select its checkpoint on the validation split. Test evaluation is opt-in.

ROOT=${ROOT:-/home/user/aaai_pro}
VENV=${VENV:-/home/user/.conda/envs/aaai_pro}
PYTHON_BIN=${PYTHON_BIN:-$VENV/bin/python}

CATEGORY=${CATEGORY:-CDs_and_Vinyl}
BASE_EMBEDDING_MODEL=${BASE_EMBEDDING_MODEL:-/home/user/models_hf/Qwen3-Embedding-0.6B}
DATA_DIR=${DATA_DIR:-$ROOT/ablantion/datas/processed_datas/cds_history_only_retriever_v2}
TRAIN_SOURCE=${TRAIN_SOURCE:-$ROOT/data/rrec_amazon/CDs_and_Vinyl/examples.jsonl}
VALID_SOURCE=${VALID_SOURCE:-$ROOT/github_artifacts/CDs_and_Vinyl/rrec_eval/valid.jsonl}
TEST_SOURCE=${TEST_SOURCE:-$ROOT/github_artifacts/CDs_and_Vinyl/rrec_eval/test.jsonl}
TRAIN_DATASET=${TRAIN_DATASET:-$DATA_DIR/cds_train_query_title_store_categories.jsonl}
VALID_DATASET=${VALID_DATASET:-$DATA_DIR/cds_valid_query_title_store_categories.jsonl}
TEST_DATASET=${TEST_DATASET:-$DATA_DIR/cds_test_query_title_store_categories.jsonl}
ITEM_INFO=${ITEM_INFO:-$ROOT/github_artifacts/CDs_and_Vinyl/rrec_eval/item_info.jsonl}
PREVIOUS_HISTORY_CHECKPOINT=${PREVIOUS_HISTORY_CHECKPOINT:-$ROOT/checkpoints/rrec_amazon_CDs_and_Vinyl/cds_query_ablation_lr2e-5_epoch4/title_store_categories/checkpoint-83}
CURRENT_REWARD_CHECKPOINT=${CURRENT_REWARD_CHECKPOINT:-$ROOT/checkpoints/rrec_amazon_CDs_and_Vinyl/cds_query_ablation_cot_lr2e-5_epoch5/title_store_categories_no_trunc_plus_cot/checkpoint-83}

RUN_NAME=${RUN_NAME:-cds_history_only_multipos_lr2e-5_epoch4_20260710}
EMBEDDER_OUT=${EMBEDDER_OUT:-$ROOT/checkpoints/rrec_amazon_CDs_and_Vinyl/$RUN_NAME}
EVAL_DIR=${EVAL_DIR:-$ROOT/outputs/rrec_amazon/eval/$CATEGORY/$RUN_NAME}
REPORT_DIR=${REPORT_DIR:-$ROOT/experiments/results/$RUN_NAME}
BEST_SUMMARY=${BEST_SUMMARY:-$REPORT_DIR/best_valid_checkpoint.json}

CONFIRM_RUN=${CONFIRM_RUN:-0}
PRINT_ARGS_ONLY=${PRINT_ARGS_ONLY:-0}
FORCE_DATA=${FORCE_DATA:-0}
FORCE_TRAIN=${FORCE_TRAIN:-0}
FORCE_EVAL=${FORCE_EVAL:-0}
RUN_REFERENCE_EVAL=${RUN_REFERENCE_EVAL:-1}
RUN_TEST=${RUN_TEST:-0}

CUDA_DEVICES=${CUDA_DEVICES:-0}
NPROC_PER_NODE=${NPROC_PER_NODE:-1}
MASTER_PORT=${MASTER_PORT:-29543}
BATCH_SIZE=${BATCH_SIZE:-128}
GRAD_ACCUM=${GRAD_ACCUM:-1}
MAX_LENGTH=${MAX_LENGTH:-4096}
EPOCHS=${EPOCHS:-4}
MAX_STEPS=${MAX_STEPS:--1}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
SAVE_STEPS=${SAVE_STEPS:-auto}
TORCH_DTYPE=${TORCH_DTYPE:-bfloat16}
GRADIENT_CHECKPOINTING=${GRADIENT_CHECKPOINTING:-auto}
ATTN_IMPLEMENTATION=${ATTN_IMPLEMENTATION:-flash_attention_2}
CROSS_GPU_NEGATIVES=${CROSS_GPU_NEGATIVES:-0}
MULTI_POSITIVE_TARGETS=${MULTI_POSITIVE_TARGETS:-1}
MAX_ITEM_CHARS=${MAX_ITEM_CHARS:-0}
SEED=${SEED:-42}

EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-256}
EVAL_MAX_LENGTH=${EVAL_MAX_LENGTH:-4096}
EVAL_DEVICE=${EVAL_DEVICE:-cuda:0}
SELECTION_METRIC=${SELECTION_METRIC:-NDCG@20}
KS=${KS:-5,10,20}

NCCL_NET=${NCCL_NET:-Socket}
NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}
NCCL_P2P_DISABLE=${NCCL_P2P_DISABLE:-1}
NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0}
NCCL_MNNVL_ENABLE=${NCCL_MNNVL_ENABLE:-0}
NCCL_COLLNET_ENABLE=${NCCL_COLLNET_ENABLE:-0}
NCCL_DEBUG=${NCCL_DEBUG:-WARN}
TORCH_NCCL_ASYNC_ERROR_HANDLING=${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}
PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

require_path() {
  local label="$1"
  local path="$2"
  if [[ ! -e "$path" ]]; then
    echo "缺少 ${label}: ${path}" >&2
    exit 1
  fi
}

require_file() {
  local label="$1"
  local path="$2"
  if [[ ! -s "$path" ]]; then
    echo "缺少或为空 ${label}: ${path}" >&2
    exit 1
  fi
}

flag_enabled() {
  [[ "$1" == "1" || "$1" == "true" ]]
}

print_param() {
  printf "%s=%s # %s\n" "$1" "$2" "$3"
}

resolve_save_steps() {
  if [[ "$SAVE_STEPS" != "auto" ]]; then
    echo "$SAVE_STEPS"
    return
  fi
  local rows
  rows=$(wc -l < "$TRAIN_SOURCE" | tr -d ' ')
  local full_batches=$((rows / (BATCH_SIZE * NPROC_PER_NODE)))
  if ((full_batches < 1)); then
    echo "训练数据不足一个完整全局 batch" >&2
    exit 1
  fi
  echo $(((full_batches + GRAD_ACCUM - 1) / GRAD_ACCUM))
}

print_plan() {
  local save_steps="$1"
  local train_rows valid_rows
  train_rows=$(wc -l < "$TRAIN_SOURCE" | tr -d ' ')
  valid_rows=$(wc -l < "$VALID_SOURCE" | tr -d ' ')
  echo "第一阶段 history-only retriever 参数核对："
  print_param "ROOT" "$ROOT" "项目根目录；执行前会读取其中的 AGENTS.md。"
  print_param "VENV" "$VENV" "A100 上包含 torch、transformers 和 flash-attn 的 Conda 环境。"
  print_param "PYTHON_BIN" "$PYTHON_BIN" "数据构建、训练、评测和选模使用的 Python。"
  print_param "CATEGORY" "$CATEGORY" "本次训练和全候选集评测的 Amazon 类目。"
  print_param "BASE_EMBEDDING_MODEL" "$BASE_EMBEDDING_MODEL" "Qwen3-Embedding-0.6B 初始权重。"
  print_param "DATA_DIR" "$DATA_DIR" "无字符截断的 train/valid/test history-only JSONL 输出目录。"
  print_param "TRAIN_SOURCE" "$TRAIN_SOURCE" "官方 train 源文件。"
  print_param "TRAIN_DATASET" "$TRAIN_DATASET" "待生成训练输入；query 只含 title/store/categories history，positive 为完整目标 item 文本。"
  print_param "TRAIN_ROWS" "$train_rows" "训练样本数。"
  print_param "VALID_SOURCE" "$VALID_SOURCE" "官方 valid split 源文件，仅用于生成同口径 history-only query。"
  print_param "VALID_ROWS" "$valid_rows" "valid 样本数。"
  print_param "VALID_DATASET" "$VALID_DATASET" "生成后的 valid query 文件。"
  print_param "TEST_SOURCE" "$TEST_SOURCE" "官方 test split 源文件。"
  print_param "TEST_DATASET" "$TEST_DATASET" "冻结方案后的最终测试文件；RUN_TEST=0 时不读取指标。"
  print_param "ITEM_INFO" "$ITEM_INFO" "12000 个候选 item 的完整文本来源。"
  print_param "PREVIOUS_HISTORY_CHECKPOINT" "$PREVIOUS_HISTORY_CHECKPOINT" "此前 title/store/categories history-only checkpoint-83，作为直接 valid 基线。"
  print_param "CURRENT_REWARD_CHECKPOINT" "$CURRENT_REWARD_CHECKPOINT" "当前 SFT/GRPO 链路使用的 history+CoT checkpoint-83，只报告跨口径参照。"
  print_param "RUN_NAME" "$RUN_NAME" "checkpoint、评测结果和审计报告的实验标识。"
  print_param "EMBEDDER_OUT" "$EMBEDDER_OUT" "模型 checkpoint 输出目录。"
  print_param "EVAL_DIR" "$EVAL_DIR" "valid/test 全候选集指标输出目录。"
  print_param "REPORT_DIR" "$REPORT_DIR" "数据审计和 valid 选模报告目录。"
  print_param "BEST_SUMMARY" "$BEST_SUMMARY" "最佳 valid checkpoint 及全部候选指标的汇总文件。"
  print_param "CUDA_DEVICES" "$CUDA_DEVICES" "训练和评测使用的 GPU 编号。"
  print_param "NPROC_PER_NODE" "$NPROC_PER_NODE" "训练进程数。"
  print_param "MASTER_PORT" "$MASTER_PORT" "多进程训练 rendezvous 端口；单卡时不使用。"
  print_param "BATCH_SIZE" "$BATCH_SIZE" "每个进程的 query batch size。"
  print_param "GRAD_ACCUM" "$GRAD_ACCUM" "梯度累积步数。"
  print_param "GLOBAL_BATCH_SIZE" "$((BATCH_SIZE * NPROC_PER_NODE * GRAD_ACCUM))" "每次参数更新覆盖的 query 数。"
  print_param "MAX_LENGTH" "$MAX_LENGTH" "训练 query 和 item 的 tokenizer 最大长度。"
  print_param "MAX_ITEM_CHARS" "$MAX_ITEM_CHARS" "history metadata 和候选 item 文本的字符上限；0 表示不做字符截断。"
  print_param "EPOCHS" "$EPOCHS" "训练轮数。"
  print_param "MAX_STEPS" "$MAX_STEPS" "最大更新步数；-1 表示按 epoch 计算。"
  print_param "LEARNING_RATE" "$LEARNING_RATE" "AdamW 学习率。"
  print_param "SAVE_STEPS" "$save_steps" "checkpoint 保存间隔；auto 对应每轮保存一次。"
  print_param "MULTI_POSITIVE_TARGETS" "$MULTI_POSITIVE_TARGETS" "同 target_item_id 的 batch 文档共同作为正样本，消除重复目标假负例。"
  print_param "CROSS_GPU_NEGATIVES" "$CROSS_GPU_NEGATIVES" "是否拼接跨 GPU 文档；单卡默认关闭。"
  print_param "GRADIENT_CHECKPOINTING" "$GRADIENT_CHECKPOINTING" "activation checkpointing 模式。"
  print_param "ATTN_IMPLEMENTATION" "$ATTN_IMPLEMENTATION" "attention 后端。"
  print_param "TORCH_DTYPE" "$TORCH_DTYPE" "模型训练数据类型。"
  print_param "EVAL_BATCH_SIZE" "$EVAL_BATCH_SIZE" "valid/test query 和 item 编码 batch size。"
  print_param "EVAL_MAX_LENGTH" "$EVAL_MAX_LENGTH" "评测 tokenizer 最大长度，与训练保持 4096。"
  print_param "EVAL_DEVICE" "$EVAL_DEVICE" "全候选集 embedding 编码和排序使用的设备。"
  print_param "SELECTION_METRIC" "$SELECTION_METRIC" "只在 valid 上选择 checkpoint 的排序指标。"
  print_param "KS" "$KS" "valid/test 输出 HR 和 NDCG 的 cutoff。"
  print_param "EVAL_QUERY_MODE" "user_history" "直接读取重建 JSONL 的 query，保证训练与评测字段一致。"
  print_param "EVAL_MASK_HISTORY_ITEMS" "1" "排序前屏蔽已交互的 history item。"
  print_param "EVAL_MASK_PAD_ITEM" "1" "排序前屏蔽 pad item。"
  print_param "EVAL_KEEP_TARGET_UNMASKED" "0" "target 若出现在 history 中不强制解除屏蔽；审计要求该情况为 0。"
  print_param "RUN_REFERENCE_EVAL" "$RUN_REFERENCE_EVAL" "是否评测 checkpoint-83 的 valid 基线。"
  print_param "RUN_TEST" "$RUN_TEST" "是否对 valid 选出的最佳 checkpoint 运行一次 test；第一阶段默认关闭。"
  print_param "FORCE_DATA" "$FORCE_DATA" "是否覆盖已经生成的 train/valid/test history-only query。"
  print_param "FORCE_TRAIN" "$FORCE_TRAIN" "是否忽略完成标记重新训练。"
  print_param "FORCE_EVAL" "$FORCE_EVAL" "是否覆盖已有 valid/test 指标。"
  print_param "CONFIRM_RUN" "$CONFIRM_RUN" "必须设为 1 才会生成数据、训练或评测。"
  print_param "PRINT_ARGS_ONLY" "$PRINT_ARGS_ONLY" "设为 1 时只打印参数。"
  print_param "SEED" "$SEED" "训练随机种子。"
  print_param "PYTORCH_CUDA_ALLOC_CONF" "$PYTORCH_CUDA_ALLOC_CONF" "CUDA allocator 使用 expandable segments，减少长序列显存碎片。"
  print_param "NCCL_NET" "$NCCL_NET" "NCCL 使用的网络后端。"
  print_param "NCCL_IB_DISABLE" "$NCCL_IB_DISABLE" "关闭容器内不可用的 InfiniBand 通路。"
  print_param "NCCL_P2P_DISABLE" "$NCCL_P2P_DISABLE" "关闭当前单卡实验不需要的 GPU P2P。"
  print_param "NCCL_NVLS_ENABLE" "$NCCL_NVLS_ENABLE" "关闭当前驱动环境不使用的 NVLS。"
  print_param "NCCL_MNNVL_ENABLE" "$NCCL_MNNVL_ENABLE" "关闭多节点 NVLink 域功能。"
  print_param "NCCL_COLLNET_ENABLE" "$NCCL_COLLNET_ENABLE" "关闭当前单机任务不使用的 CollNet。"
  print_param "NCCL_DEBUG" "$NCCL_DEBUG" "NCCL 日志级别。"
  print_param "TORCH_NCCL_ASYNC_ERROR_HANDLING" "$TORCH_NCCL_ASYNC_ERROR_HANDLING" "异步上报 NCCL 错误，避免分布式任务静默挂起。"
}

run_eval() {
  local split="$1"
  local examples="$2"
  local checkpoint_root="$3"
  local checkpoint_pattern="$4"
  local output_dir="$5"
  ROOT="$ROOT" \
  VENV="$VENV" \
  PYTHON_BIN="$PYTHON_BIN" \
  CATEGORY="$CATEGORY" \
  SPLIT="$split" \
  MAX_EXAMPLES=0 \
  CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" \
  CHECKPOINT_ROOT="$checkpoint_root" \
  CHECKPOINT_PATTERN="$checkpoint_pattern" \
  EVAL_DIR="$output_dir" \
  EVAL_EXAMPLES="$examples" \
  ITEM_INFO="$ITEM_INFO" \
  EMBEDDING_MAX_LENGTH="$EVAL_MAX_LENGTH" \
  EMBEDDING_BATCH_SIZE="$EVAL_BATCH_SIZE" \
  EMBEDDING_TORCH_DTYPE="$TORCH_DTYPE" \
  EMBEDDING_DEVICE="$EVAL_DEVICE" \
  MAX_ITEM_CHARS="$MAX_ITEM_CHARS" \
  EVAL_QUERY_MODE=user_history \
  EVAL_REQUIRE_COT=0 \
  EVAL_MASK_HISTORY_ITEMS=1 \
  EVAL_MASK_PAD_ITEM=1 \
  EVAL_KEEP_TARGET_UNMASKED=0 \
  KS="$KS" \
  FORCE_EVAL="$FORCE_EVAL" \
  bash scripts/embedding/run_eval_cds_embedding_checkpoints_tidal.sh
}

main() {
  require_path "项目根目录" "$ROOT"
  require_file "AGENTS.md" "$ROOT/AGENTS.md"
  require_path "Python" "$PYTHON_BIN"
  require_path "初始 embedding 模型" "$BASE_EMBEDDING_MODEL"
  require_file "train 源数据" "$TRAIN_SOURCE"
  require_file "valid 源数据" "$VALID_SOURCE"
  require_file "test 源数据" "$TEST_SOURCE"
  require_file "item_info" "$ITEM_INFO"

  local save_steps
  save_steps=$(resolve_save_steps)
  print_plan "$save_steps"
  if flag_enabled "$PRINT_ARGS_ONLY"; then
    echo "只打印参数，不生成数据或启动 GPU 任务。"
    exit 0
  fi
  if ! flag_enabled "$CONFIRM_RUN"; then
    echo "未启动。确认上述参数后设置 CONFIRM_RUN=1。"
    exit 2
  fi

  cd "$ROOT"
  mkdir -p "$DATA_DIR" "$EMBEDDER_OUT" "$EVAL_DIR" "$REPORT_DIR"
  if flag_enabled "$FORCE_DATA" || [[ ! -s "$TRAIN_DATASET" ]]; then
    "$PYTHON_BIN" ablantion/scripts/build_cds_query_ablation.py \
      --examples "$TRAIN_SOURCE" \
      --item-info "$ITEM_INFO" \
      --output-dir "$DATA_DIR" \
      --output-prefix cds_train_query \
      --ablations title_store_categories \
      --max-item-chars "$MAX_ITEM_CHARS"
  fi
  if flag_enabled "$FORCE_DATA" || [[ ! -s "$VALID_DATASET" ]]; then
    "$PYTHON_BIN" ablantion/scripts/build_cds_query_ablation.py \
      --examples "$VALID_SOURCE" \
      --item-info "$ITEM_INFO" \
      --output-dir "$DATA_DIR" \
      --output-prefix cds_valid_query \
      --ablations title_store_categories \
      --max-item-chars "$MAX_ITEM_CHARS"
  fi
  if flag_enabled "$FORCE_DATA" || [[ ! -s "$TEST_DATASET" ]]; then
    "$PYTHON_BIN" ablantion/scripts/build_cds_query_ablation.py \
      --examples "$TEST_SOURCE" \
      --item-info "$ITEM_INFO" \
      --output-dir "$DATA_DIR" \
      --output-prefix cds_test_query \
      --ablations title_store_categories \
      --max-item-chars "$MAX_ITEM_CHARS"
  fi
  require_file "生成后的 train 数据" "$TRAIN_DATASET"
  require_file "生成后的 valid 数据" "$VALID_DATASET"
  require_file "生成后的 test 数据" "$TEST_DATASET"

  "$PYTHON_BIN" scripts/data/audit_embedding_retriever_dataset.py \
    --dataset "$TRAIN_DATASET" \
    --expected-split train \
    --expected-query-fields title,store,categories \
    --forbid-cot-tags \
    --batch-size "$BATCH_SIZE" \
    --output "$REPORT_DIR/train_data_audit.json"
  "$PYTHON_BIN" scripts/data/audit_embedding_retriever_dataset.py \
    --dataset "$VALID_DATASET" \
    --expected-split valid \
    --expected-query-fields title,store,categories \
    --forbid-cot-tags \
    --batch-size "$BATCH_SIZE" \
    --output "$REPORT_DIR/valid_data_audit.json"
  "$PYTHON_BIN" scripts/data/audit_embedding_retriever_dataset.py \
    --dataset "$TEST_DATASET" \
    --expected-split test \
    --expected-query-fields title,store,categories \
    --forbid-cot-tags \
    --batch-size "$BATCH_SIZE" \
    --output "$REPORT_DIR/test_data_audit.json"

  export PATH="$VENV/bin:$PATH"
  export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
  export NCCL_NET NCCL_IB_DISABLE NCCL_P2P_DISABLE NCCL_NVLS_ENABLE NCCL_MNNVL_ENABLE NCCL_COLLNET_ENABLE
  export NCCL_DEBUG TORCH_NCCL_ASYNC_ERROR_HANDLING PYTORCH_CUDA_ALLOC_CONF

  local done_file="$EMBEDDER_OUT/.train_done"
  if ! flag_enabled "$FORCE_TRAIN" && [[ -s "$done_file" ]]; then
    echo "跳过已完成训练: $EMBEDDER_OUT"
  else
    local multi_positive_arg=--no-multi-positive-targets
    local cross_gpu_arg=--no-cross-gpu-negatives
    if flag_enabled "$MULTI_POSITIVE_TARGETS"; then
      multi_positive_arg=--multi-positive-targets
    fi
    if flag_enabled "$CROSS_GPU_NEGATIVES"; then
      cross_gpu_arg=--cross-gpu-negatives
    fi
    local train_args=(
      scripts/embedding/train_phase0_embedder.py
      --model "$BASE_EMBEDDING_MODEL"
      --dataset "$TRAIN_DATASET"
      --output-dir "$EMBEDDER_OUT"
      --max-length "$MAX_LENGTH"
      --batch-size "$BATCH_SIZE"
      --grad-accum "$GRAD_ACCUM"
      --epochs "$EPOCHS"
      --max-steps "$MAX_STEPS"
      --learning-rate "$LEARNING_RATE"
      --save-steps "$save_steps"
      --torch-dtype "$TORCH_DTYPE"
      --gradient-checkpointing "$GRADIENT_CHECKPOINTING"
      --attn-implementation "$ATTN_IMPLEMENTATION"
      "$multi_positive_arg"
      "$cross_gpu_arg"
      --no-sync-barriers
      --seed "$SEED"
    )
    echo "开始 history-only retriever 训练。"
    if ((NPROC_PER_NODE > 1)); then
      CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PYTHON_BIN" -m torch.distributed.run \
        --nproc_per_node "$NPROC_PER_NODE" \
        --master_port "$MASTER_PORT" \
        "${train_args[@]}"
    else
      CUDA_VISIBLE_DEVICES="$CUDA_DEVICES" "$PYTHON_BIN" "${train_args[@]}"
    fi
    date -u +"%Y-%m-%dT%H:%M:%SZ" > "$done_file"
  fi

  run_eval valid "$VALID_DATASET" "$EMBEDDER_OUT" 'checkpoint-*' "$EVAL_DIR"
  "$PYTHON_BIN" scripts/eval/select_best_embedding_checkpoint.py \
    --eval-dir "$EVAL_DIR" \
    --metric "$SELECTION_METRIC" \
    --output "$BEST_SUMMARY"

  if flag_enabled "$RUN_REFERENCE_EVAL"; then
    require_path "旧 history-only checkpoint-83" "$PREVIOUS_HISTORY_CHECKPOINT"
    require_path "当前 reward checkpoint-83" "$CURRENT_REWARD_CHECKPOINT"
    run_eval valid "$VALID_DATASET" "$PREVIOUS_HISTORY_CHECKPOINT" 'checkpoint-*' "$EVAL_DIR/reference_previous_history"
    run_eval valid "$VALID_DATASET" "$CURRENT_REWARD_CHECKPOINT" 'checkpoint-*' "$EVAL_DIR/reference_current_reward"
  fi

  if flag_enabled "$RUN_TEST"; then
    local best_checkpoint
    best_checkpoint=$("$PYTHON_BIN" -c 'import json,sys; print(json.load(open(sys.argv[1]))["best_checkpoint"])' "$BEST_SUMMARY")
    require_path "valid 选出的最佳 checkpoint" "$best_checkpoint"
    run_eval test "$TEST_DATASET" "$best_checkpoint" 'checkpoint-*' "$EVAL_DIR/final_test"
  fi
}

main "$@"
