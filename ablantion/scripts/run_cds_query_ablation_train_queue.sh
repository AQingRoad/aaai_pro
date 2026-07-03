#!/usr/bin/env bash
set -euo pipefail

# Sequentially train and evaluate the CDs query-side ablation embedding models.
# One job starts immediately after the previous job finishes, so a single GPU
# does not sit idle between ablation runs.

ROOT=${ROOT:-/home/user/aaai_pro}
VENV=${VENV:-/home/user/.conda/envs/aaai_pro}
PYTHON_BIN=${PYTHON_BIN:-$VENV/bin/python}

BASE_EMBEDDING_MODEL=${BASE_EMBEDDING_MODEL:-/home/user/models_hf/Qwen3-Embedding-0.6B}
ITEM_INFO=${ITEM_INFO:-$ROOT/github_artifacts/CDs_and_Vinyl/rrec_eval/item_info.jsonl}

TRAIN_DATA_DIR=${TRAIN_DATA_DIR:-$ROOT/ablantion/datas/processed_datas/cds_query_ablation}
TEST_DATA_DIR=${TEST_DATA_DIR:-$ROOT/ablantion/datas/processed_datas/cds_query_ablation_test}
RUN_ROOT=${RUN_ROOT:-$ROOT/checkpoints/rrec_amazon_CDs_and_Vinyl/cds_query_ablation}
EVAL_ROOT=${EVAL_ROOT:-$ROOT/outputs/rrec_amazon/eval/CDs_and_Vinyl/cds_query_ablation}
LOG_DIR=${LOG_DIR:-$ROOT/ablantion/outputs/cds_query_ablation_train_queue/logs}

ABLATIONS=${ABLATIONS:-all}
CONFIRM_RUN=${CONFIRM_RUN:-0}
PRINT_ARGS_ONLY=${PRINT_ARGS_ONLY:-0}
SKIP_DONE=${SKIP_DONE:-1}
RUN_EVAL=${RUN_EVAL:-1}
FORCE_TRAIN=${FORCE_TRAIN:-0}
FORCE_EVAL=${FORCE_EVAL:-0}

EMBEDDER_CUDA_VISIBLE_DEVICES=${EMBEDDER_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}
EMBEDDER_NPROC_PER_NODE=${EMBEDDER_NPROC_PER_NODE:-auto}
EMBEDDER_MASTER_PORT=${EMBEDDER_MASTER_PORT:-29531}
EMBEDDER_BATCH_SIZE=${EMBEDDER_BATCH_SIZE:-128}
EMBEDDER_GRAD_ACCUM=${EMBEDDER_GRAD_ACCUM:-1}
EMBEDDER_MAX_LENGTH=${EMBEDDER_MAX_LENGTH:-4096}
EMBEDDER_EPOCHS=${EMBEDDER_EPOCHS:-5}
EMBEDDER_MAX_STEPS=${EMBEDDER_MAX_STEPS:--1}
EMBEDDER_LR=${EMBEDDER_LR:-6e-6}
EMBEDDER_SAVE_STEPS=${EMBEDDER_SAVE_STEPS:-auto}
EMBEDDER_TORCH_DTYPE=${EMBEDDER_TORCH_DTYPE:-bfloat16}
EMBEDDER_GRADIENT_CHECKPOINTING=${EMBEDDER_GRADIENT_CHECKPOINTING:-auto}
EMBEDDER_ATTN_IMPLEMENTATION=${EMBEDDER_ATTN_IMPLEMENTATION:-flash_attention_2}
EMBEDDER_CROSS_GPU_NEGATIVES=${EMBEDDER_CROSS_GPU_NEGATIVES:-0}
EMBEDDER_SYNC_BARRIERS=${EMBEDDER_SYNC_BARRIERS:-0}

EVAL_SPLIT=${EVAL_SPLIT:-test}
EVAL_CHECKPOINT_PATTERN=${EVAL_CHECKPOINT_PATTERN:-checkpoint-*}
EVAL_MAX_EXAMPLES=${EVAL_MAX_EXAMPLES:-0}
EVAL_EMBEDDING_MAX_LENGTH=${EVAL_EMBEDDING_MAX_LENGTH:-4096}
EVAL_EMBEDDING_BATCH_SIZE=${EVAL_EMBEDDING_BATCH_SIZE:-256}
EVAL_EMBEDDING_DEVICE=${EVAL_EMBEDDING_DEVICE:-cuda:0}
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
SEED=${SEED:-42}

ALL_ABLATIONS=(
  title_only
  title_store
  title_store_categories
  title_store_categories_features
  title_store_categories_features_description
  full_compact_no_all_ratings
)

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

count_devices() {
  local raw="$1"
  if [[ -z "$raw" ]]; then
    echo 1
    return
  fi
  local IFS=','
  local devices=()
  read -r -a devices <<< "$raw"
  echo "${#devices[@]}"
}

resolve_nproc() {
  if [[ "$EMBEDDER_NPROC_PER_NODE" == "auto" ]]; then
    count_devices "$EMBEDDER_CUDA_VISIBLE_DEVICES"
  else
    echo "$EMBEDDER_NPROC_PER_NODE"
  fi
}

is_selected() {
  local name="$1"
  if [[ "$ABLATIONS" == "all" ]]; then
    return 0
  fi
  local IFS=','
  local selected=()
  read -r -a selected <<< "$ABLATIONS"
  for item in "${selected[@]}"; do
    if [[ "$item" == "$name" ]]; then
      return 0
    fi
  done
  return 1
}

latest_checkpoint() {
  local run_dir="$1"
  find "$run_dir" -maxdepth 1 -type d -name 'checkpoint-*' -print 2>/dev/null | sort -V | tail -n 1
}

auto_save_steps() {
  local dataset="$1"
  local nproc="$2"
  if [[ "$EMBEDDER_SAVE_STEPS" != "auto" ]]; then
    echo "$EMBEDDER_SAVE_STEPS"
    return
  fi
  local rows
  rows=$(wc -l < "$dataset" | tr -d ' ')
  local global_batch=$((EMBEDDER_BATCH_SIZE * nproc))
  local full_batches=$((rows / global_batch))
  if ((full_batches < 1)); then
    echo "数据行数不足一个完整全局批次: rows=${rows}, batch=${EMBEDDER_BATCH_SIZE}, nproc=${nproc}" >&2
    exit 1
  fi
  echo $(((full_batches + EMBEDDER_GRAD_ACCUM - 1) / EMBEDDER_GRAD_ACCUM))
}

print_param() {
  local key="$1"
  local value="$2"
  local note="$3"
  printf "%s=%s # %s\n" "$key" "$value" "$note"
}

selected_ablation_names() {
  for name in "${ALL_ABLATIONS[@]}"; do
    if is_selected "$name"; then
      echo "$name"
    fi
  done
}

print_plan() {
  local nproc="$1"
  echo "CDs query 消融训练队列参数核对："
  print_param "ROOT" "$ROOT" "项目根目录，脚本在这里执行训练和评测。"
  print_param "VENV" "$VENV" "Python/Conda 环境目录。"
  print_param "PYTHON_BIN" "$PYTHON_BIN" "训练和评测使用的 Python 可执行文件。"
  print_param "BASE_EMBEDDING_MODEL" "$BASE_EMBEDDING_MODEL" "所有消融模型共用的初始 Qwen3 embedding 模型。"
  print_param "TRAIN_DATA_DIR" "$TRAIN_DATA_DIR" "训练集 query/positive 消融 JSONL 所在目录。"
  print_param "TEST_DATA_DIR" "$TEST_DATA_DIR" "测试集 query/positive 消融 JSONL 所在目录。"
  print_param "ITEM_INFO" "$ITEM_INFO" "评测候选 item 文本来源。"
  print_param "RUN_ROOT" "$RUN_ROOT" "每个消融模型 checkpoint 写入根目录。"
  print_param "EVAL_ROOT" "$EVAL_ROOT" "每个消融模型测试指标写入根目录。"
  print_param "LOG_DIR" "$LOG_DIR" "每个训练/评测子任务的日志目录。"
  print_param "ABLATIONS" "$ABLATIONS" "要训练的消融列表，all 表示 6 个全跑。"
  print_param "RUN_EVAL" "$RUN_EVAL" "每个模型训练完成后是否立即用对应测试集评测。"
  print_param "SKIP_DONE" "$SKIP_DONE" "已成功完成并带 .train_done/.eval_done 标记的任务是否跳过。"
  print_param "FORCE_TRAIN" "$FORCE_TRAIN" "是否忽略 .train_done 标记并重新训练。"
  print_param "FORCE_EVAL" "$FORCE_EVAL" "是否忽略已有评测结果并重新评测。"
  print_param "EMBEDDER_CUDA_VISIBLE_DEVICES" "$EMBEDDER_CUDA_VISIBLE_DEVICES" "暴露给训练进程的 GPU 编号。"
  print_param "EMBEDDER_NPROC_PER_NODE" "$EMBEDDER_NPROC_PER_NODE" "每个节点启动的训练进程数，auto 表示按可见 GPU 数解析。"
  print_param "EMBEDDER_NPROC" "$nproc" "本次实际使用的训练进程数。"
  print_param "EMBEDDER_BATCH_SIZE" "$EMBEDDER_BATCH_SIZE" "每个训练进程的 batch size。"
  print_param "EMBEDDER_GRAD_ACCUM" "$EMBEDDER_GRAD_ACCUM" "梯度累积步数。"
  print_param "EMBEDDER_MAX_LENGTH" "$EMBEDDER_MAX_LENGTH" "训练时 query 和 positive 的 tokenizer 截断长度。"
  print_param "EMBEDDER_EPOCHS" "$EMBEDDER_EPOCHS" "每个消融模型训练轮数。"
  print_param "EMBEDDER_MAX_STEPS" "$EMBEDDER_MAX_STEPS" "每个消融模型最大优化步数，-1 表示按 epoch 推导。"
  print_param "EMBEDDER_LR" "$EMBEDDER_LR" "AdamW 学习率。"
  print_param "EMBEDDER_SAVE_STEPS" "$EMBEDDER_SAVE_STEPS" "checkpoint 保存间隔，auto 表示按每轮步数保存。"
  print_param "EMBEDDER_TORCH_DTYPE" "$EMBEDDER_TORCH_DTYPE" "模型加载和训练使用的数据类型。"
  print_param "EMBEDDER_GRADIENT_CHECKPOINTING" "$EMBEDDER_GRADIENT_CHECKPOINTING" "是否开启 gradient checkpointing。"
  print_param "EMBEDDER_ATTN_IMPLEMENTATION" "$EMBEDDER_ATTN_IMPLEMENTATION" "transformers attention 后端。"
  print_param "EMBEDDER_CROSS_GPU_NEGATIVES" "$EMBEDDER_CROSS_GPU_NEGATIVES" "多 GPU 时是否使用跨 GPU batch negatives。"
  print_param "EVAL_SPLIT" "$EVAL_SPLIT" "评测 split 标签。"
  print_param "EVAL_QUERY_MODE" "user_history" "评测直接读取测试 JSONL 内的 query，保证训推 query 消融口径一致。"
  print_param "EVAL_EMBEDDING_MAX_LENGTH" "$EVAL_EMBEDDING_MAX_LENGTH" "评测 query/item embedding 的 tokenizer 截断长度。"
  print_param "EVAL_EMBEDDING_BATCH_SIZE" "$EVAL_EMBEDDING_BATCH_SIZE" "评测 embedding batch size。"
  print_param "KS" "$KS" "评测输出的 HR/NDCG cutoff。"
  print_param "CONFIRM_RUN" "$CONFIRM_RUN" "必须为 1 才会真正启动训练队列。"
  print_param "PRINT_ARGS_ONLY" "$PRINT_ARGS_ONLY" "为 1 时只打印参数和任务列表，不启动训练。"
  print_param "SEED" "$SEED" "训练随机种子。"
  echo "将运行的消融任务："
  selected_ablation_names | sed 's/^/  - /'
}

run_train_one() {
  local name="$1"
  local index="$2"
  local nproc="$3"
  local train_file="$TRAIN_DATA_DIR/cds_query_${name}.jsonl"
  local run_dir="$RUN_ROOT/$name"
  local log_file="$LOG_DIR/train_${index}_${name}.log"
  local done_file="$run_dir/.train_done"
  local save_steps
  save_steps=$(auto_save_steps "$train_file" "$nproc")

  require_file "训练数据 ${name}" "$train_file"
  mkdir -p "$run_dir" "$LOG_DIR"

  if [[ "$FORCE_TRAIN" != "1" && "$SKIP_DONE" == "1" && -s "$done_file" ]]; then
    echo "跳过已完成训练: ${name} -> ${run_dir}"
    return
  fi

  echo "开始训练: ${name}"
  (
    cd "$ROOT"
    export PATH="$VENV/bin:$PATH"
    export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
    export NCCL_NET NCCL_IB_DISABLE NCCL_P2P_DISABLE NCCL_NVLS_ENABLE NCCL_MNNVL_ENABLE NCCL_COLLNET_ENABLE
    export NCCL_DEBUG TORCH_NCCL_ASYNC_ERROR_HANDLING PYTORCH_CUDA_ALLOC_CONF

    train_args=(
      scripts/embedding/train_phase0_embedder.py
      --model "$BASE_EMBEDDING_MODEL"
      --dataset "$train_file"
      --output-dir "$run_dir"
      --max-length "$EMBEDDER_MAX_LENGTH"
      --batch-size "$EMBEDDER_BATCH_SIZE"
      --grad-accum "$EMBEDDER_GRAD_ACCUM"
      --epochs "$EMBEDDER_EPOCHS"
      --max-steps "$EMBEDDER_MAX_STEPS"
      --learning-rate "$EMBEDDER_LR"
      --torch-dtype "$EMBEDDER_TORCH_DTYPE"
      --save-steps "$save_steps"
      --gradient-checkpointing "$EMBEDDER_GRADIENT_CHECKPOINTING"
      --attn-implementation "$EMBEDDER_ATTN_IMPLEMENTATION"
      --seed "$SEED"
    )
    if [[ "$EMBEDDER_CROSS_GPU_NEGATIVES" == "1" || "$EMBEDDER_CROSS_GPU_NEGATIVES" == "true" ]]; then
      train_args+=(--cross-gpu-negatives)
    else
      train_args+=(--no-cross-gpu-negatives)
    fi
    if [[ "$EMBEDDER_SYNC_BARRIERS" == "1" || "$EMBEDDER_SYNC_BARRIERS" == "true" ]]; then
      train_args+=(--sync-barriers)
    else
      train_args+=(--no-sync-barriers)
    fi

    if ((nproc > 1)); then
      CUDA_VISIBLE_DEVICES="$EMBEDDER_CUDA_VISIBLE_DEVICES" "$PYTHON_BIN" -m torch.distributed.run \
        --nproc_per_node "$nproc" \
        --master_port "$((EMBEDDER_MASTER_PORT + index))" \
        "${train_args[@]}"
    else
      CUDA_VISIBLE_DEVICES="$EMBEDDER_CUDA_VISIBLE_DEVICES" "$PYTHON_BIN" "${train_args[@]}"
    fi
  ) 2>&1 | tee "$log_file"

  local ckpt
  ckpt=$(latest_checkpoint "$run_dir")
  if [[ -z "$ckpt" ]]; then
    echo "训练结束但未找到 checkpoint: ${run_dir}" >&2
    exit 1
  fi
  date -u +"%Y-%m-%dT%H:%M:%SZ" > "$done_file"
  echo "训练完成: ${name} latest_checkpoint=${ckpt}"
}

run_eval_one() {
  local name="$1"
  local index="$2"
  local test_file="$TEST_DATA_DIR/cds_test_query_${name}.jsonl"
  local run_dir="$RUN_ROOT/$name"
  local eval_dir="$EVAL_ROOT/$name"
  local log_file="$LOG_DIR/eval_${index}_${name}.log"
  local done_file="$eval_dir/.eval_done"

  require_file "测试数据 ${name}" "$test_file"
  require_path "checkpoint 目录 ${name}" "$run_dir"
  mkdir -p "$eval_dir" "$LOG_DIR"

  if [[ "$FORCE_EVAL" != "1" && "$SKIP_DONE" == "1" && -s "$done_file" ]]; then
    echo "跳过已完成评测: ${name} -> ${eval_dir}"
    return
  fi

  echo "开始评测: ${name}"
  (
    cd "$ROOT"
    ROOT="$ROOT" \
    VENV="$VENV" \
    CATEGORY="CDs_and_Vinyl" \
    SPLIT="$EVAL_SPLIT" \
    MAX_EXAMPLES="$EVAL_MAX_EXAMPLES" \
    CUDA_VISIBLE_DEVICES="$EMBEDDER_CUDA_VISIBLE_DEVICES" \
    CHECKPOINT_ROOT="$run_dir" \
    CHECKPOINT_PATTERN="$EVAL_CHECKPOINT_PATTERN" \
    EVAL_DIR="$eval_dir" \
    EVAL_EXAMPLES="$test_file" \
    ITEM_INFO="$ITEM_INFO" \
    EMBEDDING_MAX_LENGTH="$EVAL_EMBEDDING_MAX_LENGTH" \
    EMBEDDING_BATCH_SIZE="$EVAL_EMBEDDING_BATCH_SIZE" \
    EMBEDDING_TORCH_DTYPE="$EMBEDDER_TORCH_DTYPE" \
    EMBEDDING_DEVICE="$EVAL_EMBEDDING_DEVICE" \
    EVAL_QUERY_MODE="user_history" \
    EVAL_REQUIRE_COT="0" \
    KS="$KS" \
    FORCE_EVAL="$FORCE_EVAL" \
    bash scripts/embedding/run_eval_cds_embedding_checkpoints_tidal.sh
  ) 2>&1 | tee "$log_file"

  date -u +"%Y-%m-%dT%H:%M:%SZ" > "$done_file"
  echo "评测完成: ${name} -> ${eval_dir}"
}

main() {
  require_path "项目根目录" "$ROOT"
  require_path "Python 环境" "$VENV"
  require_path "Python 可执行文件" "$PYTHON_BIN"
  require_path "初始 embedding 模型" "$BASE_EMBEDDING_MODEL"
  require_file "item_info" "$ITEM_INFO"

  local nproc
  nproc=$(resolve_nproc)
  if ((nproc < 1)); then
    echo "EMBEDDER_NPROC_PER_NODE 必须 >= 1" >&2
    exit 1
  fi

  print_plan "$nproc"
  if [[ "$PRINT_ARGS_ONLY" == "1" || "$PRINT_ARGS_ONLY" == "true" ]]; then
    echo "只打印参数，不启动训练。"
    exit 0
  fi
  if [[ "$CONFIRM_RUN" != "1" && "$CONFIRM_RUN" != "true" ]]; then
    echo "未启动训练。请设置 CONFIRM_RUN=1 后再次运行。"
    exit 2
  fi

  mkdir -p "$RUN_ROOT" "$EVAL_ROOT" "$LOG_DIR"
  local index=0
  local name
  while IFS= read -r name; do
    index=$((index + 1))
    run_train_one "$name" "$index" "$nproc"
    if [[ "$RUN_EVAL" == "1" || "$RUN_EVAL" == "true" ]]; then
      run_eval_one "$name" "$index"
    fi
  done < <(selected_ablation_names)

  echo "全部消融任务完成。"
}

main "$@"
