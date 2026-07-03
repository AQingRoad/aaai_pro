#!/usr/bin/env bash
set -euo pipefail

# Build and train a CoT-aware CDs_and_Vinyl embedding model.
# The generated dataset contains query/positive pairs for:
#   1) history -> target item
#   2) history + generated CoT -> target item

ROOT=${ROOT:-/mnt/tidal-sh01/usr/xiayu6/xiayu/aaai_pro}
VENV=${VENV:-/root/miniconda3/envs/swift}
PYTHON_BIN=${PYTHON_BIN:-$VENV/bin/python}

CATEGORY=${CATEGORY:-CDs_and_Vinyl}
OUT_DIR=${OUT_DIR:-$ROOT/outputs/rrec_amazon/$CATEGORY}
RREC_EVAL_DIR=${RREC_EVAL_DIR:-$ROOT/github_artifacts/CDs_and_Vinyl/rrec_eval}
ITEM_INFO=${ITEM_INFO:-$RREC_EVAL_DIR/item_info.jsonl}

COT_CANDIDATE_LISTS=${COT_CANDIDATE_LISTS:-$OUT_DIR/cot_candidate_one_lists_deepseek_v4_pro_low.jsonl}
COT_TEXT_MODE=${COT_TEXT_MODE:-answer}
COT_EMBEDDER_DATASET=${COT_EMBEDDER_DATASET:-$OUT_DIR/phase0_embedder_cds_with_cot_${COT_TEXT_MODE}.jsonl}
FORCE_REBUILD_DATASET=${FORCE_REBUILD_DATASET:-0}
INCLUDE_HISTORY=${INCLUDE_HISTORY:-1}
INCLUDE_COT=${INCLUDE_COT:-1}
MAX_COT_CHARS=${MAX_COT_CHARS:-1200}
MAX_ITEM_CHARS=${MAX_ITEM_CHARS:-0}
NEGATIVE_SAMPLING=${NEGATIVE_SAMPLING:-none}
NUM_NEGATIVES=${NUM_NEGATIVES:-0}
NEGATIVE_SEED=${NEGATIVE_SEED:-42}

BASE_EMBEDDING_MODEL=${BASE_EMBEDDING_MODEL:-/mnt/tidal-sh01/usr/xiayu6/xiayu/checkpoint/Qwen3_embedding/0.6B}
EMBEDDER_OUT=${EMBEDDER_OUT:-$ROOT/checkpoints/rrec_amazon_CDs_and_Vinyl/qwen3_embedding_cds_cot_tidal}

EMBEDDER_CUDA_VISIBLE_DEVICES=${EMBEDDER_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-0}}
EMBEDDER_NPROC_PER_NODE=${EMBEDDER_NPROC_PER_NODE:-auto}
EMBEDDER_MASTER_PORT=${EMBEDDER_MASTER_PORT:-29522}
EMBEDDER_BATCH_SIZE=${EMBEDDER_BATCH_SIZE:-128}
EMBEDDER_GRAD_ACCUM=${EMBEDDER_GRAD_ACCUM:-1}
EMBEDDER_MAX_LENGTH=${EMBEDDER_MAX_LENGTH:-2048}
EMBEDDER_EPOCHS=${EMBEDDER_EPOCHS:-1}
EMBEDDER_MAX_STEPS=${EMBEDDER_MAX_STEPS:--1}
EMBEDDER_LR=${EMBEDDER_LR:-3e-6}
EMBEDDER_SAVE_STEPS=${EMBEDDER_SAVE_STEPS:-auto}
EMBEDDER_TORCH_DTYPE=${EMBEDDER_TORCH_DTYPE:-bfloat16}
EMBEDDER_GRADIENT_CHECKPOINTING=${EMBEDDER_GRADIENT_CHECKPOINTING:-auto}
EMBEDDER_ATTN_IMPLEMENTATION=${EMBEDDER_ATTN_IMPLEMENTATION:-}
EMBEDDER_CROSS_GPU_NEGATIVES=${EMBEDDER_CROSS_GPU_NEGATIVES:-0}
EMBEDDER_PRINT_ARGS_ONLY=${EMBEDDER_PRINT_ARGS_ONLY:-0}
SEED=${SEED:-42}

require_path() {
  local label="$1"
  local path="$2"
  if [[ ! -e "$path" ]]; then
    echo "缺少 $label: $path" >&2
    exit 1
  fi
}

require_file() {
  local label="$1"
  local path="$2"
  if [[ ! -s "$path" ]]; then
    echo "缺少或为空 $label: $path" >&2
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

print_param() {
  local key="$1"
  local value="$2"
  local note="$3"
  printf "%s=%s # %s\n" "$key" "$value" "$note"
}

require_path "project root" "$ROOT"
require_path "python" "$PYTHON_BIN"
require_file "CoT candidate lists" "$COT_CANDIDATE_LISTS"
require_file "item info" "$ITEM_INFO"
require_path "base embedding model" "$BASE_EMBEDDING_MODEL"

mkdir -p "$OUT_DIR" "$EMBEDDER_OUT"
cd "$ROOT"

export PATH="$VENV/bin:$PATH"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export MODELSCOPE_CACHE=${MODELSCOPE_CACHE:-/mnt/tidal-sh01/usr/xiayu6/xiayu/modelscope_cache}

build_args=()
if [[ "$INCLUDE_HISTORY" == "1" || "$INCLUDE_HISTORY" == "true" ]]; then
  build_args+=(--include-history)
else
  build_args+=(--no-include-history)
fi
if [[ "$INCLUDE_COT" == "1" || "$INCLUDE_COT" == "true" ]]; then
  build_args+=(--include-cot)
else
  build_args+=(--no-include-cot)
fi

if [[ "$NEGATIVE_SAMPLING" != "none" && "$NUM_NEGATIVES" -gt 0 && -s "$COT_EMBEDDER_DATASET" ]]; then
  if ! grep -m 1 -q '"negatives"' "$COT_EMBEDDER_DATASET"; then
    echo "已有训练数据不含显式负样本，重新构建: $COT_EMBEDDER_DATASET"
    FORCE_REBUILD_DATASET=1
  fi
fi

if [[ "$FORCE_REBUILD_DATASET" == "1" || ! -s "$COT_EMBEDDER_DATASET" ]]; then
  "$PYTHON_BIN" scripts/data/make_cot_embedder_dataset.py \
    --candidate-lists "$COT_CANDIDATE_LISTS" \
    --item-info "$ITEM_INFO" \
    --output "$COT_EMBEDDER_DATASET" \
    --cot-text-mode "$COT_TEXT_MODE" \
    --max-cot-chars "$MAX_COT_CHARS" \
    --max-item-chars "$MAX_ITEM_CHARS" \
    --negative-sampling "$NEGATIVE_SAMPLING" \
    --num-negatives "$NUM_NEGATIVES" \
    --negative-seed "$NEGATIVE_SEED" \
    "${build_args[@]}"
else
  echo "使用已有 CoT 嵌入训练数据: $COT_EMBEDDER_DATASET"
fi

require_file "CoT embedder dataset" "$COT_EMBEDDER_DATASET"
COT_EMBEDDER_ROWS=$(wc -l < "$COT_EMBEDDER_DATASET" | tr -d ' ')

EMBEDDER_NPROC=$(resolve_nproc)
if ((EMBEDDER_NPROC < 1)); then
  echo "EMBEDDER_NPROC_PER_NODE 必须 >= 1" >&2
  exit 1
fi

if [[ "$EMBEDDER_SAVE_STEPS" == "auto" ]]; then
  global_train_batch=$((EMBEDDER_BATCH_SIZE * EMBEDDER_NPROC))
  full_batches=$((COT_EMBEDDER_ROWS / global_train_batch))
  if ((full_batches < 1)); then
    echo "至少需要一个完整全局批次: rows=$COT_EMBEDDER_ROWS per_device_batch_size=$EMBEDDER_BATCH_SIZE nproc=$EMBEDDER_NPROC" >&2
    exit 1
  fi
  EMBEDDER_SAVE_STEPS=$(((full_batches + EMBEDDER_GRAD_ACCUM - 1) / EMBEDDER_GRAD_ACCUM))
fi

echo "训练参数核对："
print_param "ROOT" "$ROOT" "项目根目录，用于执行脚本并解析相对产物路径。"
print_param "VENV" "$VENV" "Python 环境目录，决定训练和数据构建使用的依赖包。"
print_param "PYTHON_BIN" "$PYTHON_BIN" "运行数据构建脚本和训练脚本的 Python 可执行文件。"
print_param "CATEGORY" "$CATEGORY" "本次嵌入模型训练对应的 Amazon 类目。"
print_param "OUT_DIR" "$OUT_DIR" "该类目的输出目录，用于保存中间数据文件。"
print_param "RREC_EVAL_DIR" "$RREC_EVAL_DIR" "RRec 验证集、测试集和物品元数据所在目录。"
print_param "ITEM_INFO" "$ITEM_INFO" "物品元数据 JSONL，用于构造正样本和可选负样本文本。"
print_param "COT_CANDIDATE_LISTS" "$COT_CANDIDATE_LISTS" "作为训练来源的原始或聚合后的 CoT 候选列表 JSONL。"
print_param "COT_EMBEDDER_DATASET" "$COT_EMBEDDER_DATASET" "嵌入模型训练脚本读取的查询/正样本 JSONL。"
print_param "COT_EMBEDDER_ROWS" "$COT_EMBEDDER_ROWS" "数据构建后得到的查询/正样本训练行数。"
print_param "COT_TEXT_MODE" "$COT_TEXT_MODE" "追加到用户历史后的 CoT 字段，取值为 answer、think、tagged 或 full。"
print_param "INCLUDE_HISTORY" "$INCLUDE_HISTORY" "是否加入仅历史查询到目标物品的训练样本。"
print_param "INCLUDE_COT" "$INCLUDE_COT" "是否加入历史加 CoT 查询到目标物品的训练样本。"
print_param "MAX_COT_CHARS" "$MAX_COT_CHARS" "每条样本中保留的生成 CoT 最大字符数，0 表示不限制。"
print_param "MAX_ITEM_CHARS" "$MAX_ITEM_CHARS" "目标物品或负样本物品文本保留的最大字符数，0 表示不限制。"
print_param "NEGATIVE_SAMPLING" "$NEGATIVE_SAMPLING" "每条查询的显式负样本采样策略。"
print_param "NUM_NEGATIVES" "$NUM_NEGATIVES" "启用负样本采样时每条查询采样的显式负样本数。"
print_param "NEGATIVE_SEED" "$NEGATIVE_SEED" "显式负样本采样使用的随机种子。"
print_param "BASE_EMBEDDING_MODEL" "$BASE_EMBEDDING_MODEL" "微调前加载的 Qwen3 嵌入模型初始检查点。"
print_param "EMBEDDER_OUT" "$EMBEDDER_OUT" "微调检查点和 phase0_args.json 写入目录。"
print_param "EMBEDDER_CUDA_VISIBLE_DEVICES" "$EMBEDDER_CUDA_VISIBLE_DEVICES" "暴露给训练进程的 GPU 编号。"
print_param "EMBEDDER_NPROC_PER_NODE" "$EMBEDDER_NPROC_PER_NODE" "请求的本机训练进程数，auto 表示按可见 GPU 数自动设置。"
print_param "EMBEDDER_NPROC" "$EMBEDDER_NPROC" "本次实际解析出的本机训练进程数。"
print_param "EMBEDDER_MASTER_PORT" "$EMBEDDER_MASTER_PORT" "多进程训练时 torch distributed 使用的通信端口。"
print_param "EMBEDDER_BATCH_SIZE" "$EMBEDDER_BATCH_SIZE" "每个训练进程的数据加载批大小，未计入梯度累积。"
print_param "EMBEDDER_GLOBAL_BATCH_SIZE" "$((EMBEDDER_BATCH_SIZE * EMBEDDER_NPROC * EMBEDDER_GRAD_ACCUM))" "跨进程并计入梯度累积后的每次优化器更新样本数。"
print_param "EMBEDDER_GRAD_ACCUM" "$EMBEDDER_GRAD_ACCUM" "执行一次优化器更新前累积的小批次数。"
print_param "EMBEDDER_MAX_LENGTH" "$EMBEDDER_MAX_LENGTH" "查询和文档文本的分词器截断长度。"
print_param "EMBEDDER_EPOCHS" "$EMBEDDER_EPOCHS" "未设置最大步数时，训练数据加载器遍历轮数。"
print_param "EMBEDDER_MAX_STEPS" "$EMBEDDER_MAX_STEPS" "优化器更新步数上限，-1 表示按训练轮数推导。"
print_param "EMBEDDER_LR" "$EMBEDDER_LR" "AdamW 优化器学习率。"
print_param "EMBEDDER_SAVE_STEPS" "$EMBEDDER_SAVE_STEPS" "按优化器更新步数计算的检查点保存间隔。"
print_param "EMBEDDER_TORCH_DTYPE" "$EMBEDDER_TORCH_DTYPE" "transformers 加载模型时请求的计算数据类型。"
print_param "EMBEDDER_GRADIENT_CHECKPOINTING" "$EMBEDDER_GRADIENT_CHECKPOINTING" "激活检查点模式，用额外计算换取更低显存占用。"
print_param "EMBEDDER_ATTN_IMPLEMENTATION" "${EMBEDDER_ATTN_IMPLEMENTATION:-default}" "transformers 模型加载时使用的 attention 后端，空值表示使用模型默认实现，flash_attention_2 表示尝试启用 FlashAttention 2。"
print_param "EMBEDDER_CROSS_GPU_NEGATIVES" "$EMBEDDER_CROSS_GPU_NEGATIVES" "是否把其他 DDP 进程的文档向量加入批内负样本池。"
print_param "FORCE_REBUILD_DATASET" "$FORCE_REBUILD_DATASET" "训练前是否强制重建 COT_EMBEDDER_DATASET。"
print_param "EMBEDDER_PRINT_ARGS_ONLY" "$EMBEDDER_PRINT_ARGS_ONLY" "为 true 时只打印带注释参数表，不启动训练。"
print_param "SEED" "$SEED" "数据打乱和 torch 初始化使用的基础随机种子。"

train_args=(
  scripts/embedding/train_phase0_embedder.py
  --model "$BASE_EMBEDDING_MODEL" \
  --dataset "$COT_EMBEDDER_DATASET" \
  --output-dir "$EMBEDDER_OUT" \
  --max-length "$EMBEDDER_MAX_LENGTH" \
  --batch-size "$EMBEDDER_BATCH_SIZE" \
  --grad-accum "$EMBEDDER_GRAD_ACCUM" \
  --epochs "$EMBEDDER_EPOCHS" \
  --max-steps "$EMBEDDER_MAX_STEPS" \
  --learning-rate "$EMBEDDER_LR" \
  --torch-dtype "$EMBEDDER_TORCH_DTYPE" \
  --save-steps "$EMBEDDER_SAVE_STEPS" \
  --gradient-checkpointing "$EMBEDDER_GRADIENT_CHECKPOINTING" \
  --attn-implementation "$EMBEDDER_ATTN_IMPLEMENTATION" \
  --seed "$SEED"
)
if [[ "$EMBEDDER_CROSS_GPU_NEGATIVES" == "1" || "$EMBEDDER_CROSS_GPU_NEGATIVES" == "true" ]]; then
  train_args+=(--cross-gpu-negatives)
else
  train_args+=(--no-cross-gpu-negatives)
fi

if [[ "$EMBEDDER_PRINT_ARGS_ONLY" == "1" || "$EMBEDDER_PRINT_ARGS_ONLY" == "true" ]]; then
  echo "未启动训练。"
  exit 0
fi

if ((EMBEDDER_NPROC > 1)); then
  CUDA_VISIBLE_DEVICES="$EMBEDDER_CUDA_VISIBLE_DEVICES" "$PYTHON_BIN" -m torch.distributed.run \
    --nproc_per_node "$EMBEDDER_NPROC" \
    --master_port "$EMBEDDER_MASTER_PORT" \
    "${train_args[@]}"
else
  CUDA_VISIBLE_DEVICES="$EMBEDDER_CUDA_VISIBLE_DEVICES" "$PYTHON_BIN" "${train_args[@]}"
fi

latest_checkpoint=$(find "$EMBEDDER_OUT" -type d -name 'checkpoint-*' -print 2>/dev/null | sort -V | tail -n 1)
if [[ -n "$latest_checkpoint" ]]; then
  echo "QWEN3_EMBEDDING_MODEL=$latest_checkpoint"
fi
