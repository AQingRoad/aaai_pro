#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/user/aaai_pro}
VENV=${VENV:-/home/user/.conda/envs/aaai_pro}
MODEL=${MODEL:-}

SPLIT_DIR=${SPLIT_DIR:-$ROOT/manu_src/datas/CDs_and_Vinyl/sft_grpo/disjoint_example20_80__input_time_title_rating_store_categories_desc256_details256_v1}
GRPO_SOURCE=${GRPO_SOURCE:-$SPLIT_DIR/grpo_train80_seed42_n8578.jsonl}
SFT_EXAMPLE_IDS=${SFT_EXAMPLE_IDS:-$SPLIT_DIR/sft_example_ids.txt}
DATASET=${DATASET:-$SPLIT_DIR/grpo_cot_sim_ndcg_messages_train80_seed42_n8578.jsonl}
ITEM_INFO=${ITEM_INFO:-$ROOT/manu_src/datas/CDs_and_Vinyl/arrow_to_jsonls/item_info.jsonl}
REWARD_PLUGIN=${REWARD_PLUGIN:-$ROOT/manu_src/scripts/train/cot_sim_ndcg_reward.py}
REWARD_EMBEDDING=${REWARD_EMBEDDING:-$ROOT/manu_src/model_outputs/CDs_and_Vinyl/embedding/qwen3emb06b_history_plus_glm52_non_target_cot_input_time_title_rating_store_categories_desc256_details256_v1_bs128_ga1_lr2e5_ep5_len4096_seed42/checkpoint-epoch-01}
OUT=${OUT:-$ROOT/manu_src/model_outputs/CDs_and_Vinyl/grpo/qwen25_3b_fullsft20_grpolora80_cottrained_logsoftmaxsim_w0p8_ndcg100_w0p2_g4_genbs16_bs4_ga1_vllmsleep1_vllm0p10_lr2e5_ep1_vllmlen4608_clen512_seed42}

EXPECTED_ROWS=${EXPECTED_ROWS:-8578}
SEED=${SEED:-42}
NUM_GENERATIONS=${NUM_GENERATIONS:-4}
GENERATION_BATCH_SIZE=${GENERATION_BATCH_SIZE:-16}
BATCH_SIZE=${BATCH_SIZE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-1}
MAX_LENGTH=${MAX_LENGTH:-4096}
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-512}
GENERATION_TEMPERATURE=${GENERATION_TEMPERATURE:-1.0}
GENERATION_TOP_K=${GENERATION_TOP_K:-200}
GENERATION_TOP_P=${GENERATION_TOP_P:-1.0}
EPOCHS=${EPOCHS:-1}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
WARMUP_STEPS=${WARMUP_STEPS:-64}
BETA=${BETA:-0.04}
LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-128}
SAVE_STEPS=${SAVE_STEPS:-300}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-30}

COT_SIM_NDCG_TEMPERATURE=${COT_SIM_NDCG_TEMPERATURE:-0.05}
COT_SIM_NDCG_K=${COT_SIM_NDCG_K:-100}
COT_SIM_NDCG_SIM_WEIGHT=${COT_SIM_NDCG_SIM_WEIGHT:-0.8}
COT_SIM_NDCG_NDCG_WEIGHT=${COT_SIM_NDCG_NDCG_WEIGHT:-0.2}
COT_SIM_NDCG_ITEM_BATCH_SIZE=${COT_SIM_NDCG_ITEM_BATCH_SIZE:-128}
COT_SIM_NDCG_QUERY_BATCH_SIZE=${COT_SIM_NDCG_QUERY_BATCH_SIZE:-16}
COT_SIM_NDCG_LOG_EVERY=${COT_SIM_NDCG_LOG_EVERY:-1}
COT_SIM_NDCG_COMPONENT_LOG=${COT_SIM_NDCG_COMPONENT_LOG:-$OUT/cot_sim_ndcg_components.jsonl}

USE_VLLM=${USE_VLLM:-true}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.10}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-4608}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-16}
VLLM_SLEEP_LEVEL=${VLLM_SLEEP_LEVEL:-1}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
CONFIRM_RUN=${CONFIRM_RUN:-0}

print_param() {
  local name=$1 value=$2 description=$3
  printf '  %-38s %s\n      %s\n' "$name=$value" "" "$description"
}

require_path() {
  local label=$1 path=$2
  if [[ ! -e "$path" ]]; then
    echo "缺少${label}: $path" >&2
    exit 1
  fi
}

if [[ -z "$MODEL" ]]; then
  echo "必须用 MODEL 指定前一阶段保存的全参数 SFT checkpoint。" >&2
  exit 1
fi
for dependency in \
  "$ROOT/AGENTS.md" \
  "$VENV/bin/python" \
  "$VENV/bin/swift" \
  "$MODEL" \
  "$GRPO_SOURCE" \
  "$SFT_EXAMPLE_IDS" \
  "$DATASET" \
  "$ITEM_INFO" \
  "$REWARD_PLUGIN" \
  "$REWARD_EMBEDDING"; do
  require_path "GRPO 依赖" "$dependency"
done
if [[ -e "$MODEL/adapter_config.json" ]]; then
  echo "MODEL 指向了 LoRA adapter；本实验要求完整 SFT 模型作为 GRPO 基座。" >&2
  exit 1
fi
if [[ ! -e "$MODEL/config.json" ]]; then
  echo "完整 SFT checkpoint 缺少 config.json: $MODEL" >&2
  exit 1
fi
if [[ "$SEED" != "42" ]]; then
  echo "项目随机种子必须为 42。" >&2
  exit 1
fi
if [[ "$GRAD_ACCUM" != "1" ]]; then
  echo "当前实验固定 gradient_accumulation_steps=1。" >&2
  exit 1
fi
if [[ "$NUM_GENERATIONS" != "4" ]]; then
  echo "当前实验固定每个 prompt 生成 4 条 completion。" >&2
  exit 1
fi
if ((BATCH_SIZE % NUM_GENERATIONS != 0)); then
  echo "BATCH_SIZE 必须能被 NUM_GENERATIONS 整除。" >&2
  exit 1
fi
if ((GENERATION_BATCH_SIZE % NUM_GENERATIONS != 0)); then
  echo "GENERATION_BATCH_SIZE 必须能被 NUM_GENERATIONS 整除。" >&2
  exit 1
fi
if ((GENERATION_BATCH_SIZE % BATCH_SIZE != 0)); then
  echo "GENERATION_BATCH_SIZE 必须能被 BATCH_SIZE 整除。" >&2
  exit 1
fi
if [[ "$MAX_LENGTH" != "4096" || "$MAX_COMPLETION_LENGTH" != "512" ]]; then
  echo "当前实验固定 prompt max_length=4096、completion max_length=512。" >&2
  exit 1
fi
if [[ "$VLLM_MAX_MODEL_LEN" != "4608" ]]; then
  echo "vLLM 总上下文必须为 4096 + 512 = 4608。" >&2
  exit 1
fi

source_rows=$(wc -l < "$GRPO_SOURCE" | tr -d ' ')
dataset_rows=$(wc -l < "$DATASET" | tr -d ' ')
if [[ "$source_rows" != "$EXPECTED_ROWS" || "$dataset_rows" != "$EXPECTED_ROWS" ]]; then
  echo "GRPO 原始数据/messages 均应为 $EXPECTED_ROWS 条，当前为 $source_rows/$dataset_rows。" >&2
  exit 1
fi

prompts_per_step=$((BATCH_SIZE / NUM_GENERATIONS * GRAD_ACCUM))
approx_steps=$(((EXPECTED_ROWS + prompts_per_step - 1) / prompts_per_step))
steps_per_generation=$((GENERATION_BATCH_SIZE / BATCH_SIZE))

echo "Qwen2.5-3B 全参数 SFT 基座上的 LoRA GRPO 参数："
print_param ROOT "$ROOT" "A100 项目根目录；正式运行前重新读取 AGENTS.md。"
print_param MODEL "$MODEL" "前 20% 数据全参数 SFT 产生的完整模型，同时充当 GRPO reference policy。"
print_param BASE_ADAPTER none "GRPO 启动时不加载 SFT adapter，避免 SFT LoRA 在 vLLM 中重复叠加。"
print_param GRPO_SOURCE "$GRPO_SOURCE" "与 SFT example_id 互斥的后 80% 原始样本，共 $source_rows 条。"
print_param DATASET "$DATASET" "policy messages 只含 history；target_item_id 仅作为 reward metadata。"
print_param INPUT_SCHEMA time_title_rating_store_categories_desc256_details256_v1 "与 SFT、reward embedding 和后续评测统一的 history 口径。"
print_param ITEM_INFO "$ITEM_INFO" "构造冻结的 12000-item 全候选集合；target/item 文本不截断。"
print_param REWARD_EMBEDDING "$REWARD_EMBEDDING" "冻结的无 Mask CoT-trained embedding epoch-01 checkpoint。"
print_param REWARD_FUNC cot_sim_ndcg "唯一 reward；不增加格式、margin、rubric 或 reference CoT 奖励。"
print_param SIMILARITY "target full-catalog log-softmax" "目标物品在 seen-mask 后全候选上的 log probability。"
print_param COT_SIM_NDCG_TEMPERATURE "$COT_SIM_NDCG_TEMPERATURE" "相似度 log-softmax 温度。"
print_param COT_SIM_NDCG_SIM_WEIGHT "$COT_SIM_NDCG_SIM_WEIGHT" "组内 z-score 后的相似度 reward 权重。"
print_param COT_SIM_NDCG_K "$COT_SIM_NDCG_K" "NDCG 截断位置；目标 rank 超过 100 时该项为 0。"
print_param COT_SIM_NDCG_NDCG_WEIGHT "$COT_SIM_NDCG_NDCG_WEIGHT" "组内 z-score 后的 NDCG@100 reward 权重。"
print_param GROUP_STANDARDIZATION per_example_id "相似度与 NDCG 分别在同一 prompt 的 4 条 completion 内标准化。"
print_param SEEN_ITEM_MASK enabled "屏蔽历史物品并保留监督 target。"
print_param REWARD_QUERY "history + full generated completion" "reward embedding 输入保留完整生成，超长时按既定字段级规则截断历史。"
print_param NUM_GENERATIONS "$NUM_GENERATIONS" "每个 prompt 采样 4 条候选并全部进入标准 GRPO loss。"
print_param GENERATION_BATCH_SIZE "$GENERATION_BATCH_SIZE" "vLLM 每轮生成 16 条 completion，对应 4 个 prompt 组。"
print_param BATCH_SIZE "$BATCH_SIZE" "每次反向传播 4 条 completion，对应 1 个完整候选组。"
print_param STEPS_PER_GENERATION "$steps_per_generation" "每轮 rollout 拆成 4 个反向传播 step。"
print_param GRAD_ACCUM "$GRAD_ACCUM" "梯度累积固定为 1。"
print_param MAX_LENGTH "$MAX_LENGTH" "policy prompt 上限；超长输入从左侧截断。"
print_param MAX_COMPLETION_LENGTH "$MAX_COMPLETION_LENGTH" "每条 CoT 最多生成 512 tokens。"
print_param GENERATION_TEMPERATURE "$GENERATION_TEMPERATURE" "rollout 采样温度。"
print_param GENERATION_TOP_K "$GENERATION_TOP_K" "每步保留概率最高的 200 个 token。"
print_param GENERATION_TOP_P "$GENERATION_TOP_P" "不额外裁剪 nucleus 概率质量。"
print_param EPOCHS "$EPOCHS" "后 80% 数据训练 1 个 GRPO epoch。"
print_param APPROX_STEPS "$approx_steps" "每 step 消耗 1 个 prompt 组时的单 epoch optimizer steps。"
print_param TRAIN_TYPE lora "仅更新新建的 GRPO LoRA，完整 SFT 基座保持冻结。"
print_param LEARNING_RATE "$LEARNING_RATE" "GRPO LoRA AdamW 峰值学习率。"
print_param BETA "$BETA" "KL 系数；reference 为不启用 GRPO adapter 的完整 SFT 模型。"
print_param LORA_RANK "$LORA_RANK" "新建 GRPO LoRA 的 rank。"
print_param LORA_ALPHA "$LORA_ALPHA" "新建 GRPO LoRA 的缩放系数。"
print_param SAVE_STEPS "$SAVE_STEPS" "每 300 个 optimizer step 保存 checkpoint。"
print_param USE_VLLM "$USE_VLLM" "使用 colocate vLLM 生成 rollout。"
print_param VLLM_ENABLE_LORA true "vLLM 只同步新建的 GRPO LoRA；SFT 已固化在完整基座。"
print_param VLLM_GPU_MEMORY_UTILIZATION "$VLLM_GPU_MEMORY_UTILIZATION" "vLLM 预留 10% GPU 显存。"
print_param VLLM_MAX_MODEL_LEN "$VLLM_MAX_MODEL_LEN" "vLLM 总上下文为 4096-token prompt 加 512-token completion。"
print_param VLLM_MAX_NUM_SEQS "$VLLM_MAX_NUM_SEQS" "vLLM 最多同时调度 16 条序列。"
print_param VLLM_SLEEP_LEVEL "$VLLM_SLEEP_LEVEL" "rollout 后释放 vLLM 权重和 KV cache，为反向传播腾出显存。"
print_param COMPONENT_LOG "$COT_SIM_NDCG_COMPONENT_LOG" "逐步记录 raw similarity、rank、NDCG、组内 z-score 和最终 reward。"
print_param OUT "$OUT" "GRPO LoRA checkpoint、trainer 日志和 reward 组件日志目录。"
print_param SEED "$SEED" "数据顺序、rollout 与训练随机种子固定为 42。"
print_param CONFIRM_RUN "$CONFIRM_RUN" "必须为 1 才启动正式训练。"

if [[ "$CONFIRM_RUN" != "1" ]]; then
  echo "未启动训练。"
  exit 0
fi

source /opt/miniforge3/bin/activate "$VENV"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES
export NPROC_PER_NODE=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

mkdir -p "$OUT"
nvidia-smi -q -d ECC > "$OUT/gpu_ecc_before_train.txt"
nvidia-smi -q -d ROW_REMAPPER > "$OUT/gpu_row_remapper_before_train.txt"

export COT_SIM_NDCG_EMBEDDING_MODEL="$REWARD_EMBEDDING"
export COT_SIM_NDCG_ITEM_INFO="$ITEM_INFO"
export COT_SIM_NDCG_MAX_LENGTH="$MAX_LENGTH"
export COT_SIM_NDCG_ITEM_BATCH_SIZE
export COT_SIM_NDCG_QUERY_BATCH_SIZE
export COT_SIM_NDCG_TEMPERATURE
export COT_SIM_NDCG_K
export COT_SIM_NDCG_SIM_WEIGHT
export COT_SIM_NDCG_NDCG_WEIGHT
export COT_SIM_NDCG_GROUP_SIZE="$NUM_GENERATIONS"
export COT_SIM_NDCG_STRICT_GROUP_SIZE=1
export COT_SIM_NDCG_EXPECTED_ITEMS=12000
export COT_SIM_NDCG_COMPONENT_LOG
export COT_SIM_NDCG_LOG_EVERY
export COT_SIM_NDCG_TORCH_DTYPE=bfloat16
export COT_SIM_NDCG_ATTN_IMPLEMENTATION=flash_attention_2
export COT_SIM_NDCG_DEVICE=cuda:0

"$VENV/bin/swift" rlhf \
  --rlhf_type grpo \
  --model "$MODEL" \
  --model_type qwen2_5 \
  --template qwen2_5 \
  --dataset "$DATASET" \
  --external_plugins "$REWARD_PLUGIN" \
  --reward_funcs cot_sim_ndcg \
  --reward_weights 1.0 \
  --num_generations "$NUM_GENERATIONS" \
  --generation_batch_size "$GENERATION_BATCH_SIZE" \
  --per_device_train_batch_size "$BATCH_SIZE" \
  --gradient_accumulation_steps "$GRAD_ACCUM" \
  --num_train_epochs "$EPOCHS" \
  --max_length "$MAX_LENGTH" \
  --truncation_strategy left \
  --max_completion_length "$MAX_COMPLETION_LENGTH" \
  --temperature "$GENERATION_TEMPERATURE" \
  --top_k "$GENERATION_TOP_K" \
  --top_p "$GENERATION_TOP_P" \
  --beta "$BETA" \
  --loss_type grpo \
  --scale_rewards group \
  --num_iterations 1 \
  --train_type lora \
  --lora_rank "$LORA_RANK" \
  --lora_alpha "$LORA_ALPHA" \
  --learning_rate "$LEARNING_RATE" \
  --weight_decay "$WEIGHT_DECAY" \
  --lr_scheduler_type cosine \
  --warmup_ratio 0 \
  --warmup_steps "$WARMUP_STEPS" \
  --torch_dtype bfloat16 \
  --attn_impl flash_attn \
  --gradient_checkpointing true \
  --use_vllm "$USE_VLLM" \
  --vllm_mode colocate \
  --vllm_tensor_parallel_size 1 \
  --vllm_pipeline_parallel_size 1 \
  --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
  --vllm_max_model_len "$VLLM_MAX_MODEL_LEN" \
  --vllm_max_num_seqs "$VLLM_MAX_NUM_SEQS" \
  --sleep_level "$VLLM_SLEEP_LEVEL" \
  --vllm_enable_lora true \
  --vllm_max_lora_rank "$LORA_RANK" \
  --seed "$SEED" \
  --data_seed "$SEED" \
  --save_strategy steps \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit "$SAVE_TOTAL_LIMIT" \
  --save_only_model true \
  --logging_steps 1 \
  --log_completions true \
  --dataloader_num_workers 0 \
  --report_to none \
  --output_dir "$OUT"
