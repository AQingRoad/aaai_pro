#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/user/aaai_pro}
VENV=${VENV:-/home/user/.conda/envs/aaai_pro}
MODEL=${MODEL:-/home/user/models_hf/Qwen2.5-3B-Instruct}
SFT_ADAPTER=${SFT_ADAPTER:-$ROOT/manu_src/model_outputs/CDs_and_Vinyl/sft/qwen25_3b_lora_non_target_glm52_sft20_disjoint_time_title_rating_store_categories_desc256_details256_v1_bs32_ga1_lr2e5_ep1_len4096_seed42/v0-20260717-091634/checkpoint-67}

SPLIT_DIR=${SPLIT_DIR:-$ROOT/manu_src/datas/CDs_and_Vinyl/sft_grpo/disjoint_example20_80__input_time_title_rating_store_categories_desc256_details256_v1}
GRPO_SOURCE=${GRPO_SOURCE:-$SPLIT_DIR/grpo_train80_seed42_n8578.jsonl}
SFT_EXAMPLE_IDS=${SFT_EXAMPLE_IDS:-$SPLIT_DIR/sft_example_ids.txt}
DATASET=${DATASET:-$SPLIT_DIR/grpo_cot_sim_ndcg_messages_train80_seed42_n8578.jsonl}
ITEM_INFO=${ITEM_INFO:-$ROOT/manu_src/datas/CDs_and_Vinyl/arrow_to_jsonls/item_info.jsonl}
REWARD_PLUGIN=${REWARD_PLUGIN:-$ROOT/manu_src/scripts/train/cot_sim_ndcg_reward.py}
REWARD_EMBEDDING=${REWARD_EMBEDDING:-$ROOT/manu_src/model_outputs/CDs_and_Vinyl/embedding/qwen3emb06b_history_plus_glm52_non_target_cot_input_time_title_rating_store_categories_desc256_details256_v1_bs128_ga1_lr2e5_ep5_len4096_seed42/checkpoint-epoch-01}
OUT=${OUT:-$ROOT/manu_src/model_outputs/CDs_and_Vinyl/grpo/qwen25_3b_lora_sft20_grpo80_cottrained_logsoftmaxsim_w0p8_ndcg100_w0p2_g4_genbs8_bs8_ga1_vllm0p10_lr2e5_ep1_vllmlen4608_clen512_seed42}

EXPECTED_ROWS=${EXPECTED_ROWS:-8578}
SEED=${SEED:-42}
NUM_GENERATIONS=${NUM_GENERATIONS:-4}
GENERATION_BATCH_SIZE=${GENERATION_BATCH_SIZE:-8}
BATCH_SIZE=${BATCH_SIZE:-8}
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
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-10}

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
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-8}
VLLM_SLEEP_LEVEL=${VLLM_SLEEP_LEVEL:-0}
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
  echo "GENERATION_BATCH_SIZE 必须能被单卡 BATCH_SIZE 整除，以便 TRL 自动计算 steps_per_generation。" >&2
  exit 1
fi
if [[ "$VLLM_SLEEP_LEVEL" != "0" && "$VLLM_SLEEP_LEVEL" != "1" && "$VLLM_SLEEP_LEVEL" != "2" ]]; then
  echo "VLLM_SLEEP_LEVEL 只能为 0、1 或 2。" >&2
  exit 1
fi
if [[ "$MAX_LENGTH" != "4096" ]]; then
  echo "当前实验必须与 SFT 和 embedding 保持 max_length=4096。" >&2
  exit 1
fi

for dependency in \
  "$ROOT/AGENTS.md" \
  "$VENV/bin/python" \
  "$VENV/bin/swift" \
  "$MODEL" \
  "$SFT_ADAPTER" \
  "$GRPO_SOURCE" \
  "$SFT_EXAMPLE_IDS" \
  "$ITEM_INFO" \
  "$REWARD_PLUGIN" \
  "$REWARD_EMBEDDING"; do
  require_path "实验依赖" "$dependency"
done

source_rows=$(wc -l < "$GRPO_SOURCE" | tr -d ' ')
if [[ "$source_rows" != "$EXPECTED_ROWS" ]]; then
  echo "GRPO reserved 数据应为 $EXPECTED_ROWS 条，当前为 $source_rows。" >&2
  exit 1
fi

prompts_per_step=$((BATCH_SIZE / NUM_GENERATIONS * GRAD_ACCUM))
approx_steps=$(((EXPECTED_ROWS + prompts_per_step - 1) / prompts_per_step))
steps_per_generation=$((GENERATION_BATCH_SIZE / BATCH_SIZE))

echo "Qwen2.5-3B LoRA GRPO 试验参数："
print_param ROOT "$ROOT" "A100 项目根目录；执行前再次读取 AGENTS.md。"
print_param MODEL "$MODEL" "Qwen2.5-3B-Instruct 基座模型。"
print_param SFT_ADAPTER "$SFT_ADAPTER" "已完成的随机 20% 数据、1 epoch SFT LoRA checkpoint。"
print_param GRPO_SOURCE "$GRPO_SOURCE" "与 SFT example_id 互斥的 80% 原始训练样本，共 $source_rows 条。"
print_param DATASET "$DATASET" "只把 history 写入 messages；target_item_id 仅作为 reward metadata。"
print_param ITEM_INFO "$ITEM_INFO" "构造与 embedding 训练和全量评测一致的 12000 个候选文本。"
print_param REWARD_EMBEDDING "$REWARD_EMBEDDING" "冻结的无 Mask CoT-trained embedding checkpoint；当前 pilot 固定 epoch-01。"
print_param REWARD_FUNC cot_sim_ndcg "唯一 reward；不包含格式、rubric、reference CoT、margin 或 History Gain。"
print_param SIMILARITY "target full-catalog log-softmax" "相似度项使用目标物品在全部未屏蔽候选上的 log probability。"
print_param COT_SIM_NDCG_TEMPERATURE "$COT_SIM_NDCG_TEMPERATURE" "log-softmax 温度；与 CoT-trained embedding 的 InfoNCE temperature 一致。"
print_param COT_SIM_NDCG_SIM_WEIGHT "$COT_SIM_NDCG_SIM_WEIGHT" "组内标准化后的相似度 reward 权重。"
print_param COT_SIM_NDCG_K "$COT_SIM_NDCG_K" "硬 NDCG 的截断位置；目标 rank 超过 100 时该项为 0。"
print_param COT_SIM_NDCG_NDCG_WEIGHT "$COT_SIM_NDCG_NDCG_WEIGHT" "组内标准化后的 NDCG@100 reward 权重。"
print_param GROUP_STANDARDIZATION per_example_id "两项分别在同一 prompt 的 4 条 completion 内做 z-score，再按 0.8/0.2 相加。"
print_param SEEN_ITEM_MASK enabled "屏蔽历史物品，同时保留监督 target；候选 item 0 不进入候选表。"
print_param REWARD_QUERY "history + full generated completion" "embedding reward 保留完整生成，超长时只删最旧历史 Description、Details 和物品。"
print_param NUM_GENERATIONS "$NUM_GENERATIONS" "每个历史采样 4 条 CoT，构成 GRPO 比较组。"
print_param GENERATION_TEMPERATURE "$GENERATION_TEMPERATURE" "rollout 采样温度；初始试验与 API CoT 生成温度保持 1.0。"
print_param GENERATION_TOP_K "$GENERATION_TOP_K" "rollout 每步保留概率最高的 200 个 token。"
print_param GENERATION_TOP_P "$GENERATION_TOP_P" "不额外裁剪 nucleus 概率质量。"
print_param BATCH_SIZE "$BATCH_SIZE" "每个 optimizer step 反向传播的 completion 数；对应 $((BATCH_SIZE / NUM_GENERATIONS)) 个完整四候选组。"
print_param GENERATION_BATCH_SIZE "$GENERATION_BATCH_SIZE" "vLLM 每轮生成的 completion 总数；对应 $((GENERATION_BATCH_SIZE / NUM_GENERATIONS)) 个完整四候选组。"
print_param STEPS_PER_GENERATION "$steps_per_generation" "由 TRL 自动计算；每轮 rollout 拆成 $steps_per_generation 个 batch，每个 batch 含 $BATCH_SIZE 条 completion。"
print_param GRAD_ACCUM "$GRAD_ACCUM" "梯度累积固定为 1。"
print_param MAX_LENGTH "$MAX_LENGTH" "policy prompt 上限，与 SFT 和 reward embedding 保持 4096。"
print_param MAX_COMPLETION_LENGTH "$MAX_COMPLETION_LENGTH" "每条生成的最大 token 数；提示词仍要求 analysis+answer 不超过 512 words。"
print_param EPOCHS "$EPOCHS" "先训练 1 个完整 GRPO epoch。"
print_param APPROX_STEPS "$approx_steps" "按每 step $((BATCH_SIZE / NUM_GENERATIONS)) 个独立 prompt 估算的单 epoch optimizer steps。"
print_param LEARNING_RATE "$LEARNING_RATE" "GRPO LoRA AdamW 峰值学习率。"
print_param WARMUP_STEPS "$WARMUP_STEPS" "前 64 个 optimizer step 线性 warmup。"
print_param BETA "$BETA" "GRPO KL 系数；约束策略偏离 SFT policy 的幅度。"
print_param LORA_RANK "$LORA_RANK" "继续训练 rank 64 的 SFT LoRA。"
print_param LORA_ALPHA "$LORA_ALPHA" "LoRA 缩放系数 128。"
print_param SAVE_STEPS "$SAVE_STEPS" "每 300 个 optimizer step 保存一次 checkpoint。"
print_param USE_VLLM "$USE_VLLM" "使用 colocate vLLM 生成 rollout。"
print_param VLLM_GPU_MEMORY_UTILIZATION "$VLLM_GPU_MEMORY_UTILIZATION" "vLLM 总预留占 GPU 显存的比例；当前固定 10%。"
print_param VLLM_MAX_MODEL_LEN "$VLLM_MAX_MODEL_LEN" "vLLM 总上下文为 4096-token prompt 加 512-token completion；prompt 超过 4096 时沿用左侧截断。"
print_param VLLM_MAX_NUM_SEQS "$VLLM_MAX_NUM_SEQS" "vLLM 同时调度的最大序列数，与本轮 generation batch 对齐。"
print_param VLLM_SLEEP_LEVEL "$VLLM_SLEEP_LEVEL" "Swift 的 vLLM 休眠级别；1 表示 rollout 后释放 vLLM 权重和 KV cache，为策略模型反向传播腾出显存。"
print_param COT_SIM_NDCG_ITEM_BATCH_SIZE "$COT_SIM_NDCG_ITEM_BATCH_SIZE" "首次构建冻结候选 embedding 表的批量。"
print_param COT_SIM_NDCG_QUERY_BATCH_SIZE "$COT_SIM_NDCG_QUERY_BATCH_SIZE" "每次 reward 编码 completion query 的批量。"
print_param COMPONENT_LOG "$COT_SIM_NDCG_COMPONENT_LOG" "记录 cosine、log-softmax、rank、NDCG、组内 z-score、冲突率和最终 reward。"
print_param OUTPUT "$OUT" "GRPO checkpoint、trainer 日志和 reward 组件日志目录。"
print_param SEED "$SEED" "数据顺序、rollout 与训练随机种子固定为 42。"
print_param CONFIRM_RUN "$CONFIRM_RUN" "必须为 1 才构建 GRPO messages 并启动训练。"

if [[ "$CONFIRM_RUN" != "1" ]]; then
  echo "未执行数据构建或训练。"
  exit 0
fi

source /opt/miniforge3/bin/activate "$VENV"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES
export NPROC_PER_NODE=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

if [[ ! -s "$DATASET" ]]; then
  "$VENV/bin/python" manu_src/scripts/pre_datas/build_grpo_cot_sim_ndcg_dataset.py \
    --input "$GRPO_SOURCE" \
    --output "$DATASET" \
    --sft-example-ids "$SFT_EXAMPLE_IDS" \
    --item-type "CD or vinyl release" \
    --language en \
    --expected-rows "$EXPECTED_ROWS" \
    --seed "$SEED"
fi
dataset_rows=$(wc -l < "$DATASET" | tr -d ' ')
if [[ "$dataset_rows" != "$EXPECTED_ROWS" ]]; then
  echo "GRPO messages 应为 $EXPECTED_ROWS 条，当前为 $dataset_rows。" >&2
  exit 1
fi

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
  --adapters "$SFT_ADAPTER" \
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
