#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/user/aaai_pro}
VENV=${VENV:-/home/user/.conda/envs/aaai_pro}
MODEL=${MODEL:-$ROOT/manu_src/model_outputs/CDs_and_Vinyl/sft/qwen25_3b_full_sft20_disjoint_time_title_rating_store_categories_desc256_details256_v1_bs16_ga1_lr2e5_ep1_len4096_seed42/v0-20260718-080839/checkpoint-134}

SPLIT_DIR=${SPLIT_DIR:-$ROOT/manu_src/datas/CDs_and_Vinyl/sft_grpo/disjoint_example20_80__input_time_title_rating_store_categories_desc256_details256_v1}
MESSAGES_INPUT=${MESSAGES_INPUT:-$SPLIT_DIR/grpo_cot_sim_ndcg_messages_train80_seed42_n8578.jsonl}
RAW_INPUT=${RAW_INPUT:-$SPLIT_DIR/grpo_train80_seed42_n8578.jsonl}
REFERENCE_COT_INPUT=${REFERENCE_COT_INPUT:-$ROOT/manu_src/datas/CDs_and_Vinyl/cot/api/history_only_next_item_feature_cot__input_time_title_rating_store_categories_desc256_details256_v1/cot_non_target_glm52_train_full_seed42_temp1.jsonl}
ENRICHED_DATASET=${ENRICHED_DATASET:-$SPLIT_DIR/grpo_rubric_ndcg1000_gain_reference_cot_messages_train80_seed42_n8578.jsonl}
CACHED_DATASET=${CACHED_DATASET:-$SPLIT_DIR/grpo_rubric_ndcg1000_gain_cached_reference_messages_train80_seed42_n8578.jsonl}
PROBE_DATASET=${PROBE_DATASET:-$SPLIT_DIR/probe/grpo_rubric_ndcg1000_gain_reference_cot_messages_seed42_n32.jsonl}
ITEM_INFO=${ITEM_INFO:-$ROOT/manu_src/datas/CDs_and_Vinyl/arrow_to_jsonls/item_info.jsonl}
REWARD_PLUGIN=${REWARD_PLUGIN:-$ROOT/manu_src/scripts/train/cot_rubric_ndcg1000_gain_reward.py}
REWARD_EMBEDDING=${REWARD_EMBEDDING:-$ROOT/manu_src/model_outputs/CDs_and_Vinyl/embedding/qwen3emb06b_history_plus_glm52_non_target_cot_input_time_title_rating_store_categories_desc256_details256_v1_bs128_ga1_lr2e5_ep5_len4096_seed42/checkpoint-epoch-01}
DATASET_BUILDER=${DATASET_BUILDER:-$ROOT/manu_src/scripts/pre_datas/build_grpo_rubric_ndcg1000_gain_dataset.py}
REFERENCE_PRECOMPUTE=${REFERENCE_PRECOMPUTE:-$ROOT/manu_src/scripts/pre_datas/precompute_grpo_reference_ndcg.py}
API_CONFIG=${API_CONFIG:-$ROOT/manu_src/api_info/API_CONFIG.py}

RUN_NAME=${RUN_NAME:-qwen25_3b_fullsft20_grpolora80_dual_gpu_ddp_colocate_cottrained_simz0p6_rubric_ndcg1000gainz0p4_ks_glm5_2_g4_globalgenbs32_perdevbs8_ga1_vllmsleep1_vllm0p10_lr2e5_ep1_vllmlen4608_clen512_loradrop0_seed42}
OUT=${OUT:-$ROOT/manu_src/model_outputs/CDs_and_Vinyl/grpo/$RUN_NAME}
PROBE_OUT=${PROBE_OUT:-$ROOT/manu_src/model_outputs/CDs_and_Vinyl/grpo/probes/${RUN_NAME}_compat10step}

EXPECTED_ROWS=${EXPECTED_ROWS:-8578}
PROBE_ROWS=${PROBE_ROWS:-32}
SEED=${SEED:-42}
NUM_GENERATIONS=${NUM_GENERATIONS:-4}
GENERATION_BATCH_SIZE=${GENERATION_BATCH_SIZE:-32}
BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACCUM=${GRAD_ACCUM:-1}
PROBE_GENERATION_BATCH_SIZE=${PROBE_GENERATION_BATCH_SIZE:-32}
PROBE_BATCH_SIZE=${PROBE_BATCH_SIZE:-8}
PROBE_MAX_STEPS=${PROBE_MAX_STEPS:-10}
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
LORA_DROPOUT=${LORA_DROPOUT:-0}
SAVE_STEPS=${SAVE_STEPS:-300}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-30}

NDCG_K=${NDCG_K:-1000}
SIMILARITY_TEMPERATURE=${SIMILARITY_TEMPERATURE:-0.05}
SIMILARITY_WEIGHT=${SIMILARITY_WEIGHT:-0.6}
JOINT_REWARD_WEIGHT=${JOINT_REWARD_WEIGHT:-0.4}
RUBRIC_POWER=${RUBRIC_POWER:-1.0}
NEGATIVE_GAIN_WEIGHT=${NEGATIVE_GAIN_WEIGHT:-1.0}
ZSCORE_EPSILON=${ZSCORE_EPSILON:-1e-6}
REWARD_ITEM_BATCH_SIZE=${REWARD_ITEM_BATCH_SIZE:-128}
REWARD_QUERY_BATCH_SIZE=${REWARD_QUERY_BATCH_SIZE:-16}
REFERENCE_BATCH_SIZE=${REFERENCE_BATCH_SIZE:-16}

RUBRIC_API_PROVIDER=${RUBRIC_API_PROVIDER:-ks_tokenverse}
RUBRIC_API_MODEL=${RUBRIC_API_MODEL:-glm-5-2}
RUBRIC_API_TIMEOUT=${RUBRIC_API_TIMEOUT:-180}
RUBRIC_API_MAX_RETRIES=${RUBRIC_API_MAX_RETRIES:-0}
RUBRIC_API_CONCURRENCY_PER_KEY=${RUBRIC_API_CONCURRENCY_PER_KEY:-10}
RUBRIC_API_MAX_TOKENS=${RUBRIC_API_MAX_TOKENS:-128}
RUBRIC_API_THINKING=${RUBRIC_API_THINKING:-disabled}
RUBRIC_API_FALLBACK=${RUBRIC_API_FALLBACK:-error}
RUBRIC_HTTPS_PROXY=${RUBRIC_HTTPS_PROXY:-http://127.0.0.1:18766}

USE_VLLM=${USE_VLLM:-true}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.10}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-4608}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-16}
VLLM_SLEEP_LEVEL=${VLLM_SLEEP_LEVEL:-1}
PROBE_CUDA_VISIBLE_DEVICES=${PROBE_CUDA_VISIBLE_DEVICES:-0}
FORMAL_CUDA_VISIBLE_DEVICES=${FORMAL_CUDA_VISIBLE_DEVICES:-0,1}
FORMAL_WORLD_SIZE=${FORMAL_WORLD_SIZE:-2}
WORKFLOW=${WORKFLOW:-probe_then_full}
CONFIRM_RUN=${CONFIRM_RUN:-0}

print_param() {
  local name=$1 value=$2 description=$3
  printf '  %-44s %s\n      %s\n' "$name=$value" "" "$description"
}

require_path() {
  local label=$1 path=$2
  if [[ ! -e "$path" ]]; then
    echo "缺少${label}: $path" >&2
    exit 1
  fi
}

for dependency in \
  "$ROOT/AGENTS.md" \
  "$VENV/bin/python" \
  "$VENV/bin/swift" \
  "$MODEL" \
  "$MESSAGES_INPUT" \
  "$RAW_INPUT" \
  "$REFERENCE_COT_INPUT" \
  "$ITEM_INFO" \
  "$REWARD_PLUGIN" \
  "$REWARD_EMBEDDING" \
  "$DATASET_BUILDER" \
  "$REFERENCE_PRECOMPUTE" \
  "$API_CONFIG"; do
  require_path "新奖励 GRPO 依赖" "$dependency"
done
if [[ -e "$MODEL/adapter_config.json" || ! -e "$MODEL/config.json" ]]; then
  echo "MODEL 必须指向前 20% 全参数 SFT 的完整 checkpoint。" >&2
  exit 1
fi
if [[ "$SEED" != "42" ]]; then
  echo "项目随机种子必须为 42。" >&2
  exit 1
fi
if [[ "$GRAD_ACCUM" != "1" || "$NUM_GENERATIONS" != "4" ]]; then
  echo "当前实验固定 gradient_accumulation_steps=1、num_generations=4。" >&2
  exit 1
fi
if [[ "$PROBE_CUDA_VISIBLE_DEVICES" != "0" || "$FORMAL_CUDA_VISIBLE_DEVICES" != "0,1" || "$FORMAL_WORLD_SIZE" != "2" ]]; then
  echo "当前流程固定 GPU0 单卡 10-step 检查，随后 GPU0+GPU1 双卡 DDP 正式训练。" >&2
  exit 1
fi
for value in "$BATCH_SIZE" "$GENERATION_BATCH_SIZE" "$PROBE_BATCH_SIZE" "$PROBE_GENERATION_BATCH_SIZE"; do
  if ((value % NUM_GENERATIONS != 0)); then
    echo "训练与适配测试的 batch 都必须能被 num_generations 整除。" >&2
    exit 1
  fi
done
if ((GENERATION_BATCH_SIZE % BATCH_SIZE != 0 || PROBE_GENERATION_BATCH_SIZE % PROBE_BATCH_SIZE != 0)); then
  echo "generation_batch_size 必须能被对应的反向 batch 整除。" >&2
  exit 1
fi
if [[ "$MAX_LENGTH" != "4096" || "$MAX_COMPLETION_LENGTH" != "512" || "$VLLM_MAX_MODEL_LEN" != "4608" ]]; then
  echo "本实验固定 prompt=4096、completion=512、vLLM context=4608。" >&2
  exit 1
fi
if [[ "$WORKFLOW" != "probe" && "$WORKFLOW" != "probe_then_full" && "$WORKFLOW" != "full" ]]; then
  echo "WORKFLOW 只能为 probe、probe_then_full 或 full。" >&2
  exit 1
fi

messages_rows=$(wc -l < "$MESSAGES_INPUT" | tr -d ' ')
raw_rows=$(wc -l < "$RAW_INPUT" | tr -d ' ')
reference_rows=$(wc -l < "$REFERENCE_COT_INPUT" | tr -d ' ')
if [[ "$messages_rows" != "$EXPECTED_ROWS" || "$raw_rows" != "$EXPECTED_ROWS" ]]; then
  echo "GRPO messages/raw 应各为 $EXPECTED_ROWS 条，当前为 $messages_rows/$raw_rows。" >&2
  exit 1
fi
if ((reference_rows < EXPECTED_ROWS)); then
  echo "完整训练 reference CoT 少于 $EXPECTED_ROWS 条，当前为 $reference_rows。" >&2
  exit 1
fi

api_key_count=$(
  cd "$ROOT"
  PYTHONPATH="$ROOT" RUBRIC_API_PROVIDER="$RUBRIC_API_PROVIDER" "$VENV/bin/python" - <<'PY'
import os
from manu_src.api_info import API_CONFIG
provider = os.environ["RUBRIC_API_PROVIDER"]
name = "ks_tokenverse" if provider in {"ks", "ks_tokenverse", "tokenverse"} else "glm_official"
config = getattr(API_CONFIG, "API_PROVIDER_CONFIGS", {}).get(name, {})
print(len(config.get("api_key_list", []) or []))
PY
)
if ((api_key_count <= 0)); then
  echo "API_CONFIG.py 中没有 $RUBRIC_API_PROVIDER API key。" >&2
  exit 1
fi

global_train_batch=$((BATCH_SIZE * FORMAL_WORLD_SIZE))
if ((GENERATION_BATCH_SIZE % global_train_batch != 0)); then
  echo "generation_batch_size 必须能被双卡 global train batch 整除。" >&2
  exit 1
fi
prompts_per_step=$((global_train_batch / NUM_GENERATIONS * GRAD_ACCUM))
approx_steps=$((EXPECTED_ROWS / prompts_per_step))
steps_per_generation=$((GENERATION_BATCH_SIZE / global_train_batch))

echo "Qwen2.5-3B 新 Rubric × NDCG@1000 Gain LoRA GRPO 参数："
print_param WORKFLOW "$WORKFLOW" "GPU0 先跑 10-step 功能适配；probe_then_full 在适配成功和 reference 预计算完成后自动进入双卡正式训练。"
print_param PROBE_GPU "$PROBE_CUDA_VISIBLE_DEVICES" "10-step 阶段只使用物理 GPU0，检查生成、API Rubric、reference gain 和反向传播。"
print_param FORMAL_GPUS "$FORMAL_CUDA_VISIBLE_DEVICES" "正式阶段启动两个 DDP 进程；GPU0 与 GPU1 都执行 rollout、reward 与反向传播。"
print_param MODEL "$MODEL" "前 20% 数据训练 1 epoch 的全参数 SFT checkpoint；GRPO 新建 LoRA。"
print_param INPUT_SCHEMA time_title_rating_store_categories_desc256_details256_v1 "history 与 SFT、API CoT、reward embedding 使用相同字段口径。"
print_param POLICY_INPUT "$MESSAGES_INPUT" "policy messages 只包含 user history，不包含 target 或固定 reference CoT。"
print_param TARGET_SOURCE "$RAW_INPUT:positive" "完整 positive 复制为 reward-only target_item_text，不做字符截断。"
print_param REFERENCE_COT_SOURCE "$REFERENCE_COT_INPUT:cot" "固定 API CoT 只计算 reference rank/NDCG，不加入 policy prompt。"
print_param REFERENCE_CACHE "$CACHED_DATASET" "训练前使用同一 embedding、候选表和 seen-item mask 离线缓存 reference_rank/reference_ndcg。"
print_param ITEM_INFO "$ITEM_INFO" "全候选表固定为 12000 个 item，屏蔽历史物品并保留 target。"
print_param REWARD_EMBEDDING "$REWARD_EMBEDDING" "冻结 CoT-trained embedding epoch-01；new/reference 使用同一 query formatter 与 max_length。"
print_param REWARD_FORMULA "0.6*z(similarity)+0.4*z(joint_gain)" "joint_gain=q*max(new-reference,0)-1.0*max(reference-new,0)，两项先在四候选组内标准化。"
print_param NDCG_K "$NDCG_K" "new/reference 都按目标在 12000-item 候选表中的 NDCG@1000 计算。"
print_param SIMILARITY_TEMPERATURE "$SIMILARITY_TEMPERATURE" "相似度项使用全候选 log-softmax，温度固定 0.05。"
print_param SIMILARITY_WEIGHT "$SIMILARITY_WEIGHT" "组内 z-score 后相似度分量权重。"
print_param JOINT_REWARD_WEIGHT "$JOINT_REWARD_WEIGHT" "组内 z-score 后 Rubric 加权排名增益分量权重。"
print_param RUBRIC_POWER "$RUBRIC_POWER" "q 不做额外 z-score，使用 q^1。"
print_param NEGATIVE_GAIN_WEIGHT "$NEGATIVE_GAIN_WEIGHT" "NDCG 下降不乘 q，按 1.0 倍负增益处罚。"
print_param TRAINER_SCALE_REWARDS group "组合 reward 交给标准 GRPO，再计算同组 advantage；四条候选全部进入 loss。"
print_param RUBRIC_PROMPT_LANGUAGE English "API 运行时使用 manu_src/prompts 中的英文 Target-Relevance Rubric prompt。"
print_param RUBRIC_API "$RUBRIC_API_PROVIDER/$RUBRIC_API_MODEL" "Rubric judge 看到 history、生成 CoT 和真实 target，只返回五维 1-5 JSON。"
print_param RUBRIC_HTTPS_PROXY "$RUBRIC_HTTPS_PROXY" "服务器仅连接本机反向隧道；域名解析、TLS 上游连接与 Tokenverse 请求均从本地发出。"
print_param RUBRIC_API_KEYS "$api_key_count key(s)" "轮询配置文件中的 key；日志不记录 key 内容。"
print_param RUBRIC_API_CONCURRENCY "$RUBRIC_API_CONCURRENCY_PER_KEY per key total" "单卡检查使用 10；双卡正式阶段每进程使用 5，因此每个 key 的服务器总并发仍为 10。"
print_param RUBRIC_API_KEY_ATTEMPTS "$api_key_count" "单次评分失败后轮询其它 key；全部失败才终止该训练 step。"
print_param RUBRIC_API_TIMEOUT "$RUBRIC_API_TIMEOUT seconds" "单个 key 的请求超时；每个 key 内不等待重试。"
print_param RUBRIC_API_FALLBACK "$RUBRIC_API_FALLBACK" "关闭规则分数回退，避免同一训练混合两种 q 标准。"
print_param PROBE "$PROBE_ROWS prompts, gen_batch=$PROBE_GENERATION_BATCH_SIZE, train_batch=$PROBE_BATCH_SIZE, max_steps=$PROBE_MAX_STEPS" "随机种子 42 抽取 32 个 prompt，GPU0 连续执行 10 个 optimizer steps；batch 与单卡已跑通正式设置一致。"
print_param NUM_GENERATIONS "$NUM_GENERATIONS" "每个 history 采样 4 条候选，构成一个标准 GRPO group。"
print_param GENERATION_BATCH_SIZE "$GENERATION_BATCH_SIZE" "双卡全局每轮生成 32 条 completion，即 8 个四候选组；每张卡处理 16 条。"
print_param BATCH_SIZE "$BATCH_SIZE per device" "每张卡每次反向传播 8 条 completion；全局 batch 为 $global_train_batch 条、4 个完整候选组。"
print_param STEPS_PER_GENERATION "$steps_per_generation" "全局一轮 32 条 rollout 拆成 2 个 optimizer steps。"
print_param GRAD_ACCUM "$GRAD_ACCUM" "梯度累积固定为 1。"
print_param EPOCHS "$EPOCHS" "正式 GRPO 训练 1 个 epoch；双卡每 step 消耗 4 个 history，预计 $approx_steps 个 optimizer steps。"
print_param MAX_LENGTH "$MAX_LENGTH" "policy prompt 与 reward query 上限 4096 tokens；沿用既定左截断/字段级截断逻辑。"
print_param MAX_COMPLETION_LENGTH "$MAX_COMPLETION_LENGTH" "每条生成最多 512 tokens。"
print_param ROLLOUT_SAMPLING "temperature=$GENERATION_TEMPERATURE, top_k=$GENERATION_TOP_K, top_p=$GENERATION_TOP_P" "沿用上一轮正式 GRPO 的 rollout 采样设置。"
print_param OPTIMIZATION "lr=$LEARNING_RATE, beta=$BETA, warmup=$WARMUP_STEPS, weight_decay=$WEIGHT_DECAY" "LoRA GRPO 的学习率、KL 系数、warmup step 和权重衰减。"
print_param LORA "rank=$LORA_RANK, alpha=$LORA_ALPHA, dropout=$LORA_DROPOUT" "完整 SFT 基座冻结，只更新新建 GRPO LoRA。"
print_param VLLM "colocate per GPU, utilization=$VLLM_GPU_MEMORY_UTILIZATION, context=$VLLM_MAX_MODEL_LEN, max_seqs=$VLLM_MAX_NUM_SEQS, sleep=$VLLM_SLEEP_LEVEL" "每个 DDP rank 在本地 GPU 创建 vLLM；每卡最多调度 16 条序列，生成后释放权重和 KV cache。"
print_param SAVE "every $SAVE_STEPS steps, limit=$SAVE_TOTAL_LIMIT" "正式训练每 300 step 保存模型，最多保留 30 个 checkpoint。"
print_param OUTPUT "$OUT" "正式 checkpoint、trainer 日志与逐候选 reward 组件日志目录。"
print_param SEED "$SEED" "数据抽样、DataLoader、rollout 和训练随机种子均固定为 42。"
print_param CONFIRM_RUN "$CONFIRM_RUN" "只有设置为 1 才构建数据、请求 Rubric API、预计算 reference 并训练。"

if [[ "$CONFIRM_RUN" != "1" ]]; then
  echo "未执行数据构建、API 请求、reference 预计算或训练。"
  exit 0
fi

source /opt/miniforge3/bin/activate "$VENV"
cd "$ROOT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

"$VENV/bin/python" "$DATASET_BUILDER" \
  --messages-input "$MESSAGES_INPUT" \
  --raw-input "$RAW_INPUT" \
  --reference-cot-input "$REFERENCE_COT_INPUT" \
  --output "$ENRICHED_DATASET" \
  --probe-output "$PROBE_DATASET" \
  --expected-rows "$EXPECTED_ROWS" \
  --probe-rows "$PROBE_ROWS" \
  --seed "$SEED"

export COT_RUBRIC_NDCG_GAIN_EMBEDDING_MODEL="$REWARD_EMBEDDING"
export COT_RUBRIC_NDCG_GAIN_ITEM_INFO="$ITEM_INFO"
export COT_RUBRIC_NDCG_GAIN_MAX_LENGTH="$MAX_LENGTH"
export COT_RUBRIC_NDCG_GAIN_ITEM_BATCH_SIZE="$REWARD_ITEM_BATCH_SIZE"
export COT_RUBRIC_NDCG_GAIN_QUERY_BATCH_SIZE="$REWARD_QUERY_BATCH_SIZE"
export COT_RUBRIC_NDCG_GAIN_TEMPERATURE="$SIMILARITY_TEMPERATURE"
export COT_RUBRIC_NDCG_GAIN_K="$NDCG_K"
export COT_RUBRIC_NDCG_GAIN_SIM_WEIGHT="$SIMILARITY_WEIGHT"
export COT_RUBRIC_NDCG_GAIN_JOINT_WEIGHT="$JOINT_REWARD_WEIGHT"
export COT_RUBRIC_NDCG_GAIN_RUBRIC_POWER="$RUBRIC_POWER"
export COT_RUBRIC_NDCG_GAIN_NEGATIVE_GAIN_WEIGHT="$NEGATIVE_GAIN_WEIGHT"
export COT_RUBRIC_NDCG_GAIN_ZSCORE_EPSILON="$ZSCORE_EPSILON"
export COT_RUBRIC_NDCG_GAIN_GROUP_SIZE="$NUM_GENERATIONS"
export COT_RUBRIC_NDCG_GAIN_STRICT_GROUP_SIZE=1
export COT_RUBRIC_NDCG_GAIN_EXPECTED_ITEMS=12000
export COT_RUBRIC_NDCG_GAIN_TORCH_DTYPE=bfloat16
export COT_RUBRIC_NDCG_GAIN_ATTN_IMPLEMENTATION=flash_attention_2
export COT_RUBRIC_NDCG_GAIN_RUBRIC_SCORER=api
export COT_RUBRIC_NDCG_GAIN_STRICT_API_TARGET=1
export COT_RUBRIC_NDCG_GAIN_API_PROVIDER="$RUBRIC_API_PROVIDER"
export COT_RUBRIC_NDCG_GAIN_API_MODEL="$RUBRIC_API_MODEL"
export COT_RUBRIC_NDCG_GAIN_API_TIMEOUT="$RUBRIC_API_TIMEOUT"
export COT_RUBRIC_NDCG_GAIN_API_MAX_RETRIES="$RUBRIC_API_MAX_RETRIES"
export COT_RUBRIC_NDCG_GAIN_API_CONCURRENCY_PER_KEY="$RUBRIC_API_CONCURRENCY_PER_KEY"
export COT_RUBRIC_NDCG_GAIN_API_KEY_ATTEMPTS="$api_key_count"
export COT_RUBRIC_NDCG_GAIN_API_MAX_TOKENS="$RUBRIC_API_MAX_TOKENS"
export COT_RUBRIC_NDCG_GAIN_API_THINKING="$RUBRIC_API_THINKING"
export COT_RUBRIC_NDCG_GAIN_API_FALLBACK="$RUBRIC_API_FALLBACK"
export COT_RUBRIC_NDCG_GAIN_LOG_EVERY=1
export HTTPS_PROXY="$RUBRIC_HTTPS_PROXY"
export https_proxy="$RUBRIC_HTTPS_PROXY"

run_swift() {
  local dataset=$1 output=$2 generation_batch=$3 train_batch=$4 max_num_seqs=$5 mode=$6 visible_devices=$7 world_size=$8 api_per_key=$9
  local save_args
  if [[ "$mode" == "probe" ]]; then
    save_args=(--max_steps "$PROBE_MAX_STEPS" --save_strategy no)
  else
    save_args=(--num_train_epochs "$EPOCHS" --save_strategy steps --save_steps "$SAVE_STEPS" --save_total_limit "$SAVE_TOTAL_LIMIT" --save_only_model true)
  fi
  export COT_RUBRIC_NDCG_GAIN_API_CONCURRENCY_PER_KEY="$api_per_key"
  if [[ "$world_size" == "1" ]]; then
    export COT_RUBRIC_NDCG_GAIN_DEVICE=cuda:0
  else
    unset COT_RUBRIC_NDCG_GAIN_DEVICE
  fi
  CUDA_VISIBLE_DEVICES="$visible_devices" NPROC_PER_NODE="$world_size" "$VENV/bin/swift" rlhf \
    --rlhf_type grpo \
    --model "$MODEL" \
    --model_type qwen2_5 \
    --template qwen2_5 \
    --dataset "$dataset" \
    --external_plugins "$REWARD_PLUGIN" \
    --reward_funcs cot_rubric_ndcg1000_gain \
    --reward_weights 1.0 \
    --num_generations "$NUM_GENERATIONS" \
    --generation_batch_size "$generation_batch" \
    --per_device_train_batch_size "$train_batch" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
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
    --lora_dropout "$LORA_DROPOUT" \
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
    --vllm_max_num_seqs "$max_num_seqs" \
    --sleep_level "$VLLM_SLEEP_LEVEL" \
    --vllm_enable_lora true \
    --vllm_max_lora_rank "$LORA_RANK" \
    --seed "$SEED" \
    --data_seed "$SEED" \
    "${save_args[@]}" \
    --logging_steps 1 \
    --log_completions true \
    --dataloader_num_workers 0 \
    --report_to none \
    --output_dir "$output"
}

if [[ "$WORKFLOW" == "probe" || "$WORKFLOW" == "probe_then_full" ]]; then
  rm -rf "$PROBE_OUT"
  mkdir -p "$PROBE_OUT"
  export COT_RUBRIC_NDCG_GAIN_COMPONENT_LOG="$PROBE_OUT/reward_components_rank{rank}.jsonl"
  nvidia-smi -q -d ECC > "$PROBE_OUT/gpu_ecc_before_probe.txt"
  nvidia-smi -q -d ROW_REMAPPER > "$PROBE_OUT/gpu_row_remapper_before_probe.txt"
  run_swift "$PROBE_DATASET" "$PROBE_OUT" "$PROBE_GENERATION_BATCH_SIZE" "$PROBE_BATCH_SIZE" "$PROBE_GENERATION_BATCH_SIZE" probe "$PROBE_CUDA_VISIBLE_DEVICES" 1 "$RUBRIC_API_CONCURRENCY_PER_KEY"
  touch "$PROBE_OUT/COMPATIBILITY_PROBE_SUCCEEDED"
fi

if [[ "$WORKFLOW" == "full" || "$WORKFLOW" == "probe_then_full" ]]; then
  if [[ ! -s "$CACHED_DATASET" ]]; then
    export COT_RUBRIC_NDCG_GAIN_DEVICE=cuda:0
    export COT_RUBRIC_NDCG_GAIN_QUERY_BATCH_SIZE="$REFERENCE_BATCH_SIZE"
    CUDA_VISIBLE_DEVICES=0 "$VENV/bin/python" "$REFERENCE_PRECOMPUTE" \
      --input "$ENRICHED_DATASET" \
      --output "$CACHED_DATASET" \
      --expected-rows "$EXPECTED_ROWS" \
      --batch-size "$REFERENCE_BATCH_SIZE" \
      --ndcg-k "$NDCG_K" \
      --temperature "$SIMILARITY_TEMPERATURE" \
      --seed "$SEED"
    export COT_RUBRIC_NDCG_GAIN_QUERY_BATCH_SIZE="$REWARD_QUERY_BATCH_SIZE"
  fi
  cached_rows=$(wc -l < "$CACHED_DATASET" | tr -d ' ')
  if [[ "$cached_rows" != "$EXPECTED_ROWS" ]]; then
    echo "缓存 reference NDCG 的正式数据应为 $EXPECTED_ROWS 条，当前为 $cached_rows。" >&2
    exit 1
  fi
  if [[ -e "$OUT" && -n "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "正式输出目录已存在且非空，拒绝覆盖: $OUT" >&2
    exit 1
  fi
  mkdir -p "$OUT"
  export COT_RUBRIC_NDCG_GAIN_COMPONENT_LOG="$OUT/reward_components_rank{rank}.jsonl"
  nvidia-smi -q -d ECC > "$OUT/gpu_ecc_before_train.txt"
  nvidia-smi -q -d ROW_REMAPPER > "$OUT/gpu_row_remapper_before_train.txt"
  formal_api_per_key=$((RUBRIC_API_CONCURRENCY_PER_KEY / FORMAL_WORLD_SIZE))
  if ((formal_api_per_key * FORMAL_WORLD_SIZE != RUBRIC_API_CONCURRENCY_PER_KEY)); then
    echo "每-key总并发必须能被正式 DDP 进程数整除。" >&2
    exit 1
  fi
  run_swift "$CACHED_DATASET" "$OUT" "$GENERATION_BATCH_SIZE" "$BATCH_SIZE" "$VLLM_MAX_NUM_SEQS" full "$FORMAL_CUDA_VISIBLE_DEVICES" "$FORMAL_WORLD_SIZE" "$formal_api_per_key"
fi
