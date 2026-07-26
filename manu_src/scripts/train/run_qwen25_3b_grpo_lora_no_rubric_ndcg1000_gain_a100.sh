#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/user/aaai_pro}
VENV=${VENV:-/home/user/.conda/envs/aaai_pro}
MODEL=${MODEL:-$ROOT/manu_src/model_outputs/CDs_and_Vinyl/sft/qwen25_3b_full_sft20_disjoint_time_title_rating_store_categories_desc256_details256_v1_bs16_ga1_lr2e5_ep1_len4096_seed42/v0-20260718-080839/checkpoint-134}

SPLIT_DIR=${SPLIT_DIR:-$ROOT/manu_src/datas/CDs_and_Vinyl/sft_grpo/disjoint_example20_80__input_time_title_rating_store_categories_desc256_details256_v1}
DATASET=${DATASET:-$SPLIT_DIR/grpo_rubric_ndcg1000_gain_cached_reference_messages_train80_seed42_n8578.jsonl}
ITEM_INFO=${ITEM_INFO:-$ROOT/manu_src/datas/CDs_and_Vinyl/arrow_to_jsonls/item_info.jsonl}
REWARD_PLUGIN=${REWARD_PLUGIN:-$ROOT/manu_src/scripts/train/cot_sim_ndcg1000_gain_reward.py}
REWARD_EMBEDDING=${REWARD_EMBEDDING:-$ROOT/manu_src/model_outputs/CDs_and_Vinyl/embedding/qwen3emb06b_history_plus_glm52_non_target_cot_input_time_title_rating_store_categories_desc256_details256_v1_bs128_ga1_lr2e5_ep5_len4096_seed42/checkpoint-epoch-01}

RUN_NAME=${RUN_NAME:-qwen25_3b_fullsft20_grpolora80_single_gpu_colocate_cottrained_simz0p6_ndcg1000gainz0p4_no_rubric_g4_genbs32_bs8_ga1_vllmsleep1_vllm0p10_lr2e5_ep3_vllmlen4608_clen512_loradrop0_seed42}
OUT=${OUT:-$ROOT/manu_src/model_outputs/CDs_and_Vinyl/grpo/$RUN_NAME}

EXPECTED_ROWS=${EXPECTED_ROWS:-8578}
EXPECTED_ITEMS=${EXPECTED_ITEMS:-12000}
SEED=${SEED:-42}
NUM_GENERATIONS=${NUM_GENERATIONS:-4}
GENERATION_BATCH_SIZE=${GENERATION_BATCH_SIZE:-32}
BATCH_SIZE=${BATCH_SIZE:-8}
GRAD_ACCUM=${GRAD_ACCUM:-1}
MAX_LENGTH=${MAX_LENGTH:-4096}
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-512}
GENERATION_TEMPERATURE=${GENERATION_TEMPERATURE:-1.0}
GENERATION_TOP_K=${GENERATION_TOP_K:-200}
GENERATION_TOP_P=${GENERATION_TOP_P:-1.0}
EPOCHS=${EPOCHS:-3}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
WARMUP_STEPS=${WARMUP_STEPS:-64}
BETA=${BETA:-0.04}
LORA_RANK=${LORA_RANK:-64}
LORA_ALPHA=${LORA_ALPHA:-128}
LORA_DROPOUT=${LORA_DROPOUT:-0}
SAVE_STEPS=${SAVE_STEPS:-300}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-50}

NDCG_K=${NDCG_K:-1000}
SIMILARITY_TEMPERATURE=${SIMILARITY_TEMPERATURE:-0.05}
SIMILARITY_WEIGHT=${SIMILARITY_WEIGHT:-0.6}
GAIN_WEIGHT=${GAIN_WEIGHT:-0.4}
ZSCORE_EPSILON=${ZSCORE_EPSILON:-1e-6}
REWARD_ITEM_BATCH_SIZE=${REWARD_ITEM_BATCH_SIZE:-128}
REWARD_QUERY_BATCH_SIZE=${REWARD_QUERY_BATCH_SIZE:-16}

USE_VLLM=${USE_VLLM:-true}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.10}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-4608}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-32}
VLLM_SLEEP_LEVEL=${VLLM_SLEEP_LEVEL:-1}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
CONFIRM_RUN=${CONFIRM_RUN:-0}

print_param() {
  local name=$1 value=$2 description=$3
  printf '  %-42s %s\n      %s\n' "$name=$value" "" "$description"
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
  "$DATASET" \
  "$ITEM_INFO" \
  "$REWARD_PLUGIN" \
  "$REWARD_EMBEDDING"; do
  require_path "无 Rubric GRPO 依赖" "$dependency"
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
  echo "当前对照固定 gradient_accumulation_steps=1、num_generations=4。" >&2
  exit 1
fi
if [[ "$CUDA_VISIBLE_DEVICES" != "0" ]]; then
  echo "当前服务器只有一张 A100，本实验固定使用 GPU0。" >&2
  exit 1
fi
if ((BATCH_SIZE % NUM_GENERATIONS != 0)); then
  echo "BATCH_SIZE 必须能被 NUM_GENERATIONS 整除，保证每次反向包含完整候选组。" >&2
  exit 1
fi
if ((GENERATION_BATCH_SIZE % BATCH_SIZE != 0)); then
  echo "GENERATION_BATCH_SIZE 必须能被 BATCH_SIZE 整除。" >&2
  exit 1
fi
if [[ "$MAX_LENGTH" != "4096" || "$MAX_COMPLETION_LENGTH" != "512" || "$VLLM_MAX_MODEL_LEN" != "4608" ]]; then
  echo "本实验固定 prompt=4096、completion=512、vLLM context=4608。" >&2
  exit 1
fi
if [[ "$NDCG_K" != "1000" || "$SIMILARITY_WEIGHT" != "0.6" || "$GAIN_WEIGHT" != "0.4" ]]; then
  echo "无 Rubric 对照固定 K=1000、similarity weight=0.6、gain weight=0.4。" >&2
  exit 1
fi
if [[ "$EPOCHS" != "3" ]]; then
  echo "当前无 Rubric 对照按用户确认固定训练 3 个 epoch。" >&2
  exit 1
fi

dataset_rows=$(wc -l < "$DATASET" | tr -d ' ')
item_rows=$(wc -l < "$ITEM_INFO" | tr -d ' ')
if [[ "$dataset_rows" != "$EXPECTED_ROWS" ]]; then
  echo "GRPO 数据应为 $EXPECTED_ROWS 条，当前为 $dataset_rows。" >&2
  exit 1
fi
if ((item_rows < EXPECTED_ITEMS)); then
  echo "item_info 行数少于完整候选数 $EXPECTED_ITEMS，当前为 $item_rows。" >&2
  exit 1
fi

"$VENV/bin/python" - "$DATASET" "$EXPECTED_ROWS" "$NDCG_K" <<'PY'
import json
import re
import sys
from pathlib import Path

path = Path(sys.argv[1])
expected_rows = int(sys.argv[2])
expected_k = int(sys.argv[3])
asin_pattern = re.compile(r"(?<![A-Z0-9])B0[A-Z0-9]{8}(?![A-Z0-9])")
counts = {
    "rows": 0,
    "wrong_split": 0,
    "missing_history": 0,
    "missing_target": 0,
    "missing_reference": 0,
    "wrong_reference_k": 0,
    "target_text_truncated": 0,
    "reference_cot_truncated": 0,
    "history_raw_asin": 0,
    "target_text_in_messages": 0,
}
for line_number, line in enumerate(path.open(encoding="utf-8"), 1):
    if not line.strip():
        continue
    row = json.loads(line)
    counts["rows"] += 1
    history = str(row.get("user_history") or "")
    target_text = str(row.get("target_item_text") or "")
    reference_cot = str(row.get("reference_cot") or "")
    messages_text = json.dumps(row.get("messages") or [], ensure_ascii=False)
    counts["wrong_split"] += int(row.get("split") != "train")
    counts["missing_history"] += int(not history)
    counts["missing_target"] += int(
        row.get("target_item_id") is None or not target_text
    )
    counts["missing_reference"] += int(
        row.get("reference_ndcg") is None and row.get("reference_rank") is None
    )
    counts["wrong_reference_k"] += int(
        row.get("reference_ndcg_k") not in {None, expected_k}
    )
    counts["target_text_truncated"] += int("[TRUNCATED]" in target_text)
    counts["reference_cot_truncated"] += int("[TRUNCATED]" in reference_cot)
    counts["history_raw_asin"] += int(bool(asin_pattern.search(history)))
    counts["target_text_in_messages"] += int(
        bool(target_text) and target_text in messages_text
    )
print("训练输入审计:", json.dumps(counts, ensure_ascii=False, sort_keys=True))
if counts["rows"] != expected_rows:
    raise SystemExit("审计行数与预期不一致")
for name, value in counts.items():
    if name != "rows" and value:
        raise SystemExit(f"训练输入审计失败: {name}={value}")
PY

prompts_per_step=$((BATCH_SIZE / NUM_GENERATIONS * GRAD_ACCUM))
steps_per_epoch=$(((EXPECTED_ROWS + prompts_per_step - 1) / prompts_per_step))
total_steps=$((steps_per_epoch * EPOCHS))
steps_per_generation=$((GENERATION_BATCH_SIZE / BATCH_SIZE))

echo "Qwen2.5-3B 无 Rubric NDCG@1000 Gain LoRA GRPO 参数："
print_param GPU "$CUDA_VISIBLE_DEVICES" "当前服务器仅使用物理 GPU0：NVIDIA A100 80GB。"
print_param MODEL "$MODEL" "前 20% 数据训练 1 epoch 的全参数 SFT checkpoint；GRPO 阶段新建 LoRA。"
print_param DATASET "$DATASET" "后 80% 的 8578 条 GRPO 样本；policy messages 只含 history，reference 指标只作为 reward metadata。"
print_param INPUT_SCHEMA time_title_rating_store_categories_desc256_details256_v1 "SFT、GRPO history、reward embedding 和后续测试统一使用该字段口径。"
print_param TARGET_TEXT_TRUNCATION disabled "target_item_text 仅用于监督检索，审计要求完整文本且不含 [TRUNCATED]。"
print_param REFERENCE "$DATASET:reference_ndcg/reference_rank" "读取固定 API CoT 的离线缓存排名；reference CoT 不进入 policy prompt，也不在训练时重新编码。"
print_param ITEM_INFO "$ITEM_INFO" "冻结 12000-item 候选表；屏蔽历史物品时保留监督 target。"
print_param REWARD_EMBEDDING "$REWARD_EMBEDDING" "冻结的 CoT-trained embedding epoch-01；new/reference 使用同一候选表和 seen-item mask。"
print_param REWARD_FUNC cot_sim_ndcg1000_gain "独立无 Rubric reward 插件；不导入 Rubric scorer，不调用 API，也不读取 target 文本评分。"
print_param REWARD_FORMULA "0.6*z(similarity)+0.4*z(delta_ndcg@1000)" "delta_ndcg=new-reference；similarity 与 gain 分别在同一 history 的四候选组内标准化。"
print_param SIMILARITY_TEMPERATURE "$SIMILARITY_TEMPERATURE" "相似度项使用 seen-mask 后 12000-item 全候选 log-softmax，温度为 0.05。"
print_param NDCG_K "$NDCG_K" "new/reference 均按目标物品排名计算 NDCG@1000。"
print_param ZSCORE_EPSILON "$ZSCORE_EPSILON" "组内标准差小于该值时，该 reward 分量置 0，避免数值放大。"
print_param TRAINER_SCALE_REWARDS group "组合 reward 交给标准 GRPO 再计算同组 advantage；四条候选全部进入 loss。"
print_param NUM_GENERATIONS "$NUM_GENERATIONS" "每个 history 采样四条 completion，组成一个 GRPO group。"
print_param GENERATION_BATCH_SIZE "$GENERATION_BATCH_SIZE" "vLLM 每轮生成 32 条 completion，对应八个 history 组。"
print_param BATCH_SIZE "$BATCH_SIZE" "单卡每次反向传播八条 completion，对应两个完整候选组。"
print_param STEPS_PER_GENERATION "$steps_per_generation" "每轮 32 条 rollout 拆成四个 optimizer step。"
print_param GRAD_ACCUM "$GRAD_ACCUM" "梯度累积固定为 1。"
print_param EPOCHS "$EPOCHS" "每个 epoch 约 $steps_per_epoch 个 optimizer step，三个 epoch 总计约 $total_steps step。"
print_param MAX_LENGTH "$MAX_LENGTH" "policy prompt 与 reward query 上限 4096 tokens；policy 沿用左截断规则。"
print_param MAX_COMPLETION_LENGTH "$MAX_COMPLETION_LENGTH" "每条 CoT 最多生成 512 tokens。"
print_param ROLLOUT_SAMPLING "temperature=$GENERATION_TEMPERATURE, top_k=$GENERATION_TOP_K, top_p=$GENERATION_TOP_P" "沿用 Rubric 版本的 rollout 采样设置，固定奖励函数以外的生成变量。"
print_param OPTIMIZATION "lr=$LEARNING_RATE, beta=$BETA, warmup=$WARMUP_STEPS, weight_decay=$WEIGHT_DECAY" "LoRA GRPO 的学习率、KL 系数、warmup step 和 AdamW 权重衰减。"
print_param LORA "rank=$LORA_RANK, alpha=$LORA_ALPHA, dropout=$LORA_DROPOUT" "完整 SFT 基座冻结，只更新新建的 GRPO LoRA 参数。"
print_param VLLM "colocate, utilization=$VLLM_GPU_MEMORY_UTILIZATION, context=$VLLM_MAX_MODEL_LEN, max_seqs=$VLLM_MAX_NUM_SEQS, sleep=$VLLM_SLEEP_LEVEL" "vLLM 与训练位于同一张 A100；rollout 后释放权重和 KV cache。"
print_param SAVE "every $SAVE_STEPS steps, limit=$SAVE_TOTAL_LIMIT" "每 300 optimizer step 保存 checkpoint，三个 epoch 预计产生约 43 个周期 checkpoint；上限 50 可保留全部 checkpoint。"
print_param OUTPUT "$OUT" "独立保存无 Rubric checkpoint、trainer 日志和逐候选 reward 组件。"
print_param API_USAGE disabled "训练不请求 Tokenverse、GLM 或其它外部 API，也不依赖本地代理。"
print_param SEED "$SEED" "数据顺序、DataLoader、rollout 和训练随机种子固定为 42。"
print_param CONFIRM_RUN "$CONFIRM_RUN" "只有设置为 1 才启动正式训练；默认只打印并审计参数。"

if [[ "$CONFIRM_RUN" != "1" ]]; then
  echo "未启动训练。"
  exit 0
fi

if [[ -e "$OUT" && -n "$(find "$OUT" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
  echo "输出目录已存在且非空，拒绝覆盖: $OUT" >&2
  exit 1
fi

source /opt/miniforge3/bin/activate "$VENV"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES
export NPROC_PER_NODE=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

unset COT_RUBRIC_NDCG_GAIN_RUBRIC_SCORER
unset COT_RUBRIC_NDCG_GAIN_API_PROVIDER
unset COT_RUBRIC_NDCG_GAIN_API_MODEL

mkdir -p "$OUT"
nvidia-smi -q -d ECC > "$OUT/gpu_ecc_before_train.txt"
nvidia-smi -q -d ROW_REMAPPER > "$OUT/gpu_row_remapper_before_train.txt"

export COT_SIM_NDCG1000_GAIN_EMBEDDING_MODEL="$REWARD_EMBEDDING"
export COT_SIM_NDCG1000_GAIN_ITEM_INFO="$ITEM_INFO"
export COT_SIM_NDCG1000_GAIN_MAX_LENGTH="$MAX_LENGTH"
export COT_SIM_NDCG1000_GAIN_ITEM_BATCH_SIZE="$REWARD_ITEM_BATCH_SIZE"
export COT_SIM_NDCG1000_GAIN_QUERY_BATCH_SIZE="$REWARD_QUERY_BATCH_SIZE"
export COT_SIM_NDCG1000_GAIN_TEMPERATURE="$SIMILARITY_TEMPERATURE"
export COT_SIM_NDCG1000_GAIN_K="$NDCG_K"
export COT_SIM_NDCG1000_GAIN_SIM_WEIGHT="$SIMILARITY_WEIGHT"
export COT_SIM_NDCG1000_GAIN_GAIN_WEIGHT="$GAIN_WEIGHT"
export COT_SIM_NDCG1000_GAIN_ZSCORE_EPSILON="$ZSCORE_EPSILON"
export COT_SIM_NDCG1000_GAIN_GROUP_SIZE="$NUM_GENERATIONS"
export COT_SIM_NDCG1000_GAIN_STRICT_GROUP_SIZE=1
export COT_SIM_NDCG1000_GAIN_EXPECTED_ITEMS="$EXPECTED_ITEMS"
export COT_SIM_NDCG1000_GAIN_TORCH_DTYPE=bfloat16
export COT_SIM_NDCG1000_GAIN_ATTN_IMPLEMENTATION=flash_attention_2
export COT_SIM_NDCG1000_GAIN_DEVICE=cuda:0
export COT_SIM_NDCG1000_GAIN_LOG_EVERY=1
export COT_SIM_NDCG1000_GAIN_COMPONENT_LOG="$OUT/reward_components_rank{rank}.jsonl"

"$VENV/bin/swift" rlhf \
  --rlhf_type grpo \
  --model "$MODEL" \
  --model_type qwen2_5 \
  --template qwen2_5 \
  --dataset "$DATASET" \
  --external_plugins "$REWARD_PLUGIN" \
  --reward_funcs cot_sim_ndcg1000_gain \
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
