#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/home/user/aaai_pro}
VENV=${VENV:-/home/user/.conda/envs/aaai_pro}
MODEL=${MODEL:-$ROOT/manu_src/model_outputs/CDs_and_Vinyl/sft/qwen25_3b_full_sft20_disjoint_time_title_rating_store_categories_desc256_details256_v1_bs16_ga1_lr2e5_ep1_len4096_seed42/v0-20260718-080839/checkpoint-134}
DATASET=${DATASET:-$ROOT/manu_src/datas/CDs_and_Vinyl/sft_grpo/disjoint_example20_80__input_time_title_rating_store_categories_desc256_details256_v1/grpo_rubric_ndcg1000_gain_cached_reference_messages_train80_seed42_n8578.jsonl}
ITEM_INFO=${ITEM_INFO:-$ROOT/manu_src/datas/CDs_and_Vinyl/arrow_to_jsonls/item_info.jsonl}
REWARD_PLUGIN=${REWARD_PLUGIN:-$ROOT/manu_src/scripts/train/cot_sim_ndcg1000_gain_reward.py}
REWARD_EMBEDDING=${REWARD_EMBEDDING:-$ROOT/manu_src/model_outputs/CDs_and_Vinyl/embedding/qwen3emb06b_history_plus_glm52_non_target_cot_input_time_title_rating_store_categories_desc256_details256_v1_bs128_ga1_lr2e5_ep5_len4096_seed42/checkpoint-epoch-01}

PROBE_NAME=${PROBE_NAME:-qwen25_3b_grpo_full_batch4_gain0p4_beta0p04_10step_seed42}
OUT=${OUT:-$ROOT/manu_src/model_outputs/CDs_and_Vinyl/grpo/probes/$PROBE_NAME}
SUMMARY=${SUMMARY:-$ROOT/experiments/results/cds_qwen25_3b_grpo_full_batch4_probe_seed42.json}

EXPECTED_ROWS=${EXPECTED_ROWS:-8578}
EXPECTED_ITEMS=${EXPECTED_ITEMS:-12000}
SEED=${SEED:-42}
MAX_STEPS=${MAX_STEPS:-10}
NUM_GENERATIONS=${NUM_GENERATIONS:-4}
GENERATION_BATCH_SIZE=${GENERATION_BATCH_SIZE:-32}
BATCH_SIZE=${BATCH_SIZE:-4}
GRAD_ACCUM=${GRAD_ACCUM:-1}
MAX_LENGTH=${MAX_LENGTH:-4096}
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-512}
GENERATION_TEMPERATURE=${GENERATION_TEMPERATURE:-1.0}
GENERATION_TOP_K=${GENERATION_TOP_K:-200}
GENERATION_TOP_P=${GENERATION_TOP_P:-1.0}
LEARNING_RATE=${LEARNING_RATE:-2e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.01}
WARMUP_STEPS=${WARMUP_STEPS:-64}
BETA=${BETA:-0.04}

NDCG_K=${NDCG_K:-1000}
SIMILARITY_TEMPERATURE=${SIMILARITY_TEMPERATURE:-0.05}
SIMILARITY_WEIGHT=${SIMILARITY_WEIGHT:-0.6}
GAIN_WEIGHT=${GAIN_WEIGHT:-0.4}
ZSCORE_EPSILON=${ZSCORE_EPSILON:-1e-6}
REWARD_ITEM_BATCH_SIZE=${REWARD_ITEM_BATCH_SIZE:-128}
REWARD_QUERY_BATCH_SIZE=${REWARD_QUERY_BATCH_SIZE:-16}

VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.10}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-4608}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-32}
VLLM_SLEEP_LEVEL=${VLLM_SLEEP_LEVEL:-1}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
CONFIRM_RUN=${CONFIRM_RUN:-0}

print_param() {
  local name=$1 value=$2 description=$3
  printf '  %-40s %s\n      %s\n' "$name=$value" "" "$description"
}

for path in \
  "$ROOT/AGENTS.md" \
  "$VENV/bin/python" \
  "$VENV/bin/swift" \
  "$MODEL/config.json" \
  "$DATASET" \
  "$ITEM_INFO" \
  "$REWARD_PLUGIN" \
  "$REWARD_EMBEDDING"; do
  if [[ ! -e "$path" ]]; then
    echo "缺少探测依赖：$path" >&2
    exit 1
  fi
done

if [[ -e "$MODEL/adapter_config.json" ]]; then
  echo "全参数 GRPO 的 MODEL 必须是完整模型，不能是 LoRA adapter。" >&2
  exit 1
fi
if [[ "$SEED" != "42" || "$BATCH_SIZE" != "4" || "$GRAD_ACCUM" != "1" ]]; then
  echo "本次探测固定 seed=42、train batch=4、gradient accumulation=1。" >&2
  exit 1
fi
if [[ "$MAX_STEPS" != "10" || "$NUM_GENERATIONS" != "4" || "$GENERATION_BATCH_SIZE" != "32" ]]; then
  echo "本次探测固定 max_steps=10、group size=4、generation batch=32。" >&2
  exit 1
fi
if [[ "$MAX_LENGTH" != "4096" || "$MAX_COMPLETION_LENGTH" != "512" || "$VLLM_MAX_MODEL_LEN" != "4608" ]]; then
  echo "本次探测固定 prompt=4096、completion=512、vLLM context=4608。" >&2
  exit 1
fi
if [[ "$CUDA_VISIBLE_DEVICES" != "0" ]]; then
  echo "本次探测固定使用 GPU0。" >&2
  exit 1
fi
if ((GENERATION_BATCH_SIZE % BATCH_SIZE != 0 || BATCH_SIZE % NUM_GENERATIONS != 0)); then
  echo "generation batch、train batch 与 group size 不满足整除关系。" >&2
  exit 1
fi
if [[ "$(wc -l < "$DATASET" | tr -d ' ')" != "$EXPECTED_ROWS" ]]; then
  echo "探测数据行数应为 $EXPECTED_ROWS。" >&2
  exit 1
fi
if (( $(wc -l < "$ITEM_INFO") < EXPECTED_ITEMS )); then
  echo "item_info 少于 $EXPECTED_ITEMS 行。" >&2
  exit 1
fi

steps_per_generation=$((GENERATION_BATCH_SIZE / BATCH_SIZE))

echo "Qwen2.5-3B 全参数 GRPO batch=4 短探测参数："
print_param GPU "$CUDA_VISIBLE_DEVICES" "单张 NVIDIA A100 80GB；运行前检查 GPU 空闲和 ECC 状态。"
print_param MODEL "$MODEL" "前 20% 数据全参数 SFT checkpoint；policy 和 reference 均从该权重初始化。"
print_param TRAIN_TYPE full "更新 Qwen2.5-3B 全部参数，不创建 LoRA adapter。"
print_param DATASET "$DATASET" "后 80% 的 8578 条 GRPO 数据；target 只进入 reward。"
print_param REWARD "0.6*z(similarity)+0.4*z(delta_NDCG@1000)" "使用当前主方法的无 Rubric reward，不调用外部 API。"
print_param KL_BETA "$BETA" "固定 reference policy 的 KL 系数，与主方法 β=0.04 一致。"
print_param BATCH_SIZE "$BATCH_SIZE" "每次反向传播 4 条 completion，覆盖一个完整四候选组。"
print_param GENERATION_BATCH_SIZE "$GENERATION_BATCH_SIZE" "每轮生成 32 条 completion，拆成 $steps_per_generation 个 optimizer step。"
print_param NUM_GENERATIONS "$NUM_GENERATIONS" "每个 history 生成 4 条候选。"
print_param GRAD_ACCUM "$GRAD_ACCUM" "梯度累积固定为 1，物理 batch 与优化 batch 相同。"
print_param MAX_STEPS "$MAX_STEPS" "只跑 10 step，覆盖首次 AdamW 状态分配和第二轮 rollout。"
print_param CONTEXT "$MAX_LENGTH+$MAX_COMPLETION_LENGTH=$VLLM_MAX_MODEL_LEN" "prompt 左截断至 4096，completion 最多 512 token。"
print_param SAMPLING "temperature=$GENERATION_TEMPERATURE, top_k=$GENERATION_TOP_K, top_p=$GENERATION_TOP_P" "与现有 GRPO rollout 口径一致。"
print_param OPTIMIZATION "lr=$LEARNING_RATE, weight_decay=$WEIGHT_DECAY, warmup_steps=$WARMUP_STEPS" "学习率只保持对照一致；本轮只判断显存和功能兼容性。"
print_param VLLM "colocate, utilization=$VLLM_GPU_MEMORY_UTILIZATION, max_seqs=$VLLM_MAX_NUM_SEQS, sleep=$VLLM_SLEEP_LEVEL" "全参数 policy 每轮将完整权重同步到同卡 vLLM。"
print_param SAVE_STRATEGY no "探测不保存 checkpoint；结束后删除 completion、日志和临时显存采样。"
print_param SUMMARY "$SUMMARY" "只保留退出状态、完成 step、峰值显存和错误类别。"
print_param SEED "$SEED" "数据、rollout 与训练随机种子固定为 42。"
print_param CONFIRM_RUN "$CONFIRM_RUN" "必须为 1 才启动短探测。"

if [[ "$CONFIRM_RUN" != "1" ]]; then
  echo "未启动探测。"
  exit 0
fi

if nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits | grep -Eq '[0-9]'; then
  echo "GPU 上存在计算进程，拒绝启动探测。" >&2
  nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader
  exit 1
fi
case "$OUT" in
  "$ROOT"/manu_src/model_outputs/CDs_and_Vinyl/grpo/probes/*) ;;
  *) echo "OUT 必须位于 GRPO probes 目录。" >&2; exit 1 ;;
esac
if [[ -e "$OUT" ]]; then
  echo "探测临时目录已存在，拒绝覆盖：$OUT" >&2
  exit 1
fi

source /opt/miniforge3/bin/activate "$VENV"
cd "$ROOT"
export CUDA_VISIBLE_DEVICES
export NPROC_PER_NODE=1
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

mkdir -p "$OUT" "$(dirname "$SUMMARY")"
nvidia-smi -q -d ECC > "$OUT/gpu_ecc_before.txt"
nvidia-smi -q -d ROW_REMAPPER > "$OUT/gpu_row_remapper_before.txt"

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

GPU_CSV="$OUT/gpu_memory.csv"
LOG="$OUT/probe.log"
(
  while true; do
    printf '%s,' "$(date -u +%FT%TZ)"
    nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits
    sleep 1
  done
) > "$GPU_CSV" 2>/dev/null &
MONITOR_PID=$!

set +e
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
  --max_steps "$MAX_STEPS" \
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
  --train_type full \
  --learning_rate "$LEARNING_RATE" \
  --weight_decay "$WEIGHT_DECAY" \
  --lr_scheduler_type cosine \
  --warmup_ratio 0 \
  --warmup_steps "$WARMUP_STEPS" \
  --torch_dtype bfloat16 \
  --attn_impl flash_attn \
  --gradient_checkpointing true \
  --use_vllm true \
  --vllm_mode colocate \
  --vllm_tensor_parallel_size 1 \
  --vllm_pipeline_parallel_size 1 \
  --vllm_gpu_memory_utilization "$VLLM_GPU_MEMORY_UTILIZATION" \
  --vllm_max_model_len "$VLLM_MAX_MODEL_LEN" \
  --vllm_max_num_seqs "$VLLM_MAX_NUM_SEQS" \
  --sleep_level "$VLLM_SLEEP_LEVEL" \
  --vllm_enable_lora false \
  --seed "$SEED" \
  --data_seed "$SEED" \
  --save_strategy no \
  --save_only_model false \
  --logging_steps 1 \
  --log_completions true \
  --dataloader_num_workers 0 \
  --report_to none \
  --output_dir "$OUT/run" 2>&1 | tee "$LOG"
STATUS=${PIPESTATUS[0]}
set -e

kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true

"$VENV/bin/python" - "$LOG" "$GPU_CSV" "$SUMMARY" "$STATUS" <<'PY'
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

log_path, gpu_path, summary_path = map(Path, sys.argv[1:4])
status = int(sys.argv[4])
text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
steps = [int(value) for value in re.findall(r"'global_step/max_steps': '(\d+)/10'", text)]
gpu_rows = []
if gpu_path.exists():
    for line in gpu_path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 4:
            try:
                gpu_rows.append((parts[0], int(parts[1]), int(parts[2]), int(parts[3])))
            except ValueError:
                pass
summary = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "exit_status": status,
    "success": status == 0 and max(steps, default=0) == 10,
    "last_completed_step": max(steps, default=0),
    "peak_gpu_memory_mib": max((row[1] for row in gpu_rows), default=None),
    "gpu_total_memory_mib": max((row[2] for row in gpu_rows), default=None),
    "peak_gpu_utilization_percent": max((row[3] for row in gpu_rows), default=None),
    "oom": "CUDA out of memory" in text or "OutOfMemoryError" in text,
    "sigabrt": "SIGABRT" in text or "Fatal Python error" in text,
    "traceback": "Traceback (most recent call last)" in text,
    "config": {
        "model": "Qwen2.5-3B-Instruct full-SFT checkpoint-134",
        "train_type": "full",
        "per_device_train_batch_size": 4,
        "generation_batch_size": 32,
        "num_generations": 4,
        "gradient_accumulation_steps": 1,
        "max_prompt_tokens": 4096,
        "max_completion_tokens": 512,
        "beta": 0.04,
        "reward": "0.6*z(similarity)+0.4*z(delta_ndcg@1000)",
        "seed": 42,
    },
    "log_tail": text.splitlines()[-40:],
}
summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY

rm -rf -- "$OUT"
exit "$STATUS"
