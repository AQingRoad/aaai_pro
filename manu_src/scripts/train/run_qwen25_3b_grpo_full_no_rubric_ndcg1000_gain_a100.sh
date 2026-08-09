#!/usr/bin/env bash
set -euo pipefail

# Full-parameter counterpart of the established no-Rubric Gain=0.4 LoRA run.
# The shared launcher performs data/reward audits and preserves model files in
# every checkpoint while retaining optimizer state only in the newest one.

ROOT=${ROOT:-/home/user/aaai_pro}

export TRAIN_TYPE=full
export BATCH_SIZE=4
export GENERATION_BATCH_SIZE=32
export NUM_GENERATIONS=4
export GRAD_ACCUM=1
export EPOCHS=3
export LEARNING_RATE=2e-5
export BETA=0.04
export SIMILARITY_WEIGHT=0.6
export GAIN_WEIGHT=0.4
export NDCG_K=1000
export MAX_LENGTH=4096
export MAX_COMPLETION_LENGTH=512
export VLLM_MAX_MODEL_LEN=4608
export VLLM_GPU_MEMORY_UTILIZATION=0.10
export VLLM_MAX_NUM_SEQS=32
export VLLM_SLEEP_LEVEL=1
export SAVE_STEPS=1200
export SAVE_TOTAL_LIMIT=30
export SEED=42

export RUN_NAME=${RUN_NAME:-qwen25_3b_fullsft20_grpofull80_single_gpu_colocate_cottrained_simz0p6_ndcg1000gainz0p4_no_rubric_g4_genbs32_bs4_ga1_vllmsleep1_vllm0p10_lr2e5_ep3_vllmlen4608_clen512_seed42}

exec bash "$ROOT/manu_src/scripts/train/run_qwen25_3b_grpo_lora_no_rubric_ndcg1000_gain_a100.sh"
