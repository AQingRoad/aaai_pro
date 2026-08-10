#!/usr/bin/env bash
set -euo pipefail

# Evaluate every full-parameter GRPO checkpoint from newest to oldest with the
# established one-shot CoT generation and 12,000-item retrieval protocol.

ROOT=${ROOT:-/home/user/aaai_pro}

export CHECKPOINT_TYPE=full
export GRPO_RUN=${GRPO_RUN:-$ROOT/manu_src/model_outputs/CDs_and_Vinyl/grpo/qwen25_3b_fullsft20_grpofull80_single_gpu_colocate_cottrained_simz0p6_ndcg1000gainz0p4_no_rubric_g4_genbs32_bs4_ga1_vllmsleep1_vllm0p10_lr2e5_ep3_vllmlen4608_clen512_seed42/v0-20260809-060445}
export MODE=all
export EVAL_SPLIT=test
export CHECKPOINT_STEPS=${CHECKPOINT_STEPS:-25728,25200,24000,22800,21600,20400,19200,18000,16800,15600,14400,13200,12000,10800,9600,8400,7200,6000,4800,3600,2400,1200}
export GENERATION_BATCH_SIZE=32
export TEMPERATURE=1.0
export TOP_K=200
export TOP_P=1.0
export MAX_PROMPT_TOKENS=4096
export MAX_NEW_TOKENS=512
export VLLM_MAX_MODEL_LEN=4608
export GPU_MEMORY_UTILIZATION=0.85
export KS=5,10,20,50,100
export SEED=42
export EVAL_ROOT=${EVAL_ROOT:-$ROOT/manu_src/model_outputs/CDs_and_Vinyl/eval/qwen25_3b_grpofull_no_rubric_gain0p4_beta0p04_ep3_test_oneshot_raw_completion_cottrained_epoch01_seed42}

exec bash "$ROOT/manu_src/scripts/eval/run_qwen25_grpo_checkpoints_vllm_valid_a100.sh"
