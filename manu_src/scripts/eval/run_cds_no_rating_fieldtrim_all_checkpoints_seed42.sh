#!/usr/bin/env bash
set -euo pipefail

source /opt/miniforge3/bin/activate /home/user/.conda/envs/aaai_pro
cd /home/user/aaai_pro

# 减少长序列评测过程中的 CUDA 显存碎片。
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 关闭 tokenizer 后台线程，保持日志顺序稳定。
export TOKENIZERS_PARALLELISM=false

MODEL_DIR=/home/user/aaai_pro/manu_src/model_outputs/CDs_and_Vinyl/embedding/qwen3emb06b_time_title_store_categories_desc256_details256_fieldtrim_bs128_ga1_lr2e5_ep5_len4096_seed42
OUTPUT_DIR=/home/user/aaai_pro/manu_src/eval_results/CDs_and_Vinyl/embedding/qwen3emb06b_time_title_store_categories_desc256_details256_fieldtrim_bs128_ga1_lr2e5_ep5_len4096_seed42_test_fullset
TEST_FILE=manu_src/datas/CDs_and_Vinyl/train_datas/time_title_store_categories_desc256_details256_v1.0/test.jsonl
ITEM_INFO=manu_src/datas/CDs_and_Vinyl/arrow_to_jsonls/item_info.jsonl

mkdir -p "${OUTPUT_DIR}"
exec > >(tee -a "${OUTPUT_DIR}/eval.log") 2>&1

for epoch in 01 02 03 04 05; do
  checkpoint="${MODEL_DIR}/checkpoint-epoch-${epoch}"
  test -s "${checkpoint}/model.safetensors"
  python -u manu_src/scripts/eval/evaluate_embedding_fullset.py \
    --checkpoint "${checkpoint}" \
    --test-file "${TEST_FILE}" \
    --item-info "${ITEM_INFO}" \
    --output "${OUTPUT_DIR}/checkpoint-epoch-${epoch}_test_fullset.json" \
    --ranks-output "${OUTPUT_DIR}/checkpoint-epoch-${epoch}_test_ranks.jsonl" \
    --max-length 4096 \
    --item-batch-size 128 \
    --query-batch-size 128 \
    --score-batch-size 128 \
    --ks 5,10,20 \
    --seed 42 \
    --attn-implementation flash_attention_2
done
