#!/usr/bin/env bash
set -euo pipefail

source /opt/miniforge3/bin/activate /home/user/.conda/envs/aaai_pro
cd /home/user/aaai_pro

# 减少长序列训练过程中的 CUDA 显存碎片。
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# 关闭 tokenizer 后台线程，保持运行日志稳定。
export TOKENIZERS_PARALLELISM=false

OUTPUT_DIR=/home/user/aaai_pro/manu_src/model_outputs/CDs_and_Vinyl/embedding/qwen3emb06b_time_title_rating_store_categories_desc256_details256_fieldtrim_bs128_ga1_lr2e5_ep5_len4096_seed42
mkdir -p "${OUTPUT_DIR}"

python -u manu_src/scripts/models/train_embedding.py \
  --model /home/user/models_hf/Qwen3-Embedding-0.6B \
  --train-file manu_src/datas/CDs_and_Vinyl/train_datas/time_title_rating_store_categories_desc256_details256_v1.0/train.jsonl \
  --test-file manu_src/datas/CDs_and_Vinyl/train_datas/time_title_rating_store_categories_desc256_details256_v1.0/test.jsonl \
  --output-dir "${OUTPUT_DIR}" \
  --max-length 4096 \
  --batch-size 128 \
  --grad-accum 1 \
  --epochs 5 \
  --learning-rate 2e-5 \
  --weight-decay 0.01 \
  --warmup-ratio 0.03 \
  --temperature 0.05 \
  --seed 42 \
  --attn-implementation flash_attention_2 \
  2>&1 | tee "${OUTPUT_DIR}/train.log"
