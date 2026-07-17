#!/bin/zsh
set -euo pipefail

cd /Users/tanqing/Desktop/aaai_pro

OUTPUT_DIR=manu_src/datas/CDs_and_Vinyl/cot/api/history_only_next_item_feature_cot__input_time_title_rating_store_categories_desc256_details256_v1
OUTPUT_FILE=$OUTPUT_DIR/cot_non_target_glm52_test_full_seed42_temp1.jsonl

mkdir -p $OUTPUT_DIR
exec >> $OUTPUT_FILE.screen.log 2>&1

exec python3 manu_src/scripts/cot/generate_non_target_cot_full.py \
  --input manu_src/datas/CDs_and_Vinyl/train_datas/time_title_rating_store_categories_desc256_details256_v1.0/test.jsonl \
  --output $OUTPUT_FILE \
  --providers ks_tokenverse glm_official \
  --item-type 'CD or vinyl release' \
  --split test \
  --language en \
  --seed 42 \
  --request-interval-seconds 1 \
  --max-workers 10 \
  --attempts-per-round 4 \
  --retry-round-delay-seconds 30 \
  --max-output-words 512 \
  --max-tokens 2048 \
  --temperature 1.0 \
  --top-p 0.9 \
  --progress-every 50
