#!/usr/bin/env bash
set -euo pipefail

CONFIG_FILE="configs/glm_codeplan.env"
if [[ -f "${CONFIG_FILE}" ]]; then
  # shellcheck disable=SC1090
  source "${CONFIG_FILE}"
fi

if [[ -z "${BIGMODEL_API_KEY:-}" ]]; then
  echo "Missing BIGMODEL_API_KEY. Fill configs/glm_codeplan.env or export BIGMODEL_API_KEY." >&2
  exit 2
fi

python3 scripts/generate_cot_candidate_lists.py \
  --input data/rrec_amazon/Musical_Instruments/examples.jsonl \
  --output outputs/rrec_amazon/Musical_Instruments/cot_candidate_lists_glm_codeplan_low.jsonl \
  --num-candidates 4 \
  --max-workers 16 \
  --resume \
  --api-provider glm_codeplan \
  --api-thinking enabled \
  --api-reasoning-effort medium \
  --api-timeout 300 \
  --api-max-retries 3 \
  --api-min-interval 0.5 \
  --max-new-tokens 4096 \
  --max-prompt-tokens 2048 \
  --top-p 0.9 \
  --temperatures 0.2,0.4,0.6,0.8
