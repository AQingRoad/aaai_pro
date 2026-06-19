# Rubric-Gated CoT Recommendation Pipeline

This project implements the pipeline described in `method.md`:

1. Prepare recommendation examples with held-out positive target items.
2. Generate multiple CoT candidates with local Qwen3-4B.
3. Judge each CoT with a recommendation-specific 5D rubric.
4. Compute CoT gain against a frozen embedder proxy.
5. Select data by `rubric_score_norm * max(cot_gain, 0)`.
6. Build SFT and GRPO datasets for ms-swift.
7. Run SFT and Rubric-Gated GRPO.

The server already has the model and environment used by the scripts:

```bash
MODEL=/root/autodl-tmp/modelscope_cache/models/Qwen/Qwen3-4B
QWEN3_EMBEDDING_MODEL=/root/autodl-tmp/modelscope_cache/models/Qwen/Qwen3-Embedding-0.6B
VENV=/root/autodl-tmp/rec/ms-swift-312-cu124-venv
SOURCE_ROOT=/root/autodl-tmp/rec/Open-World-Knowledge-Augmented-Recommendation
ROOT=/root/autodl-tmp/rec/aaai_pro
```

## Full Training Runner

For a new server, edit `configs/rrec_full_pipeline.example.env` and then run:

```bash
cd /root/autodl-tmp/rec/aaai_pro
PIPELINE_ENV_FILE=configs/rrec_full_pipeline.example.env \
bash scripts/run_rrec_full_training_pipeline.sh
```

For a local config with the real API key, use:

```bash
cp configs/rrec_full_pipeline.local.env.example configs/rrec_full_pipeline.local.env
# edit configs/rrec_full_pipeline.local.env and write BIGMODEL_API_KEY
PIPELINE_ENV_FILE=configs/rrec_full_pipeline.local.env \
bash scripts/run_rrec_full_training_pipeline.sh
```

The runner also auto-loads `configs/glm_codeplan.env` if it exists.

This runner links the full chain:

```text
prepare examples -> train embedder -> generate CoT candidates -> rubric score
-> merge -> gain -> select -> SFT dataset -> GRPO dataset -> SFT -> GRPO
```

### Training Only on a New Server

If the phase0 embedding dataset and the SFT/GRPO JSONL datasets were built
locally and copied to the training server, use the training config:

```bash
cd /root/autodl-tmp/rec/aaai_pro
cp configs/rrec_train_only.example.env configs/rrec_train_only.local.env
# edit EMBEDDER_DATASET, SFT_DATASET, GRPO_DATASET, MODEL,
# BASE_EMBEDDING_MODEL, GPU settings, and BIGMODEL_API_KEY if needed.
PIPELINE_ENV_FILE=configs/rrec_train_only.local.env \
bash scripts/run_rrec_full_training_pipeline.sh
```

The CDs artifacts committed in this repo are:

```text
github_artifacts/CDs_and_Vinyl/phase0/phase0_embedder_rrec_amazon_cds_and_vinyl_train.jsonl
github_artifacts/CDs_and_Vinyl/sft/sft_rrec_amazon_cds_and_vinyl_deepseek_v4_pro_cds_embedder_partial.jsonl
github_artifacts/CDs_and_Vinyl/grpo/grpo_rrec_amazon_cds_and_vinyl_deepseek_v4_pro_cds_embedder_disjoint_full.jsonl
```

`TRAIN_ONLY=1` skips these stages:

```text
prepare examples, embedding-data build, CoT generation, rubric scoring,
merge, gain, select, SFT/GRPO dataset construction
```

The script still trains the CDs embedding model from `EMBEDDER_DATASET`, then
uses the resulting checkpoint for GRPO reward computation. It reads
`SFT_DATASET` and `GRPO_DATASET` directly for training. If a required JSONL is
missing or empty, it exits before launching ms-swift.

It defaults to full-parameter SFT/GRPO. LoRA is optional:

```bash
export SFT_TRAIN_TYPE=lora
export GRPO_TRAIN_TYPE=lora
```

Multi-GPU is configured with visible-device lists and process counts. For
example, train on GPU 0-1 and run vLLM rollout on GPU 2-3:

```bash
export SFT_CUDA_VISIBLE_DEVICES=0,1
export SFT_NPROC_PER_NODE=2
export GRPO_CUDA_VISIBLE_DEVICES=0,1
export GRPO_NPROC_PER_NODE=2
export RUN_VLLM_SERVER=1
export VLLM_CUDA_VISIBLE_DEVICES=2,3
export VLLM_TENSOR_PARALLEL_SIZE=2
```

For a cheap server sanity check:

```bash
SMOKE=1 PIPELINE_ENV_FILE=configs/rrec_full_pipeline.example.env \
bash scripts/run_rrec_full_training_pipeline.sh
```

## Quick Smoke Test

```bash
cd /root/autodl-tmp/rec/aaai_pro
source configs/ml1m_qwen3_4b.env
bash scripts/run_smoke_test.sh
```

This runs a tiny end-to-end pipeline on ML-1M:

- `data/smoke/ml1m_examples.jsonl`
- `outputs/smoke/cot_candidates.jsonl`
- `outputs/smoke/cot_judged.jsonl`
- `outputs/smoke/cot_scored.jsonl`
- `outputs/smoke/filtered_high_quality_cot.jsonl`
- `outputs/smoke/sft.jsonl`
- `outputs/smoke/grpo.jsonl`

## R2ec / Amazon Reviews 2023 Benchmark

R2ec evaluates on Amazon Reviews 2023 with three categories:

- `Musical_Instruments`
- `Video_Games`
- `CDs_and_Vinyl`

The official R2ec preprocessing output can be converted into this pipeline's JSONL schema:

```bash
cd /root/autodl-tmp/rec/aaai_pro
source configs/rrec_amazon_qwen3_4b.env
CATEGORY=Musical_Instruments MAX_EXAMPLES=1000 NUM_CANDIDATES=4 bash scripts/run_phase1_full_rrec_amazon.sh
```

Change `CATEGORY` to `Video_Games` or `CDs_and_Vinyl` to process the other benchmark splits. Main outputs:

- `data/rrec_amazon/<CATEGORY>/phase1_examples.jsonl`
- `outputs/rrec_amazon/<CATEGORY>/filtered_high_quality_cot.jsonl`
- `outputs/rrec_amazon/<CATEGORY>/sft.jsonl`
- `outputs/rrec_amazon/<CATEGORY>/grpo.jsonl`

By default, R2ec Phase 1 now computes `CoT Gain` with Qwen3-Embedding:

```bash
GAIN_EMBEDDER_MODE=qwen3_embedding \
GRPO_BASELINE_EMBEDDER_MODE=qwen3_embedding \
QWEN3_EMBEDDING_MODEL=/root/autodl-tmp/modelscope_cache/models/Qwen/Qwen3-Embedding-0.6B \
bash scripts/run_phase1_full_rrec_amazon.sh
```

The Qwen3-Embedding backend uses instruction-aware query embeddings, document
embeddings without instructions, last-token pooling, and L2 normalization.
The lexical scorer remains available for fast debugging:

```bash
GAIN_EMBEDDER_MODE=lexical GRPO_BASELINE_EMBEDDER_MODE=lexical bash scripts/run_phase1_full_rrec_amazon.sh
```

### Continue From Scored Candidate Lists

If CoT candidate lists and candidate-level rubric scores already exist, continue
from those files without regenerating or rejudging CoT:

```bash
cd /root/autodl-tmp/rec/aaai_pro
source configs/rrec_amazon_qwen3_4b.env

CATEGORY=CDs_and_Vinyl \
RUN_NAME=deepseek_v4_pro_partial \
CANDIDATE_LISTS=$ROOT/outputs/rrec_amazon/CDs_and_Vinyl/cot_candidate_lists_deepseek_v4_pro_low.jsonl \
RUBRIC_SCORES=$ROOT/outputs/rrec_amazon/CDs_and_Vinyl/cot_candidate_lists_deepseek_v4_pro_low.rubric_deepseek_v4_pro.jsonl \
RUN_PREPARE=1 \
RUN_SFT=0 \
RUN_GRPO=0 \
bash scripts/run_rrec_training_from_candidate_lists.sh
```

This produces:

- `cot_judged_<RUN_NAME>.jsonl`
- `cot_scored_<RUN_NAME>.jsonl`
- `filtered_high_quality_cot_<RUN_NAME>.jsonl`
- `sft_<RUN_NAME>.jsonl`
- `grpo_<RUN_NAME>.jsonl`

To continue into training on the server:

```bash
CATEGORY=CDs_and_Vinyl \
RUN_NAME=deepseek_v4_pro_partial \
RUN_PREPARE=0 \
RUN_SFT=1 \
RUN_GRPO=1 \
SFT_NUM_TRAIN_EPOCHS=1 \
SFT_MAX_STEPS=-1 \
SFT_SAVE_STEPS=100 \
GRPO_MAX_STEPS=20 \
GRPO_NUM_GENERATIONS=4 \
bash scripts/run_rrec_training_from_candidate_lists.sh
```

For a short training sanity check, set `SFT_MAX_STEPS=1`,
`GRPO_MAX_STEPS=1`, and `GRPO_NUM_GENERATIONS=2`. If `ADAPTERS` is not set, the
script uses the latest SFT checkpoint under its configured `SFT_OUT`.

Rubric judging supports an API boundary. The R2ec script defaults to Kuaishou TokenVerse through a local OpenAI-compatible tunnel: `JUDGE_MODE=api API_PROVIDER=openai_compatible API_BASE_URL=http://127.0.0.1:18080/v1 API_MODEL=glm-5-1`.

To force an offline Zhipu-shaped mock:

```bash
JUDGE_MODE=api API_PROVIDER=zhipu_glm_mock bash scripts/run_phase1_full_rrec_amazon.sh
```

To use another OpenAI-compatible judge:

```bash
JUDGE_MODE=api \
API_PROVIDER=openai_compatible \
RUBRIC_JUDGE_API_BASE_URL=https://<host>/v1 \
RUBRIC_JUDGE_API_KEY=<key> \
RUBRIC_JUDGE_API_MODEL=<judge-model> \
bash scripts/run_phase1_full_rrec_amazon.sh
```

To call a GLM-compatible service running on your local machine from the AutoDL
server, keep the local service running and open a reverse SSH tunnel from the
Mac:

```bash
ssh -N \
  -o ExitOnForwardFailure=yes \
  -o PubkeyAuthentication=no \
  -o PreferredAuthentications=password \
  -R 127.0.0.1:18080:127.0.0.1:<LOCAL_GLM_PORT> \
  -p 22964 root@connect.cqa1.seetacloud.com
```

Then run the server pipeline against the tunneled endpoint. For an OpenAI-style
local endpoint:

```bash
JUDGE_MODE=api \
API_PROVIDER=openai_compatible \
API_BASE_URL=http://127.0.0.1:18080/v1 \
API_MODEL=<local-glm-model> \
API_WORKERS=1 \
bash scripts/run_phase1_full_rrec_amazon.sh
```

For a Zhipu/GLM-style local endpoint:

```bash
JUDGE_MODE=api \
API_PROVIDER=zhipu_glm_local \
API_BASE_URL=http://127.0.0.1:18080/api/paas/v4 \
API_MODEL=<local-glm-model> \
API_WORKERS=1 \
bash scripts/run_phase1_full_rrec_amazon.sh
```

For Kuaishou TokenVerse, run a local proxy on the Mac so the API key stays local:

```bash
export RUBIREC_API_BASE=https://tokenverse.corp.kuaishou.com/v1
export RUBIREC_MODEL=glm-5-1
export RUBIREC_API_KEY=<tokenverse-key>
python3 scripts/local_openai_proxy.py --host 127.0.0.1 --port 18081
```

In another Mac terminal, expose that local proxy to the AutoDL server:

```bash
ssh -N \
  -o ExitOnForwardFailure=yes \
  -o PubkeyAuthentication=no \
  -o PreferredAuthentications=password \
  -R 127.0.0.1:18080:127.0.0.1:18081 \
  -p 22964 root@connect.cqa1.seetacloud.com
```

Then run the server pipeline through the tunnel:

```bash
JUDGE_MODE=api \
API_PROVIDER=openai_compatible \
API_BASE_URL=http://127.0.0.1:18080/v1 \
API_MODEL=glm-5-1 \
API_WORKERS=1 \
bash scripts/run_phase1_full_rrec_amazon.sh
```

## Full Phase 1 Data Construction

```bash
cd /root/autodl-tmp/rec/aaai_pro
source configs/ml1m_qwen3_4b.env
MAX_USERS=1000 NUM_CANDIDATES=4 GAIN_EMBEDDER_MODE=qwen_hidden bash scripts/run_phase1_full_ml1m.sh
```

For a faster CPU-only scoring pass, use:

```bash
GAIN_EMBEDDER_MODE=lexical bash scripts/run_phase1_full_ml1m.sh
```

The main output files are:

- `outputs/ml1m/filtered_high_quality_cot.jsonl`
- `outputs/ml1m/rejected_cot.jsonl`
- `outputs/ml1m/sft.jsonl`
- `outputs/ml1m/grpo.jsonl`

## SFT

```bash
cd /root/autodl-tmp/rec/aaai_pro
source configs/ml1m_qwen3_4b.env
DATASET=$ROOT/outputs/ml1m/sft.jsonl \
OUT=$ROOT/checkpoints/qwen3_4b_sft_rubric_cot \
bash scripts/run_sft_qwen3_4b.sh
```

For a short training sanity check:

```bash
MAX_STEPS=1 DATASET=$ROOT/outputs/smoke/sft.jsonl OUT=$ROOT/checkpoints/smoke_sft bash scripts/run_sft_qwen3_4b.sh
```

## Rubric-Gated GRPO

```bash
cd /root/autodl-tmp/rec/aaai_pro
source configs/ml1m_qwen3_4b.env
MODEL=/root/autodl-tmp/modelscope_cache/models/Qwen/Qwen3-4B \
ADAPTERS=$ROOT/checkpoints/qwen3_4b_sft_rubric_cot/<run>/checkpoint-<step> \
DATASET=$ROOT/outputs/ml1m/grpo.jsonl \
OUT=$ROOT/checkpoints/qwen3_4b_grpo_rubric_gated \
bash scripts/run_grpo_qwen3_4b.sh
```

The online GRPO reward plugin registers:

- `rubric_format`
- `rubric_quality`
- `rubric_gated_gain`

The online gain reward now supports Qwen3-Embedding through:

```bash
RUBRIC_GAIN_EMBEDDER_MODE=qwen3_embedding \
QWEN3_EMBEDDING_MODEL=/root/autodl-tmp/modelscope_cache/models/Qwen/Qwen3-Embedding-0.6B \
bash scripts/run_grpo_qwen3_4b.sh
```

Set `RUBRIC_GAIN_EMBEDDER_MODE=lexical` for a cheaper debug-only reward. Offline
Phase 1 also supports `--embedder-mode lexical`, `--embedder-mode qwen_hidden`,
and `--embedder-mode qwen3_embedding`.

## Completed Server Run

The current AutoDL run completed a small true-API end-to-end pass on the R2ec/Amazon benchmark:

- True GLM-judged Phase 1 data:
  - `outputs/rrec_amazon/all/sft_trueapi.jsonl`
  - `outputs/rrec_amazon/all/grpo_trueapi.jsonl`
- SFT checkpoint:
  - superseded by Qwen3-4B runs; old small-model paths are intentionally omitted here.
- GRPO checkpoint:
  - superseded by Qwen3-4B runs; old small-model paths are intentionally omitted here.
- Frozen lexical full-set sanity:
  - `outputs/rrec_amazon/eval/*.json`
- Trained-reasoner generated-CoT full-set sanity:
  - `outputs/rrec_amazon/eval_reasoner/*_grpo_reasoner_10.json`
  - `outputs/rrec_amazon/eval_reasoner/*_grpo_reasoner_10_preds.jsonl`
- Qwen3-Embedding smoke outputs:
  - `outputs/rrec_amazon/Musical_Instruments/cot_scored_qwen3_embedding_smoke.jsonl`
  - `outputs/rrec_amazon/eval/Musical_Instruments_qwen3_embedding_smoke2.json`
  - `outputs/rrec_amazon/eval_reasoner/Musical_Instruments_grpo_reasoner_qwen3_embedding_smoke1.json`

Note: `rubric_cot_pipeline/rubric.py` now treats a judge score of exactly `1`
as the 1-5 rubric value `1`; only fractional values in `[0, 1)` are rescaled as
normalized scores. Older judged files created before this fix may need
regeneration before paper-quality experiments.

The Kuaishou TokenVerse judge should be run through one worker by default:

```bash
source configs/rrec_amazon_qwen3_4b.env
CATEGORY=Video_Games \
MAX_EXAMPLES=20 \
NUM_CANDIDATES=2 \
GAIN_EMBEDDER_MODE=lexical \
API_PROVIDER=openai_compatible \
API_BASE_URL=http://127.0.0.1:18080/v1 \
API_MODEL=glm-5-1 \
API_WORKERS=1 \
bash scripts/run_phase1_full_rrec_amazon.sh
```

SFT and GRPO:

```bash
DATASET=$ROOT/outputs/rrec_amazon/all/sft_trueapi.jsonl \
OUT=$ROOT/checkpoints/rrec_amazon_qwen3_4b_sft_trueapi \
NUM_TRAIN_EPOCHS=1 MAX_STEPS=-1 SAVE_STEPS=50 \
bash scripts/run_sft_qwen3_4b.sh

MAX_STEPS=5 NUM_GENERATIONS=2 \
DATASET=$ROOT/outputs/rrec_amazon/all/grpo_trueapi.jsonl \
ADAPTERS=$ROOT/checkpoints/rrec_amazon_qwen3_4b_sft_trueapi/<run>/checkpoint-<step> \
OUT=$ROOT/checkpoints/rrec_amazon_qwen3_4b_grpo_trueapi \
bash scripts/run_grpo_qwen3_4b.sh
```

Generated-CoT full-set sanity evaluation:

```bash
ADAPTER=$ROOT/checkpoints/rrec_amazon_qwen3_4b_grpo_trueapi/<run>/checkpoint-<step>
python scripts/evaluate_reasoner_fullset_proxy.py \
  --category Musical_Instruments \
  --split test \
  --max-examples 10 \
  --adapter "$ADAPTER" \
  --adapter-name grpo_trueapi \
  --output outputs/rrec_amazon/eval_reasoner/Musical_Instruments_grpo_reasoner_10.json \
  --predictions-output outputs/rrec_amazon/eval_reasoner/Musical_Instruments_grpo_reasoner_10_preds.jsonl
```

These sanity metrics use a lexical frozen proxy scorer, so they validate pipeline mechanics rather than final paper-quality recommendation performance. The next upgrade is to replace the proxy scorer with the intended frozen embedding/retrieval scorer and then scale Phase 1 beyond the current small true-API sample.

## Notes

- The ML-1M adapter uses the available data under `SOURCE_ROOT`.
- The R2ec/Amazon adapter reads HuggingFace datasets produced by the official R2ec `preprocess.py`.
- `target_item_text` is only used for evaluation/gain/reward, not for candidate generation.
