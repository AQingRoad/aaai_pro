# CDs GLM-4.7 CoT Qwen3 Embedding 0.6B Batch 256 Runbook

This runbook records the fixed execution contract for the CDs_and_Vinyl
embedding run from the GLM-4.7 generated candidate-list file.

## Inputs

- Remote project root: `/root/autodl-tmp/rec/aaai_pro`
- Python environment: `/root/miniconda3/envs/swift`
- Base embedding model: `/root/autodl-tmp/modelscope_cache/models/Qwen/Qwen3-Embedding-0.6B`
- Raw candidate lists:
  `/root/autodl-tmp/rec/aaai_pro/outputs/rrec_amazon/CDs_and_Vinyl/cot_candidate_lists_glm47_meta_compact_one_train_raw.jsonl`
- Built training pairs:
  `/root/autodl-tmp/rec/aaai_pro/outputs/rrec_amazon/CDs_and_Vinyl/phase0_embedder_cds_glm47_meta_compact_one_tagged_cot_only.jsonl`
- Output checkpoint root:
  `/root/autodl-tmp/rec/aaai_pro/checkpoints/rrec_amazon_CDs_and_Vinyl/qwen3_embedding_0p6b_cds_glm47_meta_compact_one_tagged_cot_only_len4096_batch256_epoch5`

## Length Statistics

Use `COT_TEXT_MODE=tagged`, `MAX_COT_CHARS=0`, and `MAX_ITEM_CHARS=0` for this
run. The length statistics below explain what the previous 1200/1400 character
thresholds would have truncated:

- Source rows: `10722`
- Tagged CoT compacted character length: min `341`, mean `957.19`,
  p50 `910`, p90 `1287`, p95 `1444.95`, p99 `1732.58`, max `2241`
- Tagged CoT truncation at 1200 chars: `1568/10722`, rate `14.6241%`
- Positive item actual character length: min `69`, mean `418.88`,
  p50 `442`, p90 `638.9`, p95 `665`, p99 `867`, max `1397`
- Positive item truncation at 1400 chars: `0/10722`, rate `0%`
- All rows already contain `target_item_text`; the current script uses that
  field directly for positive item text.
- Pair construction: `INCLUDE_HISTORY=0`, `INCLUDE_COT=1`, so each source row
  creates one history-plus-tagged-CoT pair. Expected training rows: `10722`.
- Qwen3-Embedding-0.6B tokenizer length for query without instruction:
  min `172`, mean `986.18`, p50 `636`, p90 `2617`, p95 `3028.95`,
  p99 `3437`, max `3805`.
- Qwen3-Embedding-0.6B tokenizer length after adding query instruction:
  min `199`, mean `1013.18`, p50 `663`, p90 `2644`, p95 `3055.95`,
  p99 `3464`, max `3832`.
- Old `EMBEDDER_MAX_LENGTH=2048` truncated `1487/10722` queries after adding
  instruction, rate `13.8687%`. This run uses `EMBEDDER_MAX_LENGTH=4096`; the
  query max token length is `3832`, so query truncation is `0%`. Positive item
  max token length is `470`, so positive item truncation is also `0%`.

## Mandatory Confirmation Gate

Before training, run the print-only command and send the annotated parameter
table to the user. Each `KEY=value` line must include a short Chinese comment
that explains what the parameter controls. Start training only after the user
confirms the values.

```bash
cd /root/autodl-tmp/rec/aaai_pro

ROOT=/root/autodl-tmp/rec/aaai_pro \
VENV=/root/miniconda3/envs/swift \
PYTHON_BIN=/root/miniconda3/envs/swift/bin/python \
COT_CANDIDATE_LISTS=/root/autodl-tmp/rec/aaai_pro/outputs/rrec_amazon/CDs_and_Vinyl/cot_candidate_lists_glm47_meta_compact_one_train_raw.jsonl \
COT_TEXT_MODE=tagged \
INCLUDE_HISTORY=0 \
INCLUDE_COT=1 \
COT_EMBEDDER_DATASET=/root/autodl-tmp/rec/aaai_pro/outputs/rrec_amazon/CDs_and_Vinyl/phase0_embedder_cds_glm47_meta_compact_one_tagged_cot_only.jsonl \
BASE_EMBEDDING_MODEL=/root/autodl-tmp/modelscope_cache/models/Qwen/Qwen3-Embedding-0.6B \
EMBEDDER_OUT=/root/autodl-tmp/rec/aaai_pro/checkpoints/rrec_amazon_CDs_and_Vinyl/qwen3_embedding_0p6b_cds_glm47_meta_compact_one_tagged_cot_only_len4096_batch256_epoch5 \
EMBEDDER_CUDA_VISIBLE_DEVICES=0 \
EMBEDDER_NPROC_PER_NODE=1 \
EMBEDDER_BATCH_SIZE=256 \
EMBEDDER_GRAD_ACCUM=1 \
EMBEDDER_MAX_LENGTH=4096 \
MAX_COT_CHARS=0 \
MAX_ITEM_CHARS=0 \
EMBEDDER_EPOCHS=5 \
EMBEDDER_MAX_STEPS=-1 \
EMBEDDER_LR=3e-6 \
EMBEDDER_SAVE_STEPS=auto \
EMBEDDER_TORCH_DTYPE=bfloat16 \
EMBEDDER_GRADIENT_CHECKPOINTING=auto \
EMBEDDER_CROSS_GPU_NEGATIVES=0 \
FORCE_REBUILD_DATASET=1 \
EMBEDDER_PRINT_ARGS_ONLY=1 \
bash scripts/embedding/run_train_cds_cot_embedding_tidal.sh
```

## Training Command After Confirmation

Use the same variables, remove `EMBEDDER_PRINT_ARGS_ONLY=1`, and set
`FORCE_REBUILD_DATASET=0` if the printed command already built the training
pair file.

```bash
cd /root/autodl-tmp/rec/aaai_pro

ROOT=/root/autodl-tmp/rec/aaai_pro \
VENV=/root/miniconda3/envs/swift \
PYTHON_BIN=/root/miniconda3/envs/swift/bin/python \
COT_CANDIDATE_LISTS=/root/autodl-tmp/rec/aaai_pro/outputs/rrec_amazon/CDs_and_Vinyl/cot_candidate_lists_glm47_meta_compact_one_train_raw.jsonl \
COT_TEXT_MODE=tagged \
INCLUDE_HISTORY=0 \
INCLUDE_COT=1 \
COT_EMBEDDER_DATASET=/root/autodl-tmp/rec/aaai_pro/outputs/rrec_amazon/CDs_and_Vinyl/phase0_embedder_cds_glm47_meta_compact_one_tagged_cot_only.jsonl \
BASE_EMBEDDING_MODEL=/root/autodl-tmp/modelscope_cache/models/Qwen/Qwen3-Embedding-0.6B \
EMBEDDER_OUT=/root/autodl-tmp/rec/aaai_pro/checkpoints/rrec_amazon_CDs_and_Vinyl/qwen3_embedding_0p6b_cds_glm47_meta_compact_one_tagged_cot_only_len4096_batch256_epoch5 \
EMBEDDER_CUDA_VISIBLE_DEVICES=0 \
EMBEDDER_NPROC_PER_NODE=1 \
EMBEDDER_BATCH_SIZE=256 \
EMBEDDER_GRAD_ACCUM=1 \
EMBEDDER_MAX_LENGTH=4096 \
MAX_COT_CHARS=0 \
MAX_ITEM_CHARS=0 \
EMBEDDER_EPOCHS=5 \
EMBEDDER_MAX_STEPS=-1 \
EMBEDDER_LR=3e-6 \
EMBEDDER_SAVE_STEPS=auto \
EMBEDDER_TORCH_DTYPE=bfloat16 \
EMBEDDER_GRADIENT_CHECKPOINTING=auto \
EMBEDDER_CROSS_GPU_NEGATIVES=0 \
FORCE_REBUILD_DATASET=0 \
bash scripts/embedding/run_train_cds_cot_embedding_tidal.sh
```

## Evaluation After Training

Evaluate the produced checkpoints on the CDs_and_Vinyl test split.

```bash
cd /root/autodl-tmp/rec/aaai_pro

ROOT=/root/autodl-tmp/rec/aaai_pro \
VENV=/root/miniconda3/envs/swift \
PYTHON_BIN=/root/miniconda3/envs/swift/bin/python \
CHECKPOINT_ROOT=/root/autodl-tmp/rec/aaai_pro/checkpoints/rrec_amazon_CDs_and_Vinyl/qwen3_embedding_0p6b_cds_glm47_meta_compact_one_tagged_cot_only_len4096_batch256_epoch5 \
EVAL_DIR=/root/autodl-tmp/rec/aaai_pro/outputs/rrec_amazon/eval/qwen3_embedding_0p6b_cds_glm47_meta_compact_one_tagged_cot_only_len4096_batch256_epoch5_test \
SPLIT=test \
MAX_EXAMPLES=0 \
CUDA_VISIBLE_DEVICES=0 \
EMBEDDING_DEVICE=cuda:0 \
EMBEDDING_BATCH_SIZE=256 \
EMBEDDING_MAX_LENGTH=4096 \
FORCE_EVAL=1 \
bash scripts/embedding/run_eval_cds_embedding_checkpoints_tidal.sh
```

## OOM Result

The direct `EMBEDDER_MAX_LENGTH=4096`, `EMBEDDER_BATCH_SIZE=256`,
`EMBEDDER_GRAD_ACCUM=1`, `EMBEDDER_EPOCHS=5` run failed on the first forward
pass on an RTX 5090 32GB GPU. The error reported total GPU memory `31.36GiB`,
free memory `1.88GiB`, process memory `29.47GiB`, and an additional allocation
request of `6.95GiB`. No checkpoint was produced; only `phase0_args.json` was
written under the checkpoint root.

The follow-up `EMBEDDER_BATCH_SIZE=128`, `EMBEDDER_GRAD_ACCUM=2` run also
failed on the first forward pass. The error reported total GPU memory
`31.36GiB`, free memory `1.59GiB`, process memory `29.76GiB`, and an
additional allocation request of `3.48GiB`. No checkpoint was produced for this
configuration either.

The next `EMBEDDER_BATCH_SIZE=64`, `EMBEDDER_GRAD_ACCUM=4` run passed the
initial forward but failed during backward recomputation of attention. The
error reported total GPU memory `31.36GiB`, free memory `1.57GiB`, process
memory `29.78GiB`, and an additional allocation request of `1.74GiB`. No
optimizer step completed, and no checkpoint was produced.

Recommended retry while keeping effective batch size 256:

```bash
cd /root/autodl-tmp/rec/aaai_pro

ROOT=/root/autodl-tmp/rec/aaai_pro \
VENV=/root/miniconda3/envs/swift \
PYTHON_BIN=/root/miniconda3/envs/swift/bin/python \
COT_CANDIDATE_LISTS=/root/autodl-tmp/rec/aaai_pro/outputs/rrec_amazon/CDs_and_Vinyl/cot_candidate_lists_glm47_meta_compact_one_train_raw.jsonl \
COT_TEXT_MODE=tagged \
INCLUDE_HISTORY=0 \
INCLUDE_COT=1 \
COT_EMBEDDER_DATASET=/root/autodl-tmp/rec/aaai_pro/outputs/rrec_amazon/CDs_and_Vinyl/phase0_embedder_cds_glm47_meta_compact_one_tagged_cot_only.jsonl \
BASE_EMBEDDING_MODEL=/root/autodl-tmp/modelscope_cache/models/Qwen/Qwen3-Embedding-0.6B \
EMBEDDER_OUT=/root/autodl-tmp/rec/aaai_pro/checkpoints/rrec_amazon_CDs_and_Vinyl/qwen3_embedding_0p6b_cds_glm47_meta_compact_one_tagged_cot_only_len4096_batch32_accum8_epoch5 \
EMBEDDER_CUDA_VISIBLE_DEVICES=0 \
EMBEDDER_NPROC_PER_NODE=1 \
EMBEDDER_BATCH_SIZE=32 \
EMBEDDER_GRAD_ACCUM=8 \
EMBEDDER_MAX_LENGTH=4096 \
MAX_COT_CHARS=0 \
MAX_ITEM_CHARS=0 \
EMBEDDER_EPOCHS=5 \
EMBEDDER_MAX_STEPS=-1 \
EMBEDDER_LR=3e-6 \
EMBEDDER_SAVE_STEPS=auto \
EMBEDDER_TORCH_DTYPE=bfloat16 \
EMBEDDER_GRADIENT_CHECKPOINTING=auto \
EMBEDDER_CROSS_GPU_NEGATIVES=0 \
FORCE_REBUILD_DATASET=0 \
bash scripts/embedding/run_train_cds_cot_embedding_tidal.sh
```

Current retry status on 2026-07-01:

- `EMBEDDER_BATCH_SIZE=32`, `EMBEDDER_GRAD_ACCUM=8`, and
  `EMBEDDER_GLOBAL_BATCH_SIZE=256` have completed at least `step 4/210`.
- This configuration did not OOM during the first forward/backward path.
- A spot `nvidia-smi` check during training reported GPU memory used
  `29174MiB`, memory free `2938MiB`, and GPU utilization `100%`.
- Based on the first four optimizer steps, the training run is expected to take
  roughly `3.2` to `3.8` hours before evaluation.

## tmux Run With Auto Shutdown

The reusable launcher is:

```bash
cd /root/autodl-tmp/rec/aaai_pro
bash scripts/embedding/run_cds_qwen3_embedding_tmux_shutdown.sh
```

Defaults:

- tmux session: `cds_qwen3_embed_4096`
- log directory: `/root/autodl-tmp/rec/aaai_pro/logs/embedding`
- train config: `EMBEDDER_BATCH_SIZE=32`, `EMBEDDER_GRAD_ACCUM=8`,
  `EMBEDDER_MAX_LENGTH=4096`, `EMBEDDER_EPOCHS=5`
- after training succeeds, evaluate test split checkpoints under the training
  checkpoint root
- after the workflow exits, run `shutdown -h +1`; this also happens if training
  or evaluation fails, so the 5090 server does not remain idle

The 2026-07-02 tmux run was started with:

- session: `cds_qwen3_embed_4096`
- log: `/root/autodl-tmp/rec/aaai_pro/logs/embedding/cds_qwen3_embed_4096_20260702_000837.log`
- checkpoint root:
  `/root/autodl-tmp/rec/aaai_pro/checkpoints/rrec_amazon_CDs_and_Vinyl/qwen3_embedding_0p6b_cds_glm47_meta_compact_one_tagged_cot_only_len4096_batch32_accum8_epoch5`

The earlier orphaned non-tmux run did not leave a checkpoint, only
`phase0_args.json`; the tmux run therefore starts training from the base model.

## Final Test Results

The tmux workflow completed at `2026-07-02T05:06:37+08:00` with:

- `TRAIN_EXIT_STATUS=0`
- `EVAL_EXIT_STATUS=0`
- `WORKFLOW_EXIT_STATUS=0`
- log: `/root/autodl-tmp/rec/aaai_pro/logs/embedding/cds_qwen3_embed_4096_20260702_000837.log`
- eval dir:
  `/root/autodl-tmp/rec/aaai_pro/outputs/rrec_amazon/eval/qwen3_embedding_0p6b_cds_glm47_meta_compact_one_tagged_cot_only_len4096_batch32_accum8_epoch5_test`

Each checkpoint was evaluated on `1341` test examples with `12001` candidate
items.

| checkpoint | HR@5 | HR@10 | HR@20 | NDCG@5 | NDCG@10 | NDCG@20 | mean rank | median rank |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| checkpoint-42 | 0.022371 | 0.044743 | 0.070843 | 0.012622 | 0.019886 | 0.026394 | 3860.0045 | 3082 |
| checkpoint-84 | 0.026846 | 0.043997 | 0.071588 | 0.014423 | 0.019976 | 0.026966 | 3739.0858 | 2856 |
| checkpoint-126 | 0.026846 | 0.046234 | 0.073826 | 0.014116 | 0.020336 | 0.027317 | 3704.8427 | 2831 |
| checkpoint-168 | 0.027591 | 0.046234 | 0.074571 | 0.014796 | 0.020734 | 0.027892 | 3697.0917 | 2816 |
| checkpoint-210 | 0.026100 | 0.044743 | 0.072334 | 0.013911 | 0.019958 | 0.026962 | 3702.3870 | 2803 |

Best checkpoint by `NDCG@20`: `checkpoint-168`.

The auto-shutdown step failed in this run because the script used
`/sbin/shutdown`, while the current container exposes `/usr/bin/shutdown`. The
launcher has been updated to resolve `shutdown` from `PATH`, then fallback to
`poweroff` or `halt`.

## Diagnosis

The weak result is most likely caused by input and objective mismatch.

1. The training dataset uses only `history + tagged CoT` queries:
   `INCLUDE_HISTORY=0`, `INCLUDE_COT=1`, `COT_TEXT_MODE=tagged`.
   Evaluation in `scripts/eval/evaluate_rrec_jsonl_fullset.py` rebuilds test
   queries from user history only and does not append CoT. The result JSON also
   records `history_metadata_mode=none`, while the training examples contain
   richer history metadata such as store, format, categories, description,
   details, and catalog statistics. The model therefore sees long
   reasoning-rich queries during training and short metadata-poor queries at
   test time.

2. The contrastive objective has a small negative pool. The run logs show
   `candidate_docs=32` and `explicit_negatives=0`; each update only contrasts a
   query against the current batch positives. Test ranking searches across
   `12001` candidate items, so random in-batch negatives are not hard enough for
   item-level recommendation.

3. More epochs did not fix the issue. Test metrics peak at `checkpoint-168`,
   then decline at `checkpoint-210`, so the problem is not simply insufficient
   training time.

4. The evaluation script does not mask items already present in user history.
   If the intended RRec protocol excludes seen items, this implementation can
   lower next-item ranking metrics because history items can occupy high ranks.

Recommended ablations before the next long run:

- Evaluate the base `Qwen3-Embedding-0.6B` on the same test set and compare it
  with `checkpoint-168`.
- Re-evaluate `checkpoint-168` with the same metadata mode as training, such as
  `HISTORY_METADATA_MODE=compact`, if the test artifacts support it.
- Evaluate a test-time `history + generated CoT` query path. If metrics improve,
  the main bottleneck is the query-form mismatch.
- Train a mixed dataset containing both `history` and `history + CoT` pairs, or
  use CoT dropout so the model keeps performance on plain history queries.
- Add explicit negatives, preferably category/artist/genre hard negatives, and
  compare with the current in-batch-only objective.

## Test CoT History Metadata Length

The train CoT file
`outputs/rrec_amazon/CDs_and_Vinyl/cot_candidate_lists_glm47_meta_compact_one_train_raw.jsonl`
records `history_metadata_mode=compact` and `history_max_item_chars=0`.
Therefore test CoT generation should keep `HISTORY_MAX_ITEM_CHARS=0` to match
train.

On the `1341` test examples, there are `6662` history-item occurrences and
`2015` unique history items. For untruncated compact metadata per history item:

- mean chars: `487.74`
- p50: `472`
- p75: `715`
- p90: `758`
- p95: `776`
- p99: `803`
- max: `954`

If `HISTORY_MAX_ITEM_CHARS=320`, `4368/6662` history-item metadata blocks are
truncated, a `65.5659%` truncation rate. This setting is too aggressive for
train/test alignment.

Prompt-level chars on test with compact metadata:

| HISTORY_MAX_ITEM_CHARS | mean | p50 | p90 | p95 | p99 | max |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0 | 2695.49 | 1839 | 6435 | 9465 | 11666.60 | 13427 |
| 320 | 1743.69 | 1144 | 4093 | 6478 | 6996.80 | 7119 |
| 640 | 2531.62 | 1726 | 5978 | 8812 | 10747.80 | 12032 |
| 800 | 2694.05 | 1839 | 6435 | 9465 | 11666.60 | 13427 |
