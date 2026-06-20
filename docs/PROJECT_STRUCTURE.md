# Project Structure

This repository keeps runnable entrypoints stable while the pipeline is still
changing. Prefer adding new code according to this layout.

## Top-Level Layout

```text
rubric_cot_pipeline/        Core Python package shared by scripts.
scripts/                    Runnable CLI entrypoints and server pipelines.
configs/                    Example and local runtime environment configs.
github_artifacts/           Versioned small/medium datasets needed to reproduce runs.
data/                       Local raw or converted datasets. Ignored by git.
outputs/                    Local generated outputs. Ignored by git.
checkpoints/                Local model checkpoints. Ignored by git.
external/                   External repositories or downloaded source trees. Ignored by git.
prepared/                   Local prepared upload/staging files. Ignored by git.
docs/                       Project structure and operation notes.
```

## Data Policy

Use these rules when deciding where a file belongs:

```text
github_artifacts/
  Files that should travel with the GitHub repo:
  small evaluation JSONL, selected prepared datasets, rubric scores, candidate lists.

data/
  Raw or converted datasets that are large, regenerated, or server-specific.

outputs/
  Intermediate and final run outputs:
  judged CoT, scored CoT, selected SFT/GRPO JSONL, eval results, plots.

checkpoints/
  Trained models and embedding checkpoints. Never commit model weights.

configs/
  Template configs and non-secret defaults. Real API keys stay in ignored local files.
```

Do not commit API keys, model weights, or server-only cache directories.

## Script Groups

The current `scripts/` directory is flat for backward compatibility. Treat the
files as these logical groups:

```text
Data preparation
  prepare_rrec_amazon_examples.py
  prepare_ml1m_examples.py
  make_phase0_embedder_dataset.py
  make_cot_embedder_dataset.py

CoT generation and judging
  generate_cot_candidate_lists.py
  generate_cot_candidate_lists_local.py
  aggregate_cot_candidate_list_shards.py
  score_cot_candidate_lists.py
  judge_cot_quality.py
  merge_candidate_list_rubric.py

Gain and selection
  compute_cot_gain.py
  select_filtered_cot.py
  select_top_percent_cot.py
  finalize_cot_selection.py

Dataset builders
  make_sft_dataset.py
  make_grpo_dataset.py

Embedding
  train_phase0_embedder.py
  eval_cds_embedding_base_tidal.sh
  eval_cds_embedding_trained_tidal.sh
  run_train_cds_embedding_tidal.sh
  run_train_cds_cot_embedding_tidal.sh
  run_cds_qwen3_4b_cot_embedding_pipeline_tidal.sh

SFT / GRPO training
  run_sft_qwen3_4b.sh
  run_grpo_qwen3_4b.sh

Evaluation
  evaluate_rrec_jsonl_fullset.py
  evaluate_reasoner_fullset_proxy.py
  run_eval_reasoner_cds_multigpu_tidal.sh
  aggregate_reasoner_eval_shards.py

End-to-end pipelines
  prepare_cds_from_cot_tidal.sh
  run_rrec_full_training_pipeline.sh
  run_rrec_training_from_candidate_lists.sh
  build_deepseek_one_glm52_top20_sft_grpo_tidal.sh
```

## Migration Rule

Do not move a script until one of these is true:

```text
1. All callers have been updated and tested.
2. A compatibility wrapper remains at the old path.
```

This matters because many server commands call paths such as
`scripts/train_phase0_embedder.py` directly.

## Recommended Future Layout

When the pipeline stabilizes, migrate toward:

```text
scripts/
  data/
  cot/
  embedding/
  train/
  eval/
  pipelines/

rubric_cot_pipeline/
  embeddings.py
  prompts.py
  rubric.py
  judge_api.py
  io.py

github_artifacts/
  <CATEGORY>/
    cot/
    phase0/
    sft/
    grpo/
    rrec_eval/
```

For now, keep executable entrypoints at their current paths and use this
document plus `scripts/README.md` as the classification index.
