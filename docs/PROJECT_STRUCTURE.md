# Project Structure

This repository keeps runnable entrypoints stable while the pipeline is still
changing. Prefer adding new code according to this layout.

## Top-Level Layout

```text
rubric_cot_pipeline/        Core Python package shared by scripts.
scripts/                    Grouped runnable scripts plus compatibility links.
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

The actual implementation files live in grouped subdirectories under
`scripts/`. The repository also keeps root-level symlinks such as
`scripts/run_sft_qwen3_4b.sh` and `scripts/compute_cot_gain.py` so existing
server commands continue to work.

```text
Data preparation
  scripts/data/prepare_rrec_amazon_examples.py
  scripts/data/prepare_ml1m_examples.py
  scripts/data/make_phase0_embedder_dataset.py
  scripts/data/make_cot_embedder_dataset.py

CoT generation and judging
  scripts/cot/generate_cot_candidate_lists.py
  scripts/cot/generate_cot_candidate_lists_local.py
  scripts/cot/generate_cot_candidates.py
  scripts/cot/aggregate_cot_candidate_list_shards.py
  scripts/cot/score_cot_candidate_lists.py
  scripts/cot/judge_cot_quality.py
  scripts/cot/merge_candidate_list_rubric.py

Gain and selection
  scripts/selection/compute_cot_gain.py
  scripts/selection/select_filtered_cot.py
  scripts/selection/select_top_percent_cot.py
  scripts/selection/finalize_cot_selection.py

Dataset builders
  scripts/datasets/make_sft_dataset.py
  scripts/datasets/make_grpo_dataset.py

Embedding
  scripts/embedding/train_phase0_embedder.py
  scripts/embedding/eval_cds_embedding_base_tidal.sh
  scripts/embedding/eval_cds_embedding_trained_tidal.sh
  scripts/embedding/run_train_cds_embedding_tidal.sh
  scripts/embedding/run_train_cds_cot_embedding_tidal.sh
  scripts/embedding/run_cds_qwen3_4b_cot_embedding_pipeline_tidal.sh

SFT / GRPO training
  scripts/train/run_sft_qwen3_4b.sh
  scripts/train/run_grpo_qwen3_4b.sh
  scripts/train/rubric_gated_reward.py

Evaluation
  scripts/eval/evaluate_rrec_jsonl_fullset.py
  scripts/eval/evaluate_rrec_fullset_proxy.py
  scripts/eval/evaluate_reasoner_fullset_proxy.py
  scripts/eval/evaluate_proxy_retrieval.py
  scripts/eval/run_eval_reasoner_cds_multigpu_tidal.sh
  scripts/eval/aggregate_reasoner_eval_shards.py

End-to-end pipelines
  scripts/pipelines/prepare_cds_from_cot_tidal.sh
  scripts/pipelines/run_rrec_full_training_pipeline.sh
  scripts/pipelines/run_rrec_training_from_candidate_lists.sh
  scripts/pipelines/build_deepseek_one_glm52_top20_sft_grpo_tidal.sh
  scripts/pipelines/run_phase1_full_rrec_amazon.sh
  scripts/pipelines/run_phase1_full_ml1m.sh
  scripts/pipelines/run_smoke_test.sh
  scripts/pipelines/run_generate_cds_glm47_low_one_cot.sh
  scripts/pipelines/run_generate_musical_instruments_glm_codeplan.sh

Utilities
  scripts/utils/local_openai_proxy.py
  scripts/utils/test_glm_codeplan_api.py
  scripts/utils/upload_cds_prepared_to_modelscope.sh
```

## Compatibility Rule

Keep a symlink at `scripts/<old-name>` when a script moves into a grouped
directory. The training docs and old server commands use those root-level
paths, for example:

```text
scripts/train_phase0_embedder.py -> scripts/embedding/train_phase0_embedder.py
scripts/run_grpo_qwen3_4b.sh -> scripts/train/run_grpo_qwen3_4b.sh
```

When adding a new script, place the implementation in the matching grouped
directory. Add a root-level symlink only if users need a short stable command.

## Artifact Layout

```text
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

Use the grouped paths for code review and edits. Use the root-level symlinks
when you want the short commands shown in older run notes.
