# Scripts Index

Implementation files live in grouped subdirectories. Root-level files such as
`scripts/run_sft_qwen3_4b.sh` are symlinks kept for old commands and server
notes. Prefer editing the grouped implementation path.

## Data Preparation

```text
scripts/data/prepare_rrec_amazon_examples.py       Convert RRec/Amazon rows to pipeline examples.
scripts/data/prepare_ml1m_examples.py              Convert ML-1M rows to pipeline examples.
scripts/data/make_phase0_embedder_dataset.py       Build base history->target embedding pairs.
scripts/data/make_cot_embedder_dataset.py          Build history/COT->target embedding pairs.
```

## CoT Generation And Rubric

```text
scripts/cot/generate_cot_candidate_lists.py        API-based CoT candidate generation.
scripts/cot/generate_cot_candidate_lists_local.py  Local model CoT generation.
scripts/cot/generate_cot_candidates.py             Older flat CoT generator.
scripts/cot/aggregate_cot_candidate_list_shards.py Merge generated shard JSONL files.
scripts/cot/score_cot_candidate_lists.py           API/rule rubric score for candidate lists.
scripts/cot/judge_cot_quality.py                   Score flat CoT quality JSONL files.
scripts/cot/merge_candidate_list_rubric.py         Merge candidates and rubric scores.
```

## Gain, Selection, And Datasets

```text
scripts/selection/compute_cot_gain.py              Compute sim/NDCG CoT gain.
scripts/selection/select_filtered_cot.py           Select top candidate per example.
scripts/selection/select_top_percent_cot.py        Select global top-percent CoT rows.
scripts/selection/finalize_cot_selection.py        Plot/filter final selected CoT rows.
scripts/datasets/make_sft_dataset.py               Convert selected CoT rows to SFT messages.
scripts/datasets/make_grpo_dataset.py              Convert scored rows to GRPO prompts.
```

## Embedding

```text
scripts/embedding/train_phase0_embedder.py              Qwen3 embedding contrastive training.
scripts/embedding/run_train_cds_embedding_tidal.sh      Train CDs base embedding on Tidal.
scripts/embedding/run_train_cds_cot_embedding_tidal.sh  Train CDs CoT-aware embedding on Tidal.
scripts/embedding/run_cds_qwen3_4b_cot_embedding_pipeline_tidal.sh
                                                        Generate local CoT and train embedding.
scripts/embedding/eval_cds_embedding_base_tidal.sh      Evaluate base embedding.
scripts/embedding/eval_cds_embedding_trained_tidal.sh   Evaluate trained embedding.
```

## SFT / GRPO

```text
scripts/train/run_sft_qwen3_4b.sh       ms-swift SFT launcher.
scripts/train/run_grpo_qwen3_4b.sh      ms-swift GRPO launcher.
scripts/train/rubric_gated_reward.py    GRPO reward implementation.
```

## Evaluation

```text
scripts/eval/evaluate_rrec_jsonl_fullset.py        Full-candidate embedding ranking eval.
scripts/eval/evaluate_rrec_fullset_proxy.py        RRec dataset proxy eval.
scripts/eval/evaluate_reasoner_fullset_proxy.py    Reasoner generation + ranking eval.
scripts/eval/run_eval_reasoner_cds_multigpu_tidal.sh
                                                    Multi-GPU reasoner eval wrapper.
scripts/eval/aggregate_reasoner_eval_shards.py     Merge reasoner eval shards.
scripts/eval/evaluate_proxy_retrieval.py           Lexical proxy retrieval eval.
```

## Pipeline Wrappers

```text
scripts/pipelines/run_rrec_full_training_pipeline.sh
scripts/pipelines/run_rrec_training_from_candidate_lists.sh
scripts/pipelines/prepare_cds_from_cot_tidal.sh
scripts/pipelines/build_deepseek_one_glm52_top20_sft_grpo_tidal.sh
scripts/pipelines/run_phase1_full_rrec_amazon.sh
scripts/pipelines/run_phase1_full_ml1m.sh
scripts/pipelines/run_smoke_test.sh
scripts/pipelines/run_generate_cds_glm47_low_one_cot.sh
scripts/pipelines/run_generate_musical_instruments_glm_codeplan.sh
```

## Utilities

```text
scripts/utils/local_openai_proxy.py
scripts/utils/test_glm_codeplan_api.py
scripts/utils/upload_cds_prepared_to_modelscope.sh
```

See `docs/PROJECT_STRUCTURE.md` for data placement rules and compatibility
policy.
