# Scripts Index

The scripts are currently kept in one flat directory so existing server commands
continue to work. Use this index to find the right entrypoint.

## Data Preparation

```text
prepare_rrec_amazon_examples.py       Convert RRec/Amazon rows to pipeline examples.
prepare_ml1m_examples.py              Convert ML-1M rows to pipeline examples.
make_phase0_embedder_dataset.py       Build base history->target embedding pairs.
make_cot_embedder_dataset.py          Build history/COT->target embedding pairs.
```

## CoT Generation And Rubric

```text
generate_cot_candidate_lists.py       API-based CoT candidate generation.
generate_cot_candidate_lists_local.py Local model CoT generation.
aggregate_cot_candidate_list_shards.py Merge generated shard JSONL files.
score_cot_candidate_lists.py          API/rule rubric score for candidate lists.
judge_cot_quality.py                  Score flat CoT quality JSONL files.
merge_candidate_list_rubric.py        Merge candidates and rubric scores.
```

## Gain, Selection, And Datasets

```text
compute_cot_gain.py                   Compute sim/NDCG CoT gain.
select_filtered_cot.py                Select top candidate per example.
select_top_percent_cot.py             Select global top-percent CoT rows.
finalize_cot_selection.py             Plot/filter final selected CoT rows.
make_sft_dataset.py                   Convert selected CoT rows to SFT messages.
make_grpo_dataset.py                  Convert scored rows to GRPO prompts.
```

## Embedding

```text
train_phase0_embedder.py              Qwen3 embedding contrastive training.
run_train_cds_embedding_tidal.sh      Train CDs base embedding on Tidal.
run_train_cds_cot_embedding_tidal.sh  Train CDs CoT-aware embedding on Tidal.
run_cds_qwen3_4b_cot_embedding_pipeline_tidal.sh
                                      Generate local CoT and train embedding.
eval_cds_embedding_base_tidal.sh      Evaluate base embedding.
eval_cds_embedding_trained_tidal.sh   Evaluate trained embedding.
```

## SFT / GRPO

```text
run_sft_qwen3_4b.sh                   ms-swift SFT launcher.
run_grpo_qwen3_4b.sh                  ms-swift GRPO launcher.
rubric_gated_reward.py                GRPO reward implementation.
```

## Evaluation

```text
evaluate_rrec_jsonl_fullset.py        Full-candidate embedding ranking eval.
evaluate_rrec_fullset_proxy.py        RRec dataset proxy eval.
evaluate_reasoner_fullset_proxy.py    Reasoner generation + ranking eval.
run_eval_reasoner_cds_multigpu_tidal.sh
                                      Multi-GPU reasoner eval wrapper.
aggregate_reasoner_eval_shards.py     Merge reasoner eval shards.
evaluate_proxy_retrieval.py           Lexical proxy retrieval eval.
```

## Pipeline Wrappers

```text
run_rrec_full_training_pipeline.sh             Full prepare/train pipeline.
run_rrec_training_from_candidate_lists.sh      Continue from candidate lists.
prepare_cds_from_cot_tidal.sh                  Build CDs data from existing CoT.
build_deepseek_one_glm52_top20_sft_grpo_tidal.sh
                                               DeepSeek one-CoT top20 SFT/GRPO builder.
run_phase1_full_rrec_amazon.sh                 Older RRec phase1 runner.
run_phase1_full_ml1m.sh                        Older ML-1M phase1 runner.
run_smoke_test.sh                              Small smoke test.
```

## Utilities

```text
local_openai_proxy.py                  Local OpenAI-compatible proxy.
test_glm_codeplan_api.py               GLM CodePlan API smoke test.
upload_cds_prepared_to_modelscope.sh   Upload prepared CDs artifacts.
run_generate_cds_glm47_low_one_cot.sh  GLM 4.7 one-CoT generation wrapper.
run_generate_musical_instruments_glm_codeplan.sh
                                       Musical Instruments GLM generation wrapper.
```

See `docs/PROJECT_STRUCTURE.md` for the repository-level layout and migration
rules.
