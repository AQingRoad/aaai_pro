# 脚本索引

真实实现文件放在分组子目录中。`scripts/run_sft_qwen3_4b.sh` 这类根目录
文件是软链接，用来兼容旧命令和服务器记录。改代码时优先修改分组目录中的
真实文件。

## 数据准备

```text
scripts/data/prepare_rrec_amazon_examples.py       将 RRec/Amazon 样本转成 pipeline JSONL。
scripts/data/prepare_ml1m_examples.py              将 ML-1M 样本转成 pipeline JSONL。
scripts/data/make_phase0_embedder_dataset.py       构建 history->target embedding 训练对。
scripts/data/make_cot_embedder_dataset.py          构建 history/CoT->target embedding 训练对。
scripts/data/rewrite_examples_with_item_metadata.py
                                                    用 item_info 和 summary sidecar 重写 user_history。
```

默认历史输入仍使用旧版 `title + rating`。需要在历史条目中加入
`item_info.jsonl` 里的 artist/store、category、description、details 等信息时，
在命令前设置 `HISTORY_METADATA_MODE=compact`；可用
`HISTORY_MAX_ITEM_CHARS=320` 调整每个历史 item 的 metadata 长度。
如果已经生成 item-level description summary sidecar，则使用
`HISTORY_METADATA_MODE=summary ITEM_METADATA_SUMMARY=/path/to/item_metadata_summary_*.jsonl`。
`summary` 模式只拼接 store/categories/summary/精选 details，不直接拼原始 description。

## CoT 生成和 Rubric

```text
scripts/cot/generate_cot_candidate_lists.py        通过 API 生成 CoT 候选列表。
scripts/cot/generate_cot_candidate_lists_local.py  通过本地模型生成 CoT 候选列表。
scripts/cot/generate_cot_candidates.py             旧版 flat CoT 生成入口。
scripts/cot/aggregate_cot_candidate_list_shards.py 合并多 shard 生成结果。
scripts/cot/score_cot_candidate_lists.py           对候选列表做 API/rule rubric 打分。
scripts/cot/judge_cot_quality.py                   对 flat CoT JSONL 做质量打分。
scripts/cot/merge_candidate_list_rubric.py         合并候选 CoT 和 rubric 分数。
```

## vLLM 推理

```text
scripts/inference/vllm_batch_infer_jsonl.py
                                                    通用 vLLM JSONL 批量推理，支持 description_summary 和 cot_generation。
scripts/inference/run_summarize_cds_item_descriptions_vllm_tidal.sh
                                                    用 Qwen3-32B/vLLM 为 CDs item_info 生成 description_summary sidecar。
```

## Gain、筛选和数据集

```text
scripts/selection/compute_cot_gain.py              计算 sim/NDCG CoT gain。
scripts/selection/select_filtered_cot.py           每个样本选出一个高质量 CoT。
scripts/selection/select_top_percent_cot.py        全局 top-percent CoT 筛选。
scripts/selection/finalize_cot_selection.py        对筛选结果画分布图并做最终过滤。
scripts/datasets/make_sft_dataset.py               将筛选 CoT 转成 SFT messages。
scripts/datasets/make_grpo_dataset.py              将 scored rows 转成 GRPO prompts。
```

## Embedding

```text
scripts/embedding/train_phase0_embedder.py              Qwen3 embedding 对比学习训练。
scripts/embedding/run_train_cds_embedding_tidal.sh      在 Tidal 上训练 CDs base embedding。
scripts/embedding/run_train_cds_cot_embedding_tidal.sh  在 Tidal 上训练 CDs CoT-aware embedding。
scripts/embedding/run_cds_qwen3_4b_cot_embedding_pipeline_tidal.sh
                                                        本地 Qwen3-4B 生成 CoT 后训练 embedding。
scripts/embedding/eval_cds_embedding_base_tidal.sh      评测 base embedding。
scripts/embedding/eval_cds_embedding_trained_tidal.sh   评测训练后的 embedding。
```

## SFT / GRPO

```text
scripts/train/run_sft_qwen3_4b.sh       ms-swift SFT 启动脚本。
scripts/train/run_grpo_qwen3_4b.sh      ms-swift GRPO 启动脚本。
scripts/train/rubric_gated_reward.py    GRPO 在线 reward 实现。
```

## 评测

```text
scripts/eval/evaluate_rrec_jsonl_fullset.py        全量候选 embedding ranking 评测。
scripts/eval/evaluate_rrec_fullset_proxy.py        RRec 数据集 proxy 评测。
scripts/eval/evaluate_reasoner_fullset_proxy.py    Transformers reasoner 生成 + ranking 评测。
scripts/eval/evaluate_reasoner_vllm_fullset.py     vLLM reasoner 生成 + ranking 评测。
scripts/eval/run_eval_reasoner_cds_multigpu_tidal.sh
                                                    多 GPU 分片 reasoner 评测封装。
scripts/eval/run_eval_checkpoints_vllm_tidal.sh    扫描并评测目录中的所有 checkpoint。
scripts/eval/aggregate_reasoner_eval_shards.py     合并 reasoner 评测分片。
scripts/eval/evaluate_proxy_retrieval.py           lexical proxy retrieval 评测。
```

## Pipeline 封装

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
scripts/pipelines/run_cds_meta_cot_generation_vllm_tidal.sh
                                                    summary -> metadata history -> local Qwen/vLLM one-CoT 生成。
```

## 工具

```text
scripts/utils/local_openai_proxy.py
scripts/utils/test_glm_codeplan_api.py
scripts/utils/upload_cds_prepared_to_modelscope.sh
```

数据放置规则和兼容策略见 `docs/PROJECT_STRUCTURE.md`。
