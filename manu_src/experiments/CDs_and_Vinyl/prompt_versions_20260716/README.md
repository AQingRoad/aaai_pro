# CDs_and_Vinyl 数据与提示词归档

## 归档判断

本目录冻结 2026-07-16 停止生成时已有的提示词、代码和实验报告。API、CoT training 与 embedding 数据位于 `../../../datas/CDs_and_Vinyl/archived_outputs_20260716/`，query-ablation 与严格配对 processed data 位于 `../../../datas/CDs_and_Vinyl/archived_processed_20260716/`，原始数据副本位于 `../../../datas/CDs_and_Vinyl/archived_source_data_20260716/`。提示词 pilot 的评测产物位于 `../../../eval_results/CDs_and_Vinyl/prompt_versions_20260716/`，本轮严格配对运行的远端快照位于相邻目录 `../aligned_tsc_glm47_frozen_20260716/`。

所有后续生成、训练和评测均已停止。严格配对 external-CoT 分支只完成 train CoT 和部分 valid CoT，尚未构造 CoT embedding pair，也没有 external-CoT checkpoint。此状态必须保留，不能把冻结的部分 valid 数据写成完整实验。

## 提示词与数据对应表

| 提示词版本 | 提示词快照 | 对应数据 | 模型与口径 | 状态 |
|---|---|---|---|---|
| GLM-4.7 external CoT legacy no-rating | `prompts` 中的 `external_glm47_history_cot_v1.py` | `archived_outputs_20260716/CDs_and_Vinyl/cot/api/cot_candidate_lists_glm47_meta_compact_no_all_ratings_observed_one_train_raw*`；由其构造的 `cot/training/*no_all_ratings*` | GLM-4.7；train 读取 metadata compact、无评分 history | 完整 train；其中 manual-filled 版本含 35 条人工补写，不能记为纯 API 数据 |
| GLM-4.7 external CoT legacy rating | 同一文件中的 `TAGGED_RATING_USER_INSTRUCTION` | `archived_outputs_20260716/CDs_and_Vinyl/cot/api/cot_candidate_lists_glm47_meta_compact_no_trunc_one_test_raw*`、`cds_glm47_meta_compact_no_trunc_one_test_full_target.jsonl` | GLM-4.7；旧 test history 含 rating/更丰富 metadata | 完整 test；与 no-rating train 存在输入口径差异 |
| GLM-4.7 external CoT legacy metadata-rich | `external_glm47_history_cot_v1.py` 的历史 metadata-rich 规则 | `cot_candidate_lists_glm47_meta_compact_one_train_raw*`、`cds_glm47_meta_compact_one_train_full_target.jsonl`；由其构造的 `phase0_embedder_cds_glm47_meta_compact_one_tagged_cot_only_full_target.jsonl` | GLM-4.7；旧 metadata-rich history | 历史结果，不作为 2026-07-12 后默认口径 |
| 严格配对 title_store_categories external CoT | `code_snapshot/rubric_cot_pipeline/prompts.py` 中 `COT_SYSTEM` 与 `build_history_analysis_prompt(..., output_format="tagged", rating_context="no_rating")` | `../aligned_tsc_glm47_frozen_20260716/remote_processed_data/cot_generation_inputs/*`、`cot/api/train_glm47_tagged*`、`valid_glm47_tagged*` | GLM-4.7；仅 title、store/artist/format、categories；seed 42；temperature 0.6；top_p 0.9 | train 10722 条完整；valid candidates 425 条、聚合文件 400 条；test 0 条；生成已停止 |
| title_store_categories history-only aligned v2 | 无 CoT 提示词 | `../aligned_tsc_glm47_frozen_20260716/remote_processed_data/history/{train,valid,test}.jsonl` | title、store/artist/format、categories；完整 positive；seed 42 | 训练与全 checkpoint 评测完成；valid 选择 checkpoint-249 |
| GLM-5.2 target-aware teacher v1.12 | `prompts/target_aware_history_grounded_cot_teacher.py` | `manu_src/datas/CDs_and_Vinyl/cot_datas/glm52_teacher_cot_v1.12_CDs_and_Vinyl_train_head_seed42_n2144/`；`train_datas/target_aware_history_grounded_cot_teacher_v1.12/` | GLM-5.2；teacher 可读取 private positive，但输出受 query 证据边界约束 | 2144 条 teacher 生成数据及 train/valid/test 数据已在原 manu_src 目录 |
| RREC aligned next-item reasoning v1.0 | `prompts/rrec_aligned_next_item_reasoning.py` | `manu_src/datas/CDs_and_Vinyl/train_datas/rrec_aligned_next_item_reasoning_v1.0/` | 提示词在训练时动态应用；positive 不进入学生输入 | 完整 train/valid/test，映射由该目录 manifest 记录 |
| Query-only CoT student | `prompts/query_only_cot_student.py` | 与 query-only SFT/推理脚本配套；当前归档未发现单独标注为该 prompt 的新生成 raw 数据 | student 只读取 query | 仅保留提示词和代码，不虚构数据对应关系 |
| GLM-5.2 英文通用 prompt | `prompts/history_cot_prompt_en.txt` | `prompt_versions_20260716/glm52_user_english_prompt_seed42_n20/` | GLM-5.2；20 条 target-free title_store_categories history；seed 42 | 完整 pilot |
| GLM-5.2 英文精简 prompt | `prompts/history_cot_prompt_en_concise.txt` | `prompt_versions_20260716/glm52_user_english_prompt_concise_seed42_n20/` | GLM-5.2；20 条 target-free history；seed 42 | 完整 pilot |
| 中文通用与中文精简 prompt | `prompts/history_cot_prompt_zh.txt`、`history_cot_prompt_zh_concise.txt` | 当前归档未发现对应正式生成文件 | 中文提示词备选版本 | 只保留提示词，不绑定未确认数据 |
| Codex/GPT-5.5 gated concise pilot | `prompts/archive/codex_gated_history_cot_pilot_20260713.txt` 与 schema | `prompt_versions_20260716/codex_gpt55_concise_prompt_regeneration_seed42/` | Codex/GPT-5.5；target-free；seed 42 | 7 条 pilot 完整 |
| Codex fallback transport v2 | `prompts/codex_target_free_cot_fallback_transport_v2.txt` 与 schema；语义提示词直接复用 `code_snapshot/rubric_cot_pipeline/prompts.py` | 没有正式数据 | v2 只负责批量 JSON 传输，不增加新的检索语义约束 | 代码和提示词已归档，未生成正式样本 |
| GLM-5.2 general regeneration pilot | 精确提示词原文未单独落盘；报告 `reports/cds_general_prompt_regeneration_pilot_20260713.md` 保存了约束与输出格式 | `prompt_versions_20260716/general_prompt_regeneration_seed42/` | GLM-5.2；7 条筛选样本；target-free；seed 42 | 数据完整，精确 prompt 原文缺失，不能声称可逐字复现 |
| GLM-5.2 concise general regeneration pilot | 精确提示词原文未单独落盘；报告 `reports/cds_concise_general_prompt_regeneration_pilot_20260713.md` 保存了门控规则 | `prompt_versions_20260716/concise_general_prompt_regeneration_seed42/` | GLM-5.2；7 条筛选样本；target-free；seed 42 | 数据完整，精确 prompt 原文缺失 |

## 早期文件的保守标注

`archived_outputs_20260716/CDs_and_Vinyl/cot/api/` 中还包含 DeepSeek v4、rubric judge、GLM-5.2 oracle、GLM-4.7 target-aware 早期文件。这些文件全部保留。部分早期调用没有把完整 prompt 或代码 commit 写进输出，当前只能从文件名、`generation_mode` 和报告确认模型与大类，无法恢复逐字提示词。对应关系写入 `DATA_PROMPT_MAP.json` 时标为 `exact_prompt_unavailable`，后续引用时不得把当前 `prompts.py` 当成当时的逐字快照。

## 冻结实验状态

- history-only aligned v2：checkpoint-249 由 valid NDCG@20 选出，官方 test `NDCG@20=0.1185478669`、`HR@20=0.2281879195`。
- external GLM-4.7 CoT aligned v2：train raw 10722 条完整，均来自 GLM-4.7；valid 在 425 条 candidate 时停止；test 尚未生成。
- train raw 审计：10722 个唯一 ID、10722 个唯一 CoT，tag、ASIN、截断、评分禁词和敏感 metadata 检查均为 0。
- target title 诊断：190 条 CoT 与 target title 精确重合，其中 168 条 title 已在 base history 中；其余 22 条主要为 `Greatest Hits` 等通用标题。该统计只用于诊断，不代表 target 字段进入提示词。

逐文件行数、字节数和 SHA256 见 `FILE_INVENTORY.json`。
