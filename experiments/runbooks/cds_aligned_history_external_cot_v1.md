# CDs title_store_categories 与 external-CoT 严格配对实验

## 实验判断

本实验只检验一个变量：在同一条 `title_store_categories` history 后追加一条由 GLM-4.7 生成的完整 tagged CoT，是否改变全候选检索指标。两组均重新训练，旧 checkpoint 不进入本次结果表。

## 配对数据

| Split | 行数 | History 组 query | External-CoT 组 query | Positive |
|---|---:|---|---|---|
| train | 10,722 | `H` | `H + Recommendation reasoning + C` | 逐字相同 |
| valid | 1,340 | `H` | `H + Recommendation reasoning + C` | 逐字相同 |
| test | 1,341 | `H` | `H + Recommendation reasoning + C` | 逐字相同 |

`H` 仅含标题、`Store/artist/format` 和类别路径。`H` 不含评分、Description、Details、catalog stats 和 ASIN。`C` 的提示词只读取 `H` 与 category；`target_item_id` 只用于结果回填配对，不进入 API message。Positive 保留 item_info 中的完整目标物品文本，两组逐字相同。

## 固定变量

| 模块 | 配置 |
|---|---|
| 初始权重 | Qwen3-Embedding-0.6B |
| Loss | single-positive InfoNCE |
| Seed | 42 |
| Epoch | 4 |
| Batch | 128 |
| Gradient accumulation | 1 |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Warmup ratio | 0.03 |
| InfoNCE temperature | 0.05 |
| Max length | 4096 |
| Attention | FlashAttention 2 |
| Checkpoint interval | 83 steps |
| 候选集合 | 同一份 item_info，12,000 个 item |
| Seen mask | 屏蔽 history item 与 pad item |

## External-CoT 配置

| 参数 | 值 |
|---|---|
| 模型 | glm-4.7 |
| API thinking | disabled |
| 输出 | `<think>` 与 `<answer>` tagged 格式 |
| Rating context | no_rating |
| Temperature | 0.6 |
| Top-p | 0.9 |
| 每条候选数 | 1 |
| 最大输出词数 | 1,024 |
| 最大新 token | 2,048 |
| Seed | 42 |
| API raw response | 不保存 |

## 训练前门禁

1. 三个 split 的 history 与 CoT 生成输入逐字对应，行数、顺序、用户、interaction、target 和 positive 均一致。
2. history query 中 `[TRUNCATED]`、ASIN、评分、Description、Details 和 catalog stats 的命中数均为 0。
3. CoT pair 中 `base_query` 必须逐字等于对应 history query；query 的唯一增量为固定标签和完整 tagged CoT。
4. Qwen tokenizer 对全部 query、positive 和 12,000 个候选 item 进行不截断计数。任一文本超过 4,096 token 时停止训练。
5. GLM 生成缺失任意一行时停止 pair 构建。先按相同参数恢复；持续失败项允许使用 Codex 兜底，Codex 只接收 `slot`、category 和脱敏 `user_history`，不得接收 user、interaction、target 或 positive 字段。兜底候选必须标记 `generation_mode=codex_cli_fallback` 和 `target_fields_exposed=false`。
6. 全量报告 CoT 与 held-out target title 的偶然文本重叠。该报告只作诊断，不按 target overlap 筛选或重生成 CoT，避免把 held-out target 引入 CoT 选择阶段。
7. 每个 split 完成首轮生成后比较输入与聚合输出行数。缺失行使用相同 prompt、模型、temperature、top-p、seed 和输出格式执行 `--resume`，最多恢复 20 轮；仍缺行时退出。
8. Codex 兜底直接复用 `rubric_cot_pipeline.prompts.COT_SYSTEM` 和 `build_history_analysis_prompt(..., output_format="tagged", rating_context="no_rating")`，不另写内容提示词。`experiments/prompts/codex_target_free_cot_fallback_transport_v2.txt` 只定义批量 JSON 传输，模型在每个 `cot` 字段中返回原提示词要求的完整 tagged 响应。`scripts/cot/generate_codex_fallback.py` 只生成 GLM 候选文件中缺失的 example ID；`scripts/cot/merge_cot_candidate_sources.py` 拒绝跨来源重复 ID，要求合并后行数逐 split 等于输入行数，并记录各生成模型的行数和 SHA-256。

## 选模和测试

两组分别评测全部 checkpoint 的 valid `NDCG@20`，各自选择最高 checkpoint；并列时选择 step 较小的 checkpoint。配置冻结后，每组只把 valid 选中项记为官方 test。同时在独立的 `test_all_checkpoints` 目录评测每个 checkpoint，结果只作诊断，不据 test 指标改选。主结果记录 `NDCG@20` 与 `HR@20`，同时保留 `@5`、`@10`、逐 checkpoint valid 和诊断 test 文件。

## 后续基准

本实验完成且所有门禁通过后，后续 external-CoT 检索实验继承本页固定变量。新实验若修改 CoT 选择器、提示词、生成模型、采样参数或 query 拼接形式，需要单独命名变量并保留本实验作为基线。
