# Title-store-categories 与 external GLM CoT 严格配对结果 v2.0

## 当前状态

History-only 已完成训练、valid 选模和冻结 checkpoint 的一次 test。External GLM CoT 正在生成，相关结果完成前保留 `TBD`。

| 模型 | Train query | Valid 选中 checkpoint | Test NDCG@20 | Test HR@20 |
|---|---|---:|---:|---:|
| History-only | `H` | 249 | 0.118548 | 0.228188 |
| External GLM CoT | `H + tagged C` | TBD | TBD | TBD |

History-only 的四个 valid `NDCG@20` 分别为 checkpoint-83 `0.087604`、checkpoint-166 `0.088790`、checkpoint-249 `0.089020`、checkpoint-332 `0.088918`。选模规则据此冻结 checkpoint-249。该 checkpoint 在 1,341 条 test 样本和 12,000 个候选物品上得到 `NDCG@20=0.118548`、`HR@20=0.228188`。

## History 全 checkpoint test 诊断

用户要求对每个 checkpoint 运行 test，结果单独记录如下。该表只分析 epoch 变化，不参与 checkpoint 选择；正式结果仍采用 valid 选出的 checkpoint-249。

| Checkpoint | Test NDCG@20 | Test HR@20 |
|---:|---:|---:|
| 83 | 0.125025 | 0.234154 |
| 166 | 0.120831 | 0.231171 |
| 249 | 0.118548 | 0.228188 |
| 332 | 0.118216 | 0.228188 |

Test 指标随训练步数下降，但 valid 在 checkpoint-249 达到最高 `NDCG@20`。这组差异说明 valid 与 test 对 epoch 的排序不一致，不能根据上表把正式 checkpoint 改成 83。

## 唯一实验变量

两组使用相同的 train、valid、test 行、positive、顺序、Qwen3-Embedding-0.6B 初始权重、single-positive InfoNCE、seed `42`、batch `128`、4 epochs、学习率 `2e-5` 和 4096-token 上限。External-CoT query 只比对应 history query 多固定 `Recommendation reasoning:` 标签和一条完整 tagged CoT。

## 数据约束

History `H` 只含 title、`Store/artist/format` 和 categories。Rating、Description、Details、catalog stats、ASIN 和 `[TRUNCATED]` 不进入 history。Positive 与 12,000 个候选 item 保留完整 item text；字符截断参数固定为 `0`。

GLM-4.7 只读取 `H` 与 category。API thinking 关闭，输出格式为 `<think>/<answer>`，rating context 为 `no_rating`，temperature 为 `0.6`，top-p 为 `0.9`，每行生成一条 CoT。缺失行必须使用相同参数恢复；不使用手工补写。

## 完成后必须归档的证据

| 证据 | 计划路径 |
|---|---|
| History train、valid、test | `manu_src/datas/CDs_and_Vinyl/train_datas/title_store_categories_history_only_aligned_v2.0/` |
| External-CoT train、valid、test | `manu_src/datas/CDs_and_Vinyl/train_datas/title_store_categories_external_glm47_cot_aligned_v2.0/` |
| GLM train、valid、test raw 输出 | External-CoT 数据目录下的 `cot_api/` |
| 字段、配对、token、target-overlap 审计 | 本目录 `audits/` |
| 两组 `phase0_args.json` | 本目录 `training_args/` |
| 全部 valid checkpoint 指标 | 本目录 `valid/` |
| valid 选模记录 | 本目录 `selection/` |
| 冻结 checkpoint 的单次 test 指标 | 本目录 `test/` |
| prompt 快照 | `manu_src/scripts/prompts/external_glm47_history_cot_aligned_v2.py` |
| 数据和训练流水线 | `manu_src/scripts/pipelines/run_cds_aligned_history_external_cot_a100.sh` |

## Target overlap 诊断

全量审计会报告 CoT 与 held-out target title 的偶然文本重叠。该报告不参与 CoT 筛选、重生成、valid 选模或 test 决策，避免把 held-out target 引入 CoT 选择阶段。

## 与 v1.0 的关系

v1.0 保留历史 checkpoint 的真实数据和已知缺陷。v2.0 重新生成全部 CoT、重新训练两组模型，并增加独立 valid 选模、无截断 token 门禁和逐行配对审计。两个版本的指标需要分开记录。
