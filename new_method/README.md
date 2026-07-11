# GDR 第一阶段：成对检索器

本目录实现 story.md 中的第一阶段，代码与现有训练链路隔离。现阶段包含：

- core.py：CoT 提取、query 构造、排名增量和标签规则；
- make_cot_perturbations.py：不读取 target 的确定性多候选构造；
- score_paired_gain.py：使用冻结的 Qwen3 embedding 成对计算 history 与 history+think；
- build_paired_dataset.py：构造 history、good CoT、bad CoT、positive 和 hard negatives；
- audit_paired_dataset.py：训练前数据门禁；
- paired_loss.py：multi-positive InfoNCE 与 good/bad 顺序损失；
- train_paired_retriever.py：单卡 A100 三视图训练入口；
- summarize_oracle_gain.py：汇总逐样本增益、Oracle 上限和截断统计；
- extract_router_features.py：抽取 history embedding 与无 target 的初始检索置信度特征；
- router_model.py：路由模型、用户分组划分、阈值选择与路由指标；
- train_gain_router.py：在 train 内部 user-group valid 上训练 Logistic 或两层 MLP；
- eval_gain_router.py：冻结模型和阈值后评测 test；
- tests/：纯函数和损失测试。

## 固定口径

1. embedding query 只拼接 think 内容；
2. history 与 history+think 使用相同 target、候选集合和 seen-item mask；
3. target、positive、CoT 和 history 不允许出现 [TRUNCATED]；
4. 默认拒绝超长文本；复现旧实验时显式开启 allow-query-truncation 和 allow-item-truncation，并记录截断数量；
5. test 不参与标签阈值、checkpoint 和训练参数选择；
6. scorer checkpoint、query instruction、item-info 哈希和数据 split 写入产物元数据。

## 执行门禁

当前只提交代码。数据打分、数据生成、评测和训练命令需要先打印完整参数并获得用户确认。
